#include "alkali.h"
#include "generic_handlers.h"

/*
 * L3 Forwarding (single-stage): Exact Match + LPM-lite
 *
 * Contract:
 *   - Upstream prepends ingress_meta_t.
 *   - This stage extracts ingress meta + Ethernet + IPv4.
 *   - Performs EM or LPM-lite lookup based on cfg.mode.
 *   - Optionally rewrites MAC.
 *   - Emits l3sw_meta_t for dispatcher.
 *
 * Mode:
 *   0 = EM (exact dst_ip)
 *   1 = LPM-lite (/32,/24,/16,/8 priority)
 *
 * Notes:
 *   - No loops / switch.
 *   - MAC split into 32+16 bits to avoid width mixing.
 */

/* ---------------- Ingress metadata ---------------- */
struct ingress_meta_t {
  BITS_FIELD(32, ingress_port);
  BITS_FIELD(32, r0);
  BITS_FIELD(32, r1);
  BITS_FIELD(32, r2);
};

/* ---------------- Ethernet + IPv4 ---------------- */
struct eth_header_t {
  BITS_FIELD(32, dst_mac_1);
  BITS_FIELD(16, dst_mac_2);
  BITS_FIELD(32, src_mac_1);
  BITS_FIELD(16, src_mac_2);
  BITS_FIELD(16, ether_type);
};

struct ip_header_t {
  BITS_FIELD(16, misc);
  BITS_FIELD(16, length);
  BITS_FIELD(16, identification);
  BITS_FIELD(16, fragment_offset);
  BITS_FIELD(16, TTL_transport);
  BITS_FIELD(16, checksum);
  BITS_FIELD(32, source_ip);
  BITS_FIELD(32, dst_ip);
  BITS_FIELD(32, options);
};

/* ---------------- Routing tables ---------------- */
/* value = (egress_port, valid-bit) */
struct rt_val_t {
  BITS_FIELD(32, egress_port);
  BITS_FIELD(32, valid);
};

ak_TABLE(2048, BITS(32), struct rt_val_t, ) em_route;
ak_TABLE(2048, BITS(32), struct rt_val_t, ) lpm_32;
ak_TABLE(2048, BITS(32), struct rt_val_t, ) lpm_24;
ak_TABLE(2048, BITS(32), struct rt_val_t, ) lpm_16;
ak_TABLE(2048, BITS(32), struct rt_val_t, ) lpm_8;

/* ---------------- Global config ---------------- */
struct cfg_t {
  BITS_FIELD(32, mode);         /* 0=EM, 1=LPM-lite */
  BITS_FIELD(32, mac_rewrite);  /* 0/1 */
};
ak_TABLE(64, BITS(32), struct cfg_t, ) cfg_table;

/* ---------------- Per-port stats ---------------- */
struct port_stats_t {
  BITS_FIELD(32, forwarded);
  BITS_FIELD(32, dropped);
};
ak_TABLE(256, BITS(32), struct port_stats_t, ) port_stats;

/* ---------------- Dispatcher metadata ---------------- */
/* flags: bit0=drop, bit1=mac_rewritten, bit2=mode */
struct l3sw_meta_t {
  BITS_FIELD(32, egress_port);
  BITS_FIELD(32, flags);
  BITS_FIELD(32, ingress_port);
  BITS_FIELD(32, dst_ip);
};

/* Branchless MAC rewrite */
static inline void
rewrite_mac_branchless(struct eth_header_t *eth,
                       BITS(32) egress_port,
                       BITS(32) mac_rewrite) {
  BITS(32) m32   = mac_rewrite;
  BITS(32) inv32 = (BITS(32))1 - m32;

  BITS(16) m16   = (BITS(16))m32;
  BITS(16) inv16 = (BITS(16))1 - m16;

  BITS(32) new_mac_1 = (BITS(32))0x02000000;
  BITS(16) new_mac_2 = (BITS(16))egress_port;

  eth->dst_mac_1 = eth->dst_mac_1 * inv32 + new_mac_1 * m32;
  eth->src_mac_1 = eth->src_mac_1 * inv32 + new_mac_1 * m32;

  {
    BITS(16) dst2 = (BITS(16))eth->dst_mac_2;
    BITS(16) src2 = (BITS(16))eth->src_mac_2;

    dst2 = dst2 * inv16 + new_mac_2 * m16;
    src2 = src2 * inv16 + new_mac_2 * m16;

    eth->dst_mac_2 = dst2;
    eth->src_mac_2 = src2;
  }
}

void NET_RECV__process_packet(buf_t packet) {

  /* 0) Ingress meta */
  struct ingress_meta_t inmeta;
  bufextract(packet, (void*)&inmeta);
  BITS(32) ingress_port = inmeta.ingress_port;

  /* 1) Parse L2/L3 */
  struct eth_header_t eth;
  struct ip_header_t ip;
  bufextract(packet, (void*)&eth);
  bufextract(packet, (void*)&ip);
  BITS(32) dst_ip = ip.dst_ip;

  /* 2) Load config */
  BITS(32) key0 = 0;
  struct cfg_t cfg;
  table_lookup(&cfg_table, &key0, &cfg);

  BITS(32) use_lpm = cfg.mode;
  BITS(32) use_em  = (BITS(32))1 - use_lpm;

  /* 3A) EM */
  struct rt_val_t emv;
  table_lookup(&em_route, &dst_ip, &emv);
  BITS(32) em_port =
      emv.egress_port * emv.valid +
      ingress_port    * ((BITS(32))1 - emv.valid);

  /* 3B) LPM-lite */
  BITS(32) k32 = dst_ip;
  BITS(32) k24 = dst_ip & (BITS(32))0xFFFFFF00;
  BITS(32) k16 = dst_ip & (BITS(32))0xFFFF0000;
  BITS(32) k8  = dst_ip & (BITS(32))0xFF000000;

  struct rt_val_t v32, v24, v16, v8;
  table_lookup(&lpm_32, &k32, &v32);
  table_lookup(&lpm_24, &k24, &v24);
  table_lookup(&lpm_16, &k16, &v16);
  table_lookup(&lpm_8,  &k8,  &v8);

  BITS(32) pick32 = v32.valid;
  BITS(32) pick24 = ((BITS(32))1 - pick32) * v24.valid;
  BITS(32) pick16 = ((BITS(32))1 - pick32 - pick24) * v16.valid;
  BITS(32) pick8  = ((BITS(32))1 - pick32 - pick24 - pick16) * v8.valid;
  BITS(32) pickfb = (BITS(32))1 - pick32 - pick24 - pick16 - pick8;

  BITS(32) lpm_port =
      v32.egress_port * pick32 +
      v24.egress_port * pick24 +
      v16.egress_port * pick16 +
      v8.egress_port  * pick8  +
      ingress_port    * pickfb;

  /* 3C) Mode select */
  BITS(32) egress_port = em_port * use_em + lpm_port * use_lpm;

  /* 4) MAC rewrite */
  rewrite_mac_branchless(&eth, egress_port, cfg.mac_rewrite);
  bufemit(packet, (void*)&eth);

  /* 5) Stats */
  struct port_stats_t ps;
  table_lookup(&port_stats, &egress_port, &ps);
  ps.forwarded = ps.forwarded + (BITS(32))1;
  table_update(&port_stats, &egress_port, &ps);

  /* 6) Emit dispatcher meta */
  struct l3sw_meta_t outm;
  outm.egress_port = egress_port;
  outm.flags = (cfg.mac_rewrite << 1) | (use_lpm << 2);
  outm.ingress_port = ingress_port;
  outm.dst_ip = dst_ip;

  bufemit(packet, (void*)&outm);

  EXT__NET_SEND__net_send(packet);
}