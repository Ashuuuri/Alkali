#include "alkali.h"
#include "generic_handlers.h"

/* ---------------- Upstream ingress metadata ---------------- */
struct ingress_meta_t {
  BITS_FIELD(32, ingress_port);
  BITS_FIELD(32, r0);
  BITS_FIELD(32, r1);
  BITS_FIELD(32, r2);
};

/* ---------------- Ethernet header (split 32+16) ---------------- */
struct eth_header_t {
  BITS_FIELD(32, dst_mac_1);
  BITS_FIELD(16, dst_mac_2);
  BITS_FIELD(32, src_mac_1);
  BITS_FIELD(16, src_mac_2);
  BITS_FIELD(16, ether_type);
};

/* ---------------- Dispatcher metadata ---------------- */
/* flags: bit0=drop (0 forward), bit1=mac_swapped */
struct monitor_meta_t {
  BITS_FIELD(32, egress_port);
  BITS_FIELD(32, flags);
  BITS_FIELD(32, ingress_port);
  BITS_FIELD(32, r0);
};

/* ---------------- Stats tables ---------------- */
/* key=0 => global counters */
struct global_stats_t {
  BITS_FIELD(32, total_packets);
  BITS_FIELD(32, r0);
};
ak_TABLE(4, BITS(32), struct global_stats_t, ) global_stats;

/* key=port => per-port counters */
struct port_stats_t {
  BITS_FIELD(32, packets);
  BITS_FIELD(32, r0);
};
ak_TABLE(256, BITS(32), struct port_stats_t, ) port_stats;

/* Swap src/dst MAC (no branches) */
static inline void
swap_mac(struct eth_header_t *eth) {
  BITS(32) t32;
  BITS(16) t16;

  t32 = eth->dst_mac_1; eth->dst_mac_1 = eth->src_mac_1; eth->src_mac_1 = t32;
  t16 = eth->dst_mac_2; eth->dst_mac_2 = eth->src_mac_2; eth->src_mac_2 = t16;
}

void NET_RECV__process_packet(buf_t packet) {
  /* 0) Extract ingress port */
  struct ingress_meta_t inmeta;
  bufextract(packet, (void*)&inmeta);
  BITS(32) ingress_port = inmeta.ingress_port;

  /* 1) Extract Ethernet (for MAC swap) */
  struct eth_header_t eth;
  bufextract(packet, (void*)&eth);

  /* 2) Update stats (global + per-port) */
  {
    BITS(32) k0 = (BITS(32))0;
    struct global_stats_t gs;
    gs.total_packets = (BITS(32))0;
    gs.r0 = (BITS(32))0;
    table_lookup(&global_stats, &k0, &gs);
    gs.total_packets = gs.total_packets + (BITS(32))1;
    table_update(&global_stats, &k0, &gs);
  }

  {
    struct port_stats_t ps;
    ps.packets = (BITS(32))0;
    ps.r0 = (BITS(32))0;
    table_lookup(&port_stats, &ingress_port, &ps);
    ps.packets = ps.packets + (BITS(32))1;
    table_update(&port_stats, &ingress_port, &ps);
  }

  /* 3) Swap MAC addresses */
  swap_mac(&eth);

  /* 4) Emit rewritten Ethernet back (depends on your pipeline's packet model) */
  bufemit(packet, (void*)&eth);

  /* 5) Emit dispatcher meta: hairpin to the same port */
  struct monitor_meta_t outm;
  outm.egress_port  = ingress_port;
  outm.flags        = (BITS(32))0 | ((BITS(32))1 << 1); /* mac_swapped=1 */
  outm.ingress_port = ingress_port;
  outm.r0           = (BITS(32))0;

  bufemit(packet, (void*)&outm);

  /* 6) Terminate */
  EXT__NET_SEND__net_send(packet);
}