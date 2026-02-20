#include "alkali.h"
#include "generic_handlers.h"

/*
 * Version A contract:
 *  - Upstream (or host feeder) must prepend ingress_meta_t to provide ingress_port.
 *  - This stage decides egress_port and emits l2sw_meta_t for a host dispatcher.
 *  - No send-to-port primitive exists; we only net_send() to the next stage.
 */

/* Ingress metadata (must exist in packet before this stage) */
struct ingress_meta_t {
  BITS_FIELD(32, ingress_port);
  BITS_FIELD(32, reserved0);
  BITS_FIELD(32, reserved1);
  BITS_FIELD(32, reserved2);
};

/* Ethernet header with MAC split into 32+16 (Alkali-friendly) */
struct eth_header_t {
  BITS_FIELD(32, dst_mac_1);
  BITS_FIELD(16, dst_mac_2);
  BITS_FIELD(32, src_mac_1);
  BITS_FIELD(16, src_mac_2);
  BITS_FIELD(16, ether_type);
};

/* dst_port_table: ingress_port -> egress_port */
struct port_map_t {
  BITS_FIELD(32, egress_port);
};
ak_TABLE(64, BITS(32), struct port_map_t, ) dst_port_table;

/* cfg_table[0]: global switch flags */
struct sw_cfg_t {
  BITS_FIELD(32, mac_updating);  // 0/1
};
ak_TABLE(64, BITS(32), struct sw_cfg_t, ) cfg_table;

/* stats_table: per-port counters */
struct port_stats_t {
  BITS_FIELD(32, rx);
  BITS_FIELD(32, tx);
  BITS_FIELD(32, dropped);
};
ak_TABLE(64, BITS(32), struct port_stats_t, ) stats_table;

/* Metadata emitted for host dispatcher */
struct l2sw_meta_t {
  BITS_FIELD(32, egress_port);
  BITS_FIELD(32, flags);        // bit0=drop, bit1=mac_updated
  BITS_FIELD(32, ingress_port);
  BITS_FIELD(32, reserved);
};

void NET_RECV__process_packet(buf_t packet) {
  /* 0) Extract ingress meta (provided by upstream/host) */
  struct ingress_meta_t inmeta;
  bufextract(packet, (void*)&inmeta);
  BITS(32) ingress_port = inmeta.ingress_port;

  /* 1) Lookup egress port */
  struct port_map_t pm;
  table_lookup(&dst_port_table, &ingress_port, &pm);
  BITS(32) egress_port = pm.egress_port;

  /* 2) Update RX stats on ingress port */
  struct port_stats_t st_in;
  table_lookup(&stats_table, &ingress_port, &st_in);
  st_in.rx = st_in.rx + 1;
  table_update(&stats_table, &ingress_port, &st_in);

   /* 3) Read config (mac_updating: 0/1) */
  BITS(32) cfg_key = 0;
  struct sw_cfg_t cfg;
  table_lookup(&cfg_table, &cfg_key, &cfg);

  /* m32 is 0/1 in i32 domain */
  BITS(32) m32 = cfg.mac_updating;
  BITS(32) inv32 = (BITS(32))1 - m32;

  /* m16 / inv16 are 0/1 in i16 domain (for 16-bit fields) */
  BITS(16) m16 = (BITS(16))m32;
  BITS(16) inv16 = (BITS(16))1 - m16;

  struct eth_header_t eth;
  bufextract(packet, (void*)&eth);

  BITS(32) new_mac_1 = (BITS(32))0x02000000;
  BITS(16) new_mac_2 = (BITS(16))egress_port;

  /* 32-bit parts */
  eth.dst_mac_1 = eth.dst_mac_1 * inv32 + new_mac_1 * m32;
  eth.src_mac_1 = eth.src_mac_1 * inv32 + new_mac_1 * m32;

  /* 16-bit parts (stay entirely in i16 math) */
  {
    BITS(16) dst2 = (BITS(16))eth.dst_mac_2;
    BITS(16) src2 = (BITS(16))eth.src_mac_2;

    dst2 = dst2 * inv16 + new_mac_2 * m16;
    src2 = src2 * inv16 + new_mac_2 * m16;

    eth.dst_mac_2 = (BITS(16))dst2;
    eth.src_mac_2 = (BITS(16))src2;
  }

  bufemit(packet, (void*)&eth);
}