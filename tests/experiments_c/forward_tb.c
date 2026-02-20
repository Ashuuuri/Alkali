#include "alkali.h"
#include "generic_handlers.h"

// Compile-time defaults (can be replaced by cfg_table written by control-plane)
#define DESTINATION_SVC  3

// -------------------- Packet headers (minimal) --------------------
struct eth_header_t {
  BITS_FIELD(48, dst_mac);
  BITS_FIELD(48, src_mac);
  BITS_FIELD(16, ether_type);
};

struct ip_header_t {
  BITS_FIELD(16, misc);
  BITS_FIELD(16, length);          // IPv4 total length (bytes): header + payload
  BITS_FIELD(16, identification);
  BITS_FIELD(16, fragment_offset);
  BITS_FIELD(16, TTL_transport);
  BITS_FIELD(16, checksum);
  BITS_FIELD(32, source_ip);
  BITS_FIELD(32, dst_ip);
  BITS_FIELD(32, options);
};

// -------------------- Token bucket state/config --------------------
// State: current tokens (bytes)
struct tb_state_t {
  BITS_FIELD(32, tokens_bytes);
};

// Config: bucket depth (bytes). (Rate is not used in dataplane; refill is external.)
struct tb_cfg_t {
  BITS_FIELD(32, depth_bytes);
};

// key = 0 => single global bucket (you can extend to per-flow later)
ak_TABLE(64, BITS(32), struct tb_state_t, ) tb_state_table;
ak_TABLE(64, BITS(32), struct tb_cfg_t, ) tb_cfg_table;

// -------------------- Metadata emitted to packet --------------------
// flags bit0: drop (1 = should drop)
// flags bit1: allow (1 = allowed)
// You can add more bits later.
struct tb_meta_t {
  BITS_FIELD(32, destination);
  BITS_FIELD(32, flags);
  BITS_FIELD(32, pkt_len_bytes);
  BITS_FIELD(32, tokens_after);
};

void NET_RECV__process_packet(buf_t packet) {
  // 0) Parse headers to compute packet length (best-effort)
  // NOTE: If your input packets are not guaranteed to be IPv4, you should
  //       place a classifier stage before this NF. We avoid if/else here.
  struct eth_header_t eth;
  struct ip_header_t ip;

  bufextract(packet, (void*)&eth);
  bufextract(packet, (void*)&ip);

  // Approx L2+L3 total length:
  // IPv4 total length field does NOT include Ethernet header.
  // Add 14 bytes Ethernet header as in your ONVM example.
  BITS(32) pkt_len = (BITS(32))ip.length + (BITS(32))14;

  // 1) Load config/state from tables (key = 0)
  BITS(32) key = 0;

  struct tb_cfg_t cfg;
  table_lookup(&tb_cfg_table, &key, &cfg);

  struct tb_state_t st;
  table_lookup(&tb_state_table, &key, &st);

  BITS(32) depth = cfg.depth_bytes;
  BITS(32) tokens = st.tokens_bytes;

  // 2) Compute allow/drop decisions WITHOUT branching
  // too_big = 1 if pkt_len > depth else 0
  BITS(32) too_big = (pkt_len > depth);

  // enough = 1 if tokens >= pkt_len else 0
  BITS(32) enough = (tokens >= pkt_len);

  // allow = (not too_big) AND enough
  // Since values are 0/1, we can use multiplication for AND.
  BITS(32) allow = (1 - too_big) * enough;

  // drop = 1 - allow
  BITS(32) drop = 1 - allow;

  // 3) Update tokens WITHOUT branching
  // If allow==1: tokens_after = tokens - pkt_len
  // If allow==0: tokens_after = tokens
  BITS(32) tokens_after = tokens - (pkt_len * allow);

  st.tokens_bytes = tokens_after;
  table_update(&tb_state_table, &key, &st);

  // 4) Emit metadata for downstream stages
  struct tb_meta_t meta;
  meta.destination = DESTINATION_SVC;
  meta.flags = (drop) | (allow << 1);
  meta.pkt_len_bytes = pkt_len;
  meta.tokens_after = tokens_after;

  bufemit(packet, &meta);

  // 5) Forward packet (downstream can drop if meta.flags bit0 is set)
  EXT__NET_SEND__net_send(packet);
}