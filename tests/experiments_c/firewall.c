#include "alkali.h"
#include "generic_handlers.h"

// Ethernet header
struct eth_header_t {
  BITS_FIELD(48, dst_mac);
  BITS_FIELD(48, src_mac);
  BITS_FIELD(16, ether_type);
};

// IPv4 header (fixed 20B)
struct ip_header_t {
  BITS_FIELD(16, misc);
  BITS_FIELD(16, length);
  BITS_FIELD(16, identification);
  BITS_FIELD(16, fragment_offset);
  BITS_FIELD(8,  ttl);
  BITS_FIELD(8,  protocol);
  BITS_FIELD(16, hdr_checksum);
  BITS_FIELD(32, source_ip);
  BITS_FIELD(32, dst_ip);
};

// Firewall table:
//   key   = src_ip
//   value = rule (0 => allow, 1 => drop)
//
// NOTE: Use BITS(32) directly to avoid struct value lowering issues.
ak_TABLE(64, BITS(32), BITS(32), ) firewall_ip_table;

void NET_RECV__process_packet(buf_t packet) {
  struct eth_header_t eth;
  struct ip_header_t ip;

  // eth must be consumed before ip (sequential stream); ether_type not checked
  // — non-IPv4 traffic misses the table and is dropped by default.
  bufextract(packet, (void*)&eth);
  bufextract(packet, (void*)&ip);

  // Look up firewall rule for src_ip.
  // Pre-init rule=1 (drop) so a miss defaults to drop.
  BITS(32) key  = ip.source_ip;
  BITS(32) rule = (BITS(32))1;
  table_lookup(&firewall_ip_table, &key, &rule);

  // allow = 1 - rule  (assumes rule is 0 or 1)
  BITS(32) allow = (BITS(32))1 - rule;

  // Emit rule and allow as separate scalars to avoid struct lowering issues
  bufemit(packet, (void*)&rule);
  bufemit(packet, (void*)&allow);

  EXT__NET_SEND__net_send(packet);
}
