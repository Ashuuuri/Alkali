#include "alkali.h"
#include "generic_handlers.h"

/* Ethernet header */
struct eth_header_t {
  BITS_FIELD(48, dst_mac);
  BITS_FIELD(48, src_mac);
  BITS_FIELD(16, ether_type);
};

/* IPv4 header (fixed 20B) */
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

/*
 * Firewall table:
 *   key   = src_ip
 *   value = rule (0 => allow, non-zero => drop)
 *
 * IMPORTANT: Use BITS(32) directly to avoid struct value lowering issues.
 */
ak_TABLE(64, BITS(32), BITS(32), ) firewall_ip_table;

void NET_RECV__process_packet(buf_t packet) {
  struct eth_header_t eth;
  struct ip_header_t ip;

  /* Always extract in a fixed order */
  bufextract(packet, (void*)&eth);
  bufextract(packet, (void*)&ip);

  /* Default: drop */
  BITS(32) rule  = (BITS(32))1;
  BITS(32) allow = (BITS(32))0;

  /* Only apply firewall to IPv4 */
  if (eth.ether_type == (BITS(16))0x0800) {
    BITS(32) key = ip.source_ip;

    /*
     * On miss, rule should remain default=1.
     * So we pass a pre-initialized rule into lookup as the "default".
     */
    BITS(32) looked_rule = (BITS(32))1;
    table_lookup(&firewall_ip_table, &key, &looked_rule);

    rule = looked_rule;

    if (rule == (BITS(32))0) {
      allow = (BITS(32))1;
    } else {
      allow = (BITS(32))0;
    }
  }

  /*
   * Append meta (rule, allow) WITHOUT using a struct.
   * This avoids ep2 struct_update / stackslot init bugs.
   */
  bufemit(packet, (void*)&rule);
  bufemit(packet, (void*)&allow);

  /* Must terminate by calling the next stage */
  EXT__NET_SEND__net_send(packet);
}