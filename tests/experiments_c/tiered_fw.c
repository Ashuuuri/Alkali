#include "alkali.h"
#include "generic_handlers.h"

struct eth_header_t {
    BITS_FIELD(48, dst_mac);
    BITS_FIELD(48, src_mac);
    BITS_FIELD(16, ether_type);
};

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

struct tcp_header_t {
    BITS_FIELD(16, sport);
    BITS_FIELD(16, dport);
    BITS_FIELD(32, seq);
    BITS_FIELD(32, ack);
    BITS_FIELD(8,  flags);
    BITS_FIELD(8,  off);
    BITS_FIELD(16, win);
    BITS_FIELD(16, sum);
    BITS_FIELD(16, urp);
};

// flow_cache: 64 entries x (32b key + 2x32b value) = 64x12B = 768B
// Always fits in LMEM (1024B). Hot or cold, this table is always fast.
struct cache_val_t {
    BITS_FIELD(32, decision);
    BITS_FIELD(32, timestamp);
};
ak_TABLE(64, BITS(32), struct cache_val_t, ) flow_cache;

// session_table: 1536 entries x (32b key + 9x32b value) = 1536x40B = 60KB
// Accessed 8 times per packet (high density = 8/61440 ~ 1.3e-4).
// hot workload: highest density among large tables -> placed in CLS (16c/lookup)
// cold workload: larger than acl_table -> placed after acl in CLS -> overflows to CTM (40c/lookup)
// Note: 60KB + 24KB = 84KB > CLS(64KB), so exactly one of the two tables overflows.
struct session_val_t {
    BITS_FIELD(32, state);
    BITS_FIELD(32, flags);
    BITS_FIELD(32, byte_count_lo);
    BITS_FIELD(32, byte_count_hi);
    BITS_FIELD(32, pkt_count);
    BITS_FIELD(32, last_seq);
    BITS_FIELD(32, last_ack);
    BITS_FIELD(32, timeout);
    BITS_FIELD(32, checksum);
};
ak_TABLE(1536, BITS(32), struct session_val_t, ) session_table;

// acl_table: 1024 entries x (32b key + 5x32b value) = 1024x24B = 24KB
// Accessed 1 time per packet (low density = 1/24576 ~ 4.1e-5).
// cold workload: smaller than session_table -> placed first in CLS (16c/lookup)
// hot workload: lower density than session_table -> displaced to CTM (40c/lookup)
struct acl_val_t {
    BITS_FIELD(32, action);
    BITS_FIELD(32, priority);
    BITS_FIELD(32, src_mask);
    BITS_FIELD(32, dst_mask);
    BITS_FIELD(32, port_range);
};
ak_TABLE(1024, BITS(32), struct acl_val_t, ) acl_table;

void NET_RECV__process_packet(buf_t packet) {
    struct eth_header_t eth;
    struct ip_header_t  ip;
    struct tcp_header_t tcp;

    bufextract(packet, (void*)&eth);
    bufextract(packet, (void*)&ip);
    bufextract(packet, (void*)&tcp);

    BITS(32) src_key = ip.source_ip;
    BITS(32) dst_key = ip.dst_ip;
    BITS(32) flow_key = ip.source_ip + ip.dst_ip + tcp.sport + tcp.dport;

    // flow_cache: 2 lookups + 1 update (always LMEM, fast path)
    struct cache_val_t cache_fwd;
    struct cache_val_t cache_rev;
    table_lookup(&flow_cache, &src_key, &cache_fwd);
    table_lookup(&flow_cache, &dst_key, &cache_rev);

    // session_table: 6 lookups + 2 updates (8 ops total, high density)
    // In hot workload this table is in CLS; cold workload it overflows to CTM.
    struct session_val_t sess_fwd;
    struct session_val_t sess_rev;

    table_lookup(&session_table, &src_key, &sess_fwd);
    table_lookup(&session_table, &dst_key, &sess_rev);

    sess_fwd.pkt_count  = sess_fwd.pkt_count + (BITS(32))1;
    sess_fwd.byte_count_lo = sess_fwd.byte_count_lo + ip.length;
    sess_fwd.last_seq   = tcp.seq;
    table_update(&session_table, &src_key, &sess_fwd);

    table_lookup(&session_table, &src_key, &sess_fwd);
    table_lookup(&session_table, &dst_key, &sess_rev);

    sess_rev.last_ack   = tcp.ack;
    sess_rev.flags      = sess_rev.flags + tcp.flags;
    sess_rev.pkt_count  = sess_rev.pkt_count + (BITS(32))1;
    table_update(&session_table, &dst_key, &sess_rev);

    table_lookup(&session_table, &src_key, &sess_fwd);
    table_lookup(&session_table, &dst_key, &sess_rev);

    // acl_table: 1 lookup (low density)
    // Cold workload: CLS.  Hot workload: CTM.
    struct acl_val_t acl_entry;
    table_lookup(&acl_table, &flow_key, &acl_entry);

    // flow_cache update (LMEM)
    BITS(32) decision = acl_entry.action + sess_fwd.state + sess_rev.state;
    cache_fwd.decision = decision;
    cache_fwd.timestamp = sess_fwd.pkt_count;
    table_update(&flow_cache, &src_key, &cache_fwd);

    bufemit(packet, (void*)&sess_fwd);
    bufemit(packet, (void*)&acl_entry);

    EXT__NET_SEND__net_send(packet);
}
