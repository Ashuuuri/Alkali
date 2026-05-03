// Hash-based load balancer: maps flows to backends via IP-pair hash.
// Flow-to-backend assignments are persisted in flow_table for stickiness.
//
// Limitations (EP2 dialect):
//   - Hash uses src_ip + dst_ip only; including port fields (i16) in XOR
//     with i32 values produces llvm.xor which EmitNetronomePass cannot print.
//   - backend_num must be a power of 2 (bitwise & used instead of %).

#include "alkali.h"
#include "generic_handlers.h"

struct eth_header_t {
    BITS_FIELD(48, dst_mac);
    BITS_FIELD(48, src_mac);
    BITS_FIELD(16, ether_type);
};

struct ip_header_t {
    BITS_FIELD(16, misc);           // Version(4), IHL(4), DSCP(6), ECN(2)
    BITS_FIELD(16, length);
    BITS_FIELD(16, identification);
    BITS_FIELD(16, fragment_offset); // Flags(3), Fragment Offset(13)
    BITS_FIELD(16, TTL_transport);   // TTL(8), Protocol(8)
    BITS_FIELD(16, checksum);
    BITS_FIELD(32, source_ip);
    BITS_FIELD(32, dst_ip);
    BITS_FIELD(32, options);
};

struct tcp_header_t {
    BITS_FIELD(16, sport);
    BITS_FIELD(16, dport);
    BITS_FIELD(32, seq);
    BITS_FIELD(32, ack);
    BITS_FIELD(8,  off);
    BITS_FIELD(8,  flags);
    BITS_FIELD(16, win);
    BITS_FIELD(16, sum);
    BITS_FIELD(16, urp);
};

struct lb_entry_t {
    BITS_FIELD(1,  if_alloc);
    BITS_FIELD(32, backend_ip);
    BITS_FIELD(16, backend_port);
};

struct backend_t {
    BITS_FIELD(32, ip);
    BITS_FIELD(16, port);
};

struct lb_fwd_meta_t {
    BITS_FIELD(32, flow_id);
    BITS_FIELD(32, backend_ip);
    BITS_FIELD(16, backend_port);
};

#define BACKEND_NUM  4   // must be power of 2
#define BASE_IP      0x08080807  // 8.8.8.7
#define BASE_PORT    8000

ak_TABLE(16,   BITS(32), struct backend_t)   backend_table;
ak_TABLE(1024, BITS(32), struct lb_entry_t)  flow_table;

void NET_RECV__process_packet(buf_t packet) {
    struct eth_header_t eth_header;
    struct ip_header_t  ip_header;
    struct tcp_header_t tcp_header;

    bufextract(packet, (void *)&eth_header); // consumed to advance packet offset
    bufextract(packet, (void *)&ip_header);
    bufextract(packet, (void *)&tcp_header);

    // Initialize backend table on every packet (no init callback in Alkali).
    struct backend_t be;
    BITS(32) idx;

    idx = 0; be.ip = BASE_IP;     be.port = BASE_PORT;
    table_update(&backend_table, &idx, &be);

    idx = 1; be.ip = BASE_IP + 1; be.port = BASE_PORT + 1;
    table_update(&backend_table, &idx, &be);

    idx = 2; be.ip = BASE_IP + 2; be.port = BASE_PORT + 2;
    table_update(&backend_table, &idx, &be);

    idx = 3; be.ip = BASE_IP + 3; be.port = BASE_PORT + 3;
    table_update(&backend_table, &idx, &be);

    // Select backend by hashing src+dst IP; & (BACKEND_NUM-1) avoids llvm.urem.
    BITS(32) fid = ip_header.source_ip + ip_header.dst_ip;
    BITS(32) backend_idx = fid & (BACKEND_NUM - 1);
    be.ip = 0; be.port = 0;
    table_lookup(&backend_table, &backend_idx, &be);

    // Pre-initialize with fresh allocation so a flow_table miss assigns be directly.
    // On hit, stored values overwrite; on miss, pre-initialized values are kept.
    struct lb_entry_t ent;
    ent.if_alloc     = 1;
    ent.backend_ip   = be.ip;
    ent.backend_port = be.port;
    table_lookup(&flow_table, &fid, &ent);
    table_update(&flow_table, &fid, &ent);

    // Rewrite destination to backend
    ip_header.dst_ip = ent.backend_ip;
    tcp_header.dport = ent.backend_port;

    struct lb_fwd_meta_t meta;
    meta.flow_id      = fid;
    meta.backend_ip   = ent.backend_ip;
    meta.backend_port = ent.backend_port;

    bufemit(packet, &meta);
    EXT__NET_SEND__net_send(packet);
}
