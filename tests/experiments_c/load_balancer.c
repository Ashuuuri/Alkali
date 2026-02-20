// Round-robin (TODO)
// Flow timeout / eviction (TODO)

#include "alkali.h"
#include "generic_handlers.h"

struct eth_header_t {
    BITS_FIELD(48, dst_mac);
    BITS_FIELD(48, src_mac);
    BITS_FIELD(16, ether_type);
};

struct ip_header_t {
    // Version (4), IHL (4), DSCP (6), ECN (2)
    BITS_FIELD(16, misc);

    BITS_FIELD(16, length);
    BITS_FIELD(16, identification);

    // Flags (3), Fragment Offset (13)
    BITS_FIELD(16, fragment_offset);

    // TTL (8), Transport Protocol (8)
    BITS_FIELD(16, TTL_transport);

    BITS_FIELD(16, checksum);
    BITS_FIELD(32, source_ip);
    BITS_FIELD(32, dst_ip);
    BITS_FIELD(32, options);
};

struct tcp_header_t {
    BITS_FIELD(16, sport); // Source port
    BITS_FIELD(16, dport); // Destination port

    BITS_FIELD(32, seq);   // Sequence number
    BITS_FIELD(32, ack);   // Acknowledgement number

    BITS_FIELD(8,  off);   // Data offset
    BITS_FIELD(8,  flags); // Flags
    BITS_FIELD(16, win);   // Window

    BITS_FIELD(16, sum);   // Checksum
    BITS_FIELD(16, urp);   // Urgent pointer
};

struct lb_entry_t {
    BITS_FIELD(32, flow_id);
    BITS_FIELD(1,  if_alloc);
    BITS_FIELD(32, backend_ip);
    BITS_FIELD(16, backend_port);
    //BITS_FIELD(48, backend_mac);
};

struct backend_t {
    BITS_FIELD(32, ip);
    BITS_FIELD(16, port);
    //BITS_FIELD(48, mac);
};

struct lb_fwd_meta_t {
    BITS_FIELD(32, flow_id);
    BITS_FIELD(32, backend_ip);
    BITS_FIELD(16, backend_port);
    //BITS_FIELD(48, backend_mac);
};

ak_TABLE(16,   BITS(32), struct backend_t)   backend_table;
ak_TABLE(1024, BITS(32), struct lb_entry_t)  flow_table;

void NET_RECV__process_packet(buf_t packet) {
    struct eth_header_t eth_header;
    struct ip_header_t  ip_header;
    struct tcp_header_t tcp_header;

    bufextract(packet, (void *)&eth_header);
    bufextract(packet, (void *)&ip_header);
    bufextract(packet, (void *)&tcp_header); // Extract headers from packet

    // Backend init (demo values)
    BITS(32) base_ip   = 134744071;
    BITS(16) base_port = 8000;
    //BITS(64) base_mac = 0x001122334400;

    struct backend_t be;
    BITS(32) idx;

    /* backend 0 */
    idx = 0;
    be.ip   = base_ip + 0;
    be.port = base_port + 0;
    //be.mac = base_mac + 0;
    //table_update(&backend_table, &idx, &be);

    /* backend 1 */
    idx = 1;
    be.ip   = base_ip + 1;
    be.port = base_port + 1;
    //be.mac = base_mac + 1;
    //table_update(&backend_table, &idx, &be);

    /* backend 2 */
    idx = 2;
    be.ip   = base_ip + 2;
    be.port = base_port + 2;
    //be.mac = base_mac + 2;
    //table_update(&backend_table, &idx, &be);

    /* backend 3 */
    idx = 3;
    be.ip   = base_ip + 3;
    be.port = base_port + 3;
    //be.mac = base_mac + 3;
    //table_update(&backend_table, &idx, &be);

    BITS(32) backend_num = 4;

    // Flow label using XOR of 4-tuple fields (simple hash)
    // TODO: confirm Alkali supports 4-tuple as a direct table key if needed.
    BITS(32) fid = ip_header.source_ip ^ ip_header.dst_ip ^ tcp_header.sport ^ tcp_header.dport;

    struct lb_entry_t ent;
    table_lookup(&flow_table, &fid, &ent);

    // If not allocated yet, assign a backend for this flow
    if (ent.if_alloc == 0) {
        BITS(32) idx = fid % backend_num; // Same flow goes to the same backend

        table_lookup(&backend_table, &idx, &be);

        ent.if_alloc     = 1;
        ent.backend_ip   = be.ip;
        ent.backend_port = be.port;
        //ent.backend_mac = be.mac;

        table_update(&flow_table, &fid, &ent);
    }

    // Rewrite destination to backend
    ip_header.dst_ip = ent.backend_ip;
    tcp_header.dport = ent.backend_port;
    //eth_header.dst_mac = ent.backend_mac;

    // Emit metadata for downstream stages
    struct lb_fwd_meta_t meta;
    meta.flow_id      = fid;
    meta.backend_ip   = ent.backend_ip;
    meta.backend_port = ent.backend_port;
    //meta.backend_mac = ent.backend_mac;

    bufemit(packet, &meta);
    EXT__NET_SEND__net_send(packet);
}