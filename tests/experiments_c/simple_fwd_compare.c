#include "alkali.h"
#include "generic_handlers.h"

// Minimal IPv4 header fields used in this handler
struct ip_header_t {
    BITS_FIELD(16, misc);             // Version/IHL/DSCP/ECN
    BITS_FIELD(16, length);           // Total length of the IP packet
    BITS_FIELD(16, identification);   // Identification field
    BITS_FIELD(16, fragment_offset);  // Flags + Fragment offset
    BITS_FIELD(16, TTL_transport);    // TTL + Protocol
    BITS_FIELD(16, checksum);         // Header checksum
    BITS_FIELD(32, source_ip);        // Source IP address
    BITS_FIELD(32, dst_ip);           // Destination IP address
    BITS_FIELD(32, options);          // Options (if present)
};

// Metadata emitted to indicate whether the packet should be dropped
struct drop_it {
    BITS_FIELD(16, if_drop);          // 1 = drop, 0 = forward
};

void NET_RECV__process_packet(buf_t packet) {

    // Extract IP header from incoming packet buffer
    struct ip_header_t ip_header;
    bufextract(packet, (void *)&ip_header);

    // Maximum allowed packet length (token threshold)
    BITS(16) max_token;
    max_token = 20000;

    // Read packet length from IP header
    BITS(16) length;
    length = ip_header.length;

    // Drop condition:
    // If packet length exceeds the configured threshold,
    // mark it to be dropped.
    BITS(16) if_drop;
    if_drop = (length > max_token);

    // Emit drop decision as metadata for downstream processing
    struct drop_it res;
    res.if_drop = if_drop;
    bufemit(packet, &res);

    // Forward packet (actual drop handling may occur in later stages)
    EXT__NET_SEND__net_send(packet);
}


// Known limitations:
//   - Netronome: min-cut fails (FPE in ep2c-opt); handler runs as single stage.
//   - FPGA: boolean comparison (icmp + llvm.zext i1→i32) hits an unhandled case
//     in EmitFPGAPass::getValName; not supported on this target.