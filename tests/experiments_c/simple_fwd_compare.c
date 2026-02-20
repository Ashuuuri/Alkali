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


/*
netronome build error:

/home/zy/Desktop/Alkali/scripts/netronome_compile.sh: line 50:
8610 Floating point exception (core dumped)

./build/bin/ep2c-opt
-ep2-pipeline-handler="mode=loop target=netronome"
"netronome_out/commonopt.mlir"
-o "netronome_out/cut.mlir"

Observed during Netronome target compilation.
*/


/*
rtl build error:

Info: __handler_NET_RECV_process_packet_1 has 0 replicated args
%7 = llvm.zext %6 : i1 to i32

ep2c-opt:
.../EmitFPGAHelper.cpp:520:
std::string mlir::ep2::EmitFPGAPass::getValName(mlir::Value):
Assertion `false' failed.

Likely related to boolean comparison (icmp) and zero-extension handling
in the RTL backend.
*/