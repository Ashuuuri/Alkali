#include "alkali.h"
#include "generic_handlers.h"

/* Upstream ingress metadata */
struct ingress_meta_t {
  BITS_FIELD(32, ingress_port);
  BITS_FIELD(32, r0);
  BITS_FIELD(32, r1);
  BITS_FIELD(32, r2);
};

/* Meta consumed by the dispatcher */
struct bridge_meta_t {
  BITS_FIELD(32, egress_port);
  BITS_FIELD(32, flags);        /* bit0=drop (0 forward), other bits reserved */
  BITS_FIELD(32, ingress_port); /* optional: for tracing/debug */
  BITS_FIELD(32, r0);
};

void NET_RECV__process_packet(buf_t packet) {
  /* 0) Extract upstream ingress metadata */
  struct ingress_meta_t inmeta;
  bufextract(packet, (void*)&inmeta);

  BITS(32) in_port = inmeta.ingress_port;

  /* 1) Compute egress port: swap 0 <-> 1 */
  BITS(32) out_port = in_port ^ (BITS(32))1;

  /* 2) Emit dispatcher metadata */
  struct bridge_meta_t outm;
  outm.egress_port  = out_port;
  outm.flags        = (BITS(32))0;  /* no drop */
  outm.ingress_port = in_port;
  outm.r0           = (BITS(32))0;

  bufemit(packet, (void*)&outm);

  /* 3) Terminate stage */
  EXT__NET_SEND__net_send(packet);
}