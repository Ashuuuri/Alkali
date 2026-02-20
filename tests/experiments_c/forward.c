#include "alkali.h"
#include "generic_handlers.h"

// ---- compile-time configuration (replaces CLI arguments) ----
#define DESTINATION_SVC  3
#define PRINT_DELAY      1000000

// ---- state: use a table to store a global packet counter (fixed key = 0) ----
struct counter_state_t {
  BITS_FIELD(32, cnt);
};
ak_TABLE(64, BITS(32), struct counter_state_t, ) counter_table;

// ---- custom forwarding metadata (replaces onvm_pkt_meta.destination) ----
struct forward_meta_t {
  BITS_FIELD(32, destination); // destination service identifier
  BITS_FIELD(32, flags);       // bit0 = should_report
  BITS_FIELD(32, seq);         // local packet counter snapshot
  BITS_FIELD(32, reserved);
};

void NET_RECV__process_packet(buf_t packet) {
  // 1) Load and increment the global packet counter
  BITS(32) key = 0;
  struct counter_state_t st;
  table_lookup(&counter_table, &key, &st);

  BITS(32) new_cnt = st.cnt + 1;

  // Compute hit without control-flow:
  // hit = 1 if new_cnt >= PRINT_DELAY, else 0
  BITS(32) hit = (new_cnt >= PRINT_DELAY);

  // Reset counter without using branches:
  // if hit == 1 -> st.cnt = 0
  // else        -> st.cnt = new_cnt
  st.cnt = new_cnt * (1 - hit);

  table_update(&counter_table, &key, &st);

  // 2) Emit forwarding metadata
  struct forward_meta_t meta;
  meta.destination = DESTINATION_SVC;
  meta.flags = hit;      // downstream stages can act on this flag
  meta.seq = new_cnt;    // snapshot of the counter before reset
  meta.reserved = 0;

  bufemit(packet, &meta);

  // 3) Forward the packet to the next pipeline stage
  EXT__NET_SEND__net_send(packet);
}
