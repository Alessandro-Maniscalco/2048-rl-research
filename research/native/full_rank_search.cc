// GPL-3.0: isolated adapter to MacroXue's cache-corrected search implementation.
// Full tile ranks are retained. Search still uses heuristics and probability
// pruning: "full rank" does not mean exact optimal full-game values.
#include <mutex>
#include "node.h"

#define EXPORT __attribute__((visibility("default")))
static std::once_flag initialized;
static std::mutex search_mutex;

static bool make_node(const int* ranks, Node& node) {
  if (!ranks) return false;
  int layout[4][4];
  for (int i=0; i<16; ++i) {
    if (ranks[i]<0 || ranks[i]>17) return false;
    layout[i/4][i%4]=ranks[i];
  }
  node=Node(layout);
  return true;
}

static void initialize() {
  std::call_once(initialized, [] {
    Board::BuildMoveMap();
    Node::BuildScoreMap();
  });
}

extern "C" {
EXPORT int full_rank_query(const int* ranks, int depth, int* action, int* value) {
  if (!action || !value || depth<1 || depth>10) return -1;
  std::lock_guard<std::mutex> guard(search_mutex);
  initialize();
  Node node;
  if (!make_node(ranks,node)) return -1;
  options.max_depth=depth;
  options.UpdateMinProbFromDepth();
  // A late-stage board can be anchored in any corner. This built-in evaluator
  // takes the best row/column orientation; no extra human-chosen weights.
  options.interactive=true;
  options.tuple_moves=true;
  Node::cache.Clear();
  int native_action=-1;
  *value=node.Search(depth,&native_action);
  if (native_action<0) { *action=-1; return 0; }
  constexpr int map[] = {0,3,1,2};
  *action=map[native_action];
  return 1;
}

EXPORT int full_rank_slide(const int* ranks, int action, int* output, int* reward) {
  if (!output || !reward || action<0 || action>3) return -1;
  std::lock_guard<std::mutex> guard(search_mutex);
  initialize();
  Node node;
  if (!make_node(ranks,node)) return -1;
  int before=node.GameScore();
  constexpr int map[] = {0,2,3,1};
  bool changed=(node.*Board::moves[map[action]])();
  *reward=node.GameScore()-before;
  for (int i=0;i<16;++i) output[i]=node[i%4][i/4];
  return changed ? 1 : 0;
}
}
