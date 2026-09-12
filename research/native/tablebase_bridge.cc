// GPL-3.0: adapter to the vendored MacroXue table implementation.
// This translation unit uses a generated read-only tuple.h. It cannot compute
// or save new table entries. The actual game and RNG stay in Python.
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include "tuple.h"

struct FrozenTables {
  Tuple10 ten;
  Tuple11 eleven;
  explicit FrozenTables(const std::string& directory)
      : ten(1.0, (directory + "/tuple_moves.10a").c_str()),
        eleven(1.0, (directory + "/tuple_moves.11a").c_str()) {}
};

extern "C" {
void* frozen_tables_open(const char* directory) {
  if (!directory) return nullptr;
  const std::string path(directory);
  if (!std::ifstream(path + "/tuple_moves.10a").good() ||
      !std::ifstream(path + "/tuple_moves.11a").good()) return nullptr;
  try { return new FrozenTables(path); }
  catch (...) { return nullptr; }
}

void frozen_tables_close(void* handle) {
  delete static_cast<FrozenTables*>(handle);
}

// Return 0 for no qualifying cached recommendation, 10/11 for a hit, -1 on
// invalid input. Directions returned are our up/right/down/left order.
int frozen_tables_query(void* handle, const int* ranks, int* action, double* probability) {
  if (!handle || !ranks || !action || !probability) return -1;
  int layout[4][4];
  for (int i = 0; i < 16; ++i) {
    if (ranks[i] < 0 || ranks[i] > 17) return -1;
    layout[i / 4][i % 4] = ranks[i];
  }
  Board board(layout);
  auto* tables = static_cast<FrozenTables*>(handle);
  int move = -1;
  float p = tables->eleven.SuggestMove(board, &move);
  int kind = 11;
  if (p < 0.9) {  // Match upstream's double-literal threshold exactly.
    p = tables->ten.SuggestMove(board, &move);
    kind = 10;
    if (p <= 0) { *action = -1; *probability = 0; return 0; }
  }
  constexpr int native_to_python[] = {0, 3, 1, 2};
  if (move < 0 || move > 3 || !std::isfinite(p)) return -1;
  *action = native_to_python[move];
  *probability = p;
  return kind;
}
}
