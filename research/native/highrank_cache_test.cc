#include <assert.h>
#include <stdio.h>
#include "node.h"

int main() {
  Board::BuildMoveMap();
  Node::BuildScoreMap();
  options.tuple_moves = false;  // Full search, no score threshold or solved tables.
  options.min_prob = 0;
  int a[N][N] = {}, b[N][N] = {};
  a[3][2] = 1;  // A 2 in the penultimate cell.
  b[3][3] = 16; // A 65536 in the last cell: a different board, same 4-bit key.
  assert(Board(a).Compact() == Board(b).Compact());

  int layout[N][N] = {{15,14,13,12},{8,9,10,11},{7,6,5,4},{1,2,3,0}};
  Node ordinary(layout);
  int action;
  ordinary.Search(1, &action);  // Initializes private search settings.
  Node::cache.Clear();
  Node::cache.Update(ordinary.Compact(), 1, 0, 12345);
  assert(ordinary.TryAllTiles(0, 1) == 12345); // Ordinary keys still use the cache.

  layout[0][0] = 16;
  Node high(layout);
  high.Search(1, &action);
  Node::cache.Clear();
  const int expected = high.TryAllTiles(0, 1);
  assert(expected != 12345);
  Node::cache.Clear();
  Node::cache.Update(high.Compact(), 1, 0, 12345); // Poison the ambiguous key.
  const int actual = high.TryAllTiles(0, 1);
  if (actual != expected) {
    fprintf(stderr, "High-rank board accepted an ambiguous cached value: expected %d, got %d\n", expected, actual);
    return 1;
  }
  Node::cache.Clear();
  high.TryAllTiles(0, 1);
  int ignored;
  assert(!Node::cache.Lookup(high.Compact(), 1, 0, &ignored));
  puts("Ordinary cache hits preserved; high-rank cache reads and writes bypassed.");
}
