#include <cassert>
#include <cmath>
#include "exact_wide_cache.h"

int main() {
  ExactWideCache cache;
  cache.Clear();
  int value = 0;
  cache.Update(16, 0, .1f, 3, -400000000);
  assert(cache.Lookup(16, 0, .1f, 3, &value) && value == -400000000);
  assert(!cache.Lookup(32, 0, .1f, 3, &value));
  assert(!cache.Lookup(16, 1, .1f, 3, &value));
  assert(!cache.Lookup(16, 0, .1f, 2, &value));
  assert(!cache.Lookup(16, 0, std::nextafter(.1f, 1.f), 3, &value));
  cache.Clear();
  assert(!cache.Lookup(16, 0, .1f, 3, &value));
}
