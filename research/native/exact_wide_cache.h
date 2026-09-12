// GPL-3.0: cache for the isolated MacroXue search adapter.
// A hit requires all 80 board bits, exact remaining depth and identical float
// probability bits. Epochs prevent reuse across different root search settings.
#ifndef EXACT_WIDE_CACHE_H
#define EXACT_WIDE_CACHE_H
#include <cstdint>
#include <cstring>
#include <memory>

class ExactWideCache {
 public:
  struct Entry {
    uint64_t low = 0, high = 0;
    uint32_t epoch = 0, probability = 0;
    int depth = 0, score = 0;
  };
  ExactWideCache() : entries(new Entry[kSize]) {}
  bool enabled = true;
  uint64_t lookups = 0, hits = 0, updates = 0, tile_calls = 0;
  void Clear() {
    if (++epoch == 0) {
      for (size_t i = 0; i < kSize; ++i) entries[i].epoch = 0;
      epoch = 1;
    }
    lookups = hits = updates = tile_calls = 0;
  }
  bool Lookup(uint64_t low, uint64_t high, float prob, int depth, int* score) {
    ++lookups;
    auto bits = Bits(prob);
    const auto& entry = entries[Index(low, high, bits, depth)];
    if (entry.epoch != epoch || entry.low != low || entry.high != high ||
        entry.probability != bits || entry.depth != depth) return false;
    ++hits;
    *score = entry.score;
    return true;
  }
  void Update(uint64_t low, uint64_t high, float prob, int depth, int score) {
    ++updates;
    auto bits = Bits(prob);
    auto& entry = entries[Index(low, high, bits, depth)];
    entry = Entry{low, high, epoch, bits, depth, score};
  }
 private:
  static uint32_t Bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  }
  static uint64_t Mix(uint64_t x) {
    x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
  }
  static size_t Index(uint64_t low, uint64_t high, uint32_t prob, int depth) {
    return Mix(low ^ Mix(high) ^ Mix((uint64_t(prob) << 8) | uint32_t(depth))) & (kSize - 1);
  }
  static constexpr size_t kSize = 1 << 20;
  std::unique_ptr<Entry[]> entries;
  uint32_t epoch = 0;
};
#endif
