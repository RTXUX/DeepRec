#ifndef DEEPREC_CACHE_TUNING_STRATEGY_H
#define DEEPREC_CACHE_TUNING_STRATEGY_H

#include <map>
#include <memory>
#include <random>

#include "tensorflow/core/framework/embedding/cache_profiler.h"

namespace tensorflow {
namespace embedding {

double InterpolateMRC(const std::vector<double>& mrc, size_t bucket_size,
                      size_t target);

class CacheItem {
 public:
  size_t bucket_size;
  size_t orig_size;
  size_t new_size;
  size_t entry_size;
  size_t min_size;
  uint64_t vc;
  uint64_t mc;
  double mr;
  std::vector<double> mrc;

  CacheItem(size_t bucketSize, size_t origSize, size_t newSize,
            size_t entrySize, size_t minSize, uint64_t vc, uint64_t mc, double mr,
            const std::vector<double>& mrc);

  CacheItem();
};

class CacheTuningStrategy {
 public:
  virtual bool DoTune(size_t total_size,
                      std::map<CacheMRCProfiler*, CacheItem>& caches,
                      size_t unit) = 0;
};

class MinimalizeMissCountRandomGreedyTuningStrategy
    : public CacheTuningStrategy {
 public:
  bool DoTune(size_t total_size, std::map<CacheMRCProfiler*, CacheItem>& caches,
              size_t unit) override;
};

class MinimalizeMissCountLocalGreedyTuningStrategy
    : public CacheTuningStrategy {
 public:
  bool DoTune(size_t total_size, std::map<CacheMRCProfiler*, CacheItem>& caches,
              size_t unit) override;
};

class MinimalizeMissCountDynamicProgrammingTuningStrategy
    : public CacheTuningStrategy {
 public:
  bool DoTune(size_t total_size, std::map<CacheMRCProfiler*, CacheItem>& caches,
              size_t unit) override;
};

class MinimalizeMissRateDynamicProgrammingTuningStrategy
    : public CacheTuningStrategy {
 public:
  bool DoTune(size_t total_size, std::map<CacheMRCProfiler*, CacheItem>& caches,
              size_t unit) override;
};

class CacheTuningStrategyCreator {
 public:
  static CacheTuningStrategy* Create(const std::string& type);
};

}  // namespace embedding
}  // namespace tensorflow

#endif  // DEEPREC_CACHE_TUNING_STRATEGY_H
