#ifndef DEEPREC_CACHE_MANAGER_H
#define DEEPREC_CACHE_MANAGER_H

#include <atomic>
#include <cstddef>
#include <map>
#include <memory>
#include <random>

#include "tensorflow/core/framework/embedding/cache_profiler.h"
#include "tensorflow/core/framework/embedding/cache_tuning_strategy.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace embedding {

class MockTunableCache : public TunableCache {
 public:
  explicit MockTunableCache(size_t numEntries);

  size_t GetCacheSize() const override;

  void SetCacheSize(size_t new_size) override;

  size_t GetCacheEntrySize() const override;

 private:
  size_t num_entries_;
};

struct CacheStat {
  uint64 prev_promotion = 0;
  uint64 prev_demotion = 0;
  uint64 promotion = 0;
  uint64 demotion = 0;
  uint64 visit_count = 0;
  double hit_rate = 0;
};

class CacheManager {
 public:
  static CacheManager& GetInstance();

  void RegisterCache(CacheMRCProfiler& cache);

  void UnregisterCache(const std::string& name);

  void Tune(size_t total_size, size_t unit);

  void DoTune(size_t total_size, std::vector<CacheMRCProfiler*> caches,
              size_t unit);

  void Access(size_t size);

  bool CheckCache();

  void StartThread();

  void TuneLoop();

  void IncreaseNanos(uint64_t lru_nano, uint64_t profiler_nano);

  bool SamplingActive() const;

  // Notify CacheManager batch size in bytes
  void NotifyBatchSize(size_t batch_size);

 private:
  mutex mu_;
  mutex min_size_mu_;
  std::atomic<uint64> num_active_threads_;
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::unique_ptr<CacheTuningStrategy> tuning_strategy_;
  std::map<std::string, CacheMRCProfiler*> registry_;
  std::map<CacheMRCProfiler*, CacheStat> cache_stats_;

  std::atomic<uint64> access_count_;
  uint64 tuning_interval_;
  uint64 step_ = 1;

  std::atomic<uint64_t> lru_nanos;
  std::atomic<uint64_t> profiler_nanos;

  std::atomic<bool> sampling_active_;
  std::atomic<bool> should_tune_;
  int64_t notune_counter_ = 0;
  int64_t notune_threshold_;

  size_t total_size_;
  size_t min_size_;
  size_t max_batch_size_;
  size_t tuning_unit_;
  std::atomic<size_t> access_size_;
  std::atomic_flag access_size_lock_ = ATOMIC_FLAG_INIT;

  bool clear_stat_;
  bool min_size_specified_;

  explicit CacheManager();
};

}  // namespace embedding
}  // namespace tensorflow
#endif  // DEEPREC_CACHE_MANAGER_H
