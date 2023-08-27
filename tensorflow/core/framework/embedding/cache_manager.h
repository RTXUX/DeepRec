#ifndef DEEPREC_CACHE_MANAGER_H
#define DEEPREC_CACHE_MANAGER_H

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

class CacheManager {
 public:
  static CacheManager& GetInstance();

  void RegisterCache(CacheMRCProfiler& cache);

  void UnregisterCache(const std::string& name);

  void Tune(size_t total_size, size_t unit);

  void DoTune(size_t total_size, std::vector<CacheMRCProfiler*> caches,
              size_t unit);

  void Access();

  bool CheckCache();

  void StartThread();

  void TuneLoop();

  void IncreaseNanos(uint64_t lru_nano, uint64_t profiler_nano);

 private:
  mutex mu_;
  std::atomic<uint64> num_active_threads_;
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::unique_ptr<CacheTuningStrategy> tuning_strategy_;
  std::map<std::string, CacheMRCProfiler*> registry_;

  std::atomic<uint64> access_count_;
  uint64 tuning_interval_;
  uint64 step_ = 1;

  std::atomic<uint64_t> lru_nanos;
  std::atomic<uint64_t> profiler_nanos;

  size_t total_size_;
  size_t min_size_;
  size_t tuning_unit_;

  bool clear_stat_;

  explicit CacheManager();
};

}  // namespace embedding
}  // namespace tensorflow
#endif  // DEEPREC_CACHE_MANAGER_H
