#ifndef DEEPREC_PROFILED_CACHE_H
#define DEEPREC_PROFILED_CACHE_H

#include <chrono>

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/cache_manager.h"
#include "tensorflow/core/framework/embedding/cache_profiler.h"

namespace tensorflow {
namespace embedding {
template <typename K>
class ProfiledLRUCache : public LRUCache<K> {
 public:
  explicit ProfiledLRUCache(const std::string& name, const size_t bucket_size,
                            const size_t max_reuse_time,
                            const uint64_t sampling_interval,
                            TunableCache* tunable_cache = nullptr)
      : LRUCache<K>(name),
        profiler_(name, bucket_size, max_reuse_time, sampling_interval,
                  tunable_cache),
        entry_size(-1),
        tunable_cache_(tunable_cache),
        cm_(CacheManager::GetInstance()) {}

  //    void add_to_cache(const K *batch_ids, const size_t batch_size) override
  //    {
  //      LRUCache<K>::add_to_cache(batch_ids, batch_size);
  //      profiler_.ReferenceKeyBatch(batch_ids, batch_size);
  //      CacheManager<K>::GetInstance().Access();
  //    }

  SamplingLRUAETProfiler<K>* GetProfiler() { return &profiler_; }

  void update(const K* batch_ids, size_t batch_size,
              bool use_locking) override {
    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();
    auto prev_entry_size = entry_size;
    if (entry_size >= 0xFFFFFFFFL) {
      entry_size = tunable_cache_->GetCacheEntrySize();
    }
    
    auto lru_start = Clock::now();
    LRUCache<K>::update(batch_ids, batch_size, use_locking);
    auto lru_end = Clock::now();
    if (cm_.SamplingActive()) {
      profiler_.ReferenceKeyBatch(batch_ids, batch_size);
    }
    if (entry_size < 0xFFFFFFFFL) {
      const size_t access_size = batch_size * entry_size;
      if (prev_entry_size != entry_size) {
        cm_.NotifyBatchSize(&profiler_, access_size);
      }
      cm_.Access(access_size);
    }
    
    // auto end_profiler = Clock::now();
    
    // auto lru_time =
    //     std::chrono::duration_cast<std::chrono::nanoseconds>(end_base - start)
    //         .count();
    // auto profiler_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
    //                          end_profiler - end_base)
    //                          .count();
    // cm_.IncreaseNanos(lru_time, profiler_time);
    auto end = Clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto lru_time = std::chrono::duration_cast<std::chrono::nanoseconds>(lru_end - lru_start).count();
    // cm_.IncreaseNanos(lru_time, time);
  }

  ~ProfiledLRUCache() override {
    cm_.UnregisterCache(profiler_.GetName());
  }

 private:
  SamplingLRUAETProfiler<K> profiler_;
  size_t entry_size;
  TunableCache *tunable_cache_;
  CacheManager& cm_;
};

template <typename K>
class ProfiledShardedLRUCache : public ShardedLRUCache<K> {
 public:
  explicit ProfiledShardedLRUCache(const std::string& name, const size_t bucket_size,
                         const size_t max_reuse_time,
                         const uint64_t sampling_interval,
                         const int shard_shift,
                         TunableCache* tunable_cache = nullptr)
    : ShardedLRUCache<K>(name, shard_shift),
    profiler_(name, bucket_size, max_reuse_time, sampling_interval,
              tunable_cache),
    entry_size(-1),
    tunable_cache_(tunable_cache),
    cm_(CacheManager::GetInstance()) {}

  SamplingLRUAETProfiler<K>* GetProfiler() { return &profiler_; }

  void update(const K* batch_ids, size_t batch_size,
              bool use_locking) override {
    if (entry_size >= 0xFFFFFFFFL) {
      entry_size = tunable_cache_->GetCacheEntrySize();
    }
    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();
    ShardedLRUCache<K>::update(batch_ids, batch_size, use_locking);
    auto end_base = Clock::now();
    if (cm_.SamplingActive()) {
      profiler_.ReferenceKeyBatch(batch_ids, batch_size);
    }
    auto end_profiler = Clock::now();
    if (entry_size < 0xFFFFFFFFL) {
      const size_t access_size = batch_size * entry_size;
      cm_.NotifyBatchSize(&profiler_, access_size);
      cm_.Access(access_size);
    }
    auto lru_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_base - start)
            .count();
    auto profiler_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             end_profiler - end_base)
                             .count();
    cm_.IncreaseNanos(lru_time, profiler_time);
  }

  ~ProfiledShardedLRUCache() {
    cm_.UnregisterCache(profiler_.GetName());
  }

 private:
  SamplingLRUAETProfiler<K> profiler_;
  size_t entry_size = -1;
  TunableCache *tunable_cache_;
  CacheManager& cm_;
};

}  // namespace embedding
}  // namespace tensorflow
#endif  // DEEPREC_PROFILED_CACHE_H
