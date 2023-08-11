#ifndef DEEPREC_PROFILED_CACHE_H
#define DEEPREC_PROFILED_CACHE_H

#include <chrono>

#include "tensorflow/core/framework/embedding/cache_profiler.h"
#include "tensorflow/core/framework/embedding/cache_manager.h"

namespace tensorflow {
namespace embedding {
template<typename K>
class ProfiledLRUCache : public LRUCache<K> {
public:
    explicit ProfiledLRUCache(const std::string &name,
                              const size_t bucket_size,
                              const size_t max_reuse_time,
                              const uint64_t sampling_interval,
                              TunableCache *tunable_cache = nullptr
    ) : LRUCache<K>(name), profiler_(name, bucket_size, max_reuse_time, sampling_interval, tunable_cache) {

    }

//    void add_to_cache(const K *batch_ids, const size_t batch_size) override {
//      LRUCache<K>::add_to_cache(batch_ids, batch_size);
//      profiler_.ReferenceKeyBatch(batch_ids, batch_size);
//      CacheManager<K>::GetInstance().Access();
//    }

    SamplingLRUAETProfiler<K> *GetProfiler() {
      return &profiler_;
    }

    void update(const K *batch_ids, size_t batch_size, bool use_locking) override {
      using Clock = std::chrono::high_resolution_clock;
      auto start = Clock::now();
      LRUCache<K>::update(batch_ids, batch_size, use_locking);
      auto end_base = Clock::now();
      profiler_.ReferenceKeyBatch(batch_ids, batch_size);
      auto end_profiler = Clock::now();
      CacheManager<K>::GetInstance().Access();
      auto lru_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_base - start).count();
      auto profiler_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_profiler - end_base).count();
      CacheManager<K>::GetInstance().IncreaseNanos(lru_time, profiler_time);
    }

    ~ProfiledLRUCache() override {
      CacheManager<K>::GetInstance().UnregisterCache(profiler_.GetName());
    }

private:
    SamplingLRUAETProfiler<K> profiler_;
};

}
}
#endif //DEEPREC_PROFILED_CACHE_H
