#ifndef DEEPREC_CACHE_MANAGER_H
#define DEEPREC_CACHE_MANAGER_H

#include <memory>
#include <map>
#include <random>

#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/embedding/cache_profiler.h"
#include "tensorflow/core/framework/embedding/cache_tuning_strategy.h"

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

template<typename K>
class CacheManager {
public:

    static CacheManager<K> &GetInstance() {
      const static std::unique_ptr<CacheManager<K>> instance_(new CacheManager<K>());
      return *instance_;
    }

    void RegisterCache(CacheMRCProfiler<K> &cache) {
      mutex_lock lock(mu_);
      if (registry_.find(cache.GetName()) != registry_.cend()) {
        // TODO: name conflict
      }
      registry_[cache.GetName()] = &cache;
      std::vector<size_t> parts(registry_.size());
      // RandomApportion(parts, total_size_);
      size_t size = total_size_ / registry_.size();
      for (auto &p : parts) {
        p = size;
      }
      size_t i = 0;
      for (auto &kv: registry_) {
        kv.second->SetCacheSize(parts[i++]);
      }
      if (num_active_threads_ < 1) {
        StartThread();
      }
    }

    void UnregisterCache(const std::string &name) {
      mutex_lock lock(mu_);
      registry_.erase(name);
    }

    void Tune(size_t total_size, size_t unit) {
      mutex_lock lock(mu_);
      std::vector<CacheMRCProfiler<K> *> caches;
      for (auto &kv: registry_) {
        caches.emplace_back(kv.second);
      }
      DoTune(total_size, std::move(caches), unit);
      LOG(INFO) << "LRU Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::nanoseconds(lru_nanos.load())).count() << "ms, Profiler Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::nanoseconds(profiler_nanos.load())).count() << "ms";
    }

    void DoTune(size_t total_size, std::vector<CacheMRCProfiler<K> *> caches, size_t unit) {

      std::map<CacheMRCProfiler<K> *, CacheItem> items;
      uint64_t orig_mc_sum = 0;

      for (auto cache: caches) {

        const size_t bucket_size = cache->GetBucketSize();
        const size_t size = cache->GetCacheSize();
        const size_t entry_size = cache->GetCacheEntrySize();
        const size_t num_entries = size / entry_size;
        std::vector<double> mrc = cache->GetMRC(size * 10);
        const double mr = InterpolateMRC(mrc, bucket_size, num_entries);
        const uint64_t vc = (uint64_t) mrc[mrc.size() - 1];
        const uint64_t mc = vc * mr;
        const double actual_hr = cache->GetHitRate();
        const uint64_t actual_hc = (uint64_t) (actual_hr * vc);
        LOG(INFO) << "Cache \"" << cache->GetName() << "\" estimated hit count=" << vc - mc << ", actual hit count="
                  << actual_hc << ", relative error=" << (double) (int64_t) (vc - mc - actual_hc) / actual_hc;
        orig_mc_sum += mc;
        items.emplace(std::piecewise_construct, std::forward_as_tuple(cache),
                      std::forward_as_tuple(bucket_size, size, size, entry_size, vc, mc, mr, std::move(mrc)));
        // cache->ResetProfiling();
        // cache->ResetStat();
      }

      bool success = tuning_strategy_->DoTune(total_size, items, unit, min_size_);
      if (success) {
        for (auto &kv: items) {
          kv.first->SetCacheSize(kv.second.new_size);
        }
      }

      LOG(INFO) << "Tuning Done";
    }



    void Access() {
      access_count_.fetch_add(1, std::memory_order_relaxed);
    }

    bool CheckCache() {
      mutex_lock l(mu_);
      return !registry_.empty();
    }

    void StartThread() {
      while (flag_.test_and_set(std::memory_order_acquire));
      if (num_active_threads_ < 1) {
        num_active_threads_.fetch_add(1, std::memory_order_relaxed);
        LOG(INFO) << "Scheduling Tuning Thread";
        thread_pool_->Schedule([this]() {
            LOG(INFO) << "Scheduled Tuning Thread";
            this->TuneLoop();
        });
      }
      flag_.clear(std::memory_order_release);
    }

    void TuneLoop() {
      LOG(INFO) << "Tuning Loop Begin";
      while (CheckCache()) {
        LOG(INFO) << "access count: " << access_count_.load(std::memory_order_relaxed);
        size_t cache_count = registry_.size();
        if (access_count_.load(std::memory_order_relaxed) > step_ * tuning_interval_ * cache_count) {
          LOG(INFO) << "access count: " << access_count_ << ", do tune";
          Tune(total_size_, tuning_unit_);
          step_ = std::round(access_count_.load(std::memory_order_relaxed) / (tuning_interval_ * cache_count)) + 1;
        }
        Env::Default()->SleepForMicroseconds(1000000);
      }
      num_active_threads_.fetch_sub(1, std::memory_order_relaxed);
      LOG(INFO) << "Tuning thread exit";
    }

    void IncreaseNanos(uint64_t lru_nano, uint64_t profiler_nano) {
      lru_nanos.fetch_add(lru_nano, std::memory_order_relaxed);
      profiler_nanos.fetch_add(profiler_nano, std::memory_order_relaxed);
    }

private:
    mutex mu_;
    std::atomic<uint64> num_active_threads_;
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::unique_ptr<CacheTuningStrategy<K>> tuning_strategy_;
  std::map<std::string, CacheMRCProfiler<K> *> registry_;

    std::atomic<uint64> access_count_;
    uint64 tuning_interval_;
    uint64 step_ = 1;

    std::atomic<uint64_t> lru_nanos;
    std::atomic<uint64_t> profiler_nanos;

    size_t total_size_;
    size_t min_size_;
    size_t tuning_unit_;

    static const size_t total_cache_size = 1 * 1024 * 1024 * 1024;

    explicit CacheManager() : thread_pool_(
            std::make_unique<thread::ThreadPool>(Env::Default(), ThreadOptions(), "CACHE_MANAGER", 1, false)),
                              access_count_(0), lru_nanos(0), profiler_nanos(0) {
      ReadInt64FromEnvVar("CACHE_TUNING_INTERVAL", 100000, reinterpret_cast<int64 *>(&tuning_interval_));
      ReadInt64FromEnvVar("CACHE_TOTAL_SIZE", 32 * 1024 * 1024, reinterpret_cast<int64 *>(&total_size_));
      ReadInt64FromEnvVar("CACHE_MIN_SIZE", 2048 * 128 * 8, reinterpret_cast<int64 *>(&min_size_));
      ReadInt64FromEnvVar("CACHE_TUNING_UNIT", 8 * 128, reinterpret_cast<int64 *>(&tuning_unit_));
      std::string tuning_strategy_name;
      ReadStringFromEnvVar("CACHE_TUNING_STRATEGY", "min_mc_random_greedy", &tuning_strategy_name);
      tuning_strategy_.reset(CacheTuningStrategyCreator<K>::Create(tuning_strategy_name));
      num_active_threads_ = 0;
    }
};

} // namespace embedding
} // namespace tensorflow
#endif //DEEPREC_CACHE_MANAGER_H
