#include "tensorflow/core/framework/embedding/cache_manager.h"

#include <cstdlib>

namespace tensorflow {
namespace embedding {

size_t MockTunableCache::GetCacheSize() const {
  return num_entries_ * GetCacheEntrySize();
}

void MockTunableCache::SetCacheSize(size_t new_size) {
  num_entries_ = new_size / GetCacheEntrySize();
}

size_t MockTunableCache::GetCacheEntrySize() const { return 8; }

MockTunableCache::MockTunableCache(size_t size)
    : num_entries_(size / GetCacheEntrySize()) {}

CacheManager& CacheManager::GetInstance() {
  const static std::unique_ptr<CacheManager> instance_(new CacheManager());
  return *instance_;
}

void CacheManager::RegisterCache(CacheMRCProfiler& cache) {
  mutex_lock lock(mu_);
  if (registry_.find(cache.GetName()) != registry_.cend()) {
    // TODO: name conflict
  }
  registry_[cache.GetName()] = &cache;
  cache_stats_.emplace(std::make_pair(&cache, CacheStat()));
  std::vector<size_t> parts(registry_.size());
  // RandomApportion(parts, total_size_);
  size_t size = total_size_ / registry_.size();
  for (auto& p : parts) {
    p = size;
  }
  size_t i = 0;
  for (auto& kv : registry_) {
    kv.second->SetCacheSize(parts[i++]);
  }
  if (num_active_threads_ < 1) {
    StartThread();
  }
}

void CacheManager::UnregisterCache(const std::string& name) {
  mutex_lock lock(mu_);
  CacheMRCProfiler *cache = registry_.find(name)->second;
  registry_.erase(name);
  cache_stats_.erase(cache);
}

void CacheManager::Tune(size_t total_size, size_t unit) {
  mutex_lock lock(mu_);
  if (!sampling_active_.load(std::memory_order_relaxed)) return;
  std::vector<CacheMRCProfiler*> caches;
  for (auto& kv : registry_) {
    caches.emplace_back(kv.second);
  }
  DoTune(total_size, std::move(caches), unit);
  LOG(INFO) << "LRU Time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::nanoseconds(lru_nanos.load()))
                   .count()
            << "ms, Profiler Time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::nanoseconds(profiler_nanos.load()))
                   .count()
            << "ms";
}

void CacheManager::DoTune(size_t total_size,
                          std::vector<CacheMRCProfiler*> caches, size_t unit) {
  std::map<CacheMRCProfiler*, CacheItem> items;
  uint64_t orig_mc_sum = 0;

  for (auto cache : caches) {
    const size_t bucket_size = cache->GetBucketSize();
    const size_t size = cache->GetCacheSize();
    const size_t entry_size = cache->GetCacheEntrySize();
    const size_t num_entries = size / entry_size;
    std::vector<double> mrc = cache->GetMRC(size * 10);
    const double mr = InterpolateMRC(mrc, bucket_size, num_entries);
    const uint64_t vc = (uint64_t)mrc[mrc.size() - 1];
    const uint64_t mc = vc * mr;
    const double actual_hr = cache->GetHitRate();
    const uint64_t actual_hc = (uint64_t)(actual_hr * vc);
    LOG(INFO) << "Cache \"" << cache->GetName()
              << "\" estimated hit count=" << vc - mc
              << ", actual hit count=" << actual_hc << ", relative error="
              << (double)(int64_t)(vc - mc - actual_hc) / actual_hc;
    orig_mc_sum += mc;
    items.emplace(std::piecewise_construct, std::forward_as_tuple(cache),
                  std::forward_as_tuple(bucket_size, size, size, entry_size, vc,
                                        mc, mr, std::move(mrc)));
    if (clear_stat_) {
      cache->ResetProfiling();
      cache->ResetStat();
    }
  }

  bool success = tuning_strategy_->DoTune(total_size, items, unit, min_size_);
  if (success) {
    for (auto& kv : items) {
      kv.first->SetCacheSize(kv.second.new_size);
    }
    notune_counter_ = 0;
  } else {
    notune_counter_++;
  }

  if (notune_counter_ > notune_threshold_) {
    sampling_active_.store(false, std::memory_order_release);
    for (auto cache : caches) {
      cache->ResetProfiling();
    }
    LOG(INFO) << notune_counter_ << "continuous tuning did not succeed, stop sampling!";
  }

  LOG(INFO) << "Tuning Done";
}

void CacheManager::Access() {
  access_count_.fetch_add(1, std::memory_order_relaxed);
}

bool CacheManager::CheckCache() {
  mutex_lock l(mu_);
  return !registry_.empty();
}

void CacheManager::StartThread() {
  while (flag_.test_and_set(std::memory_order_acquire))
    ;
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

void CacheManager::TuneLoop() {
  LOG(INFO) << "Tuning Loop Begin";
  while (CheckCache()) {
    LOG(INFO) << "access count: "
              << access_count_.load(std::memory_order_relaxed);
    size_t cache_count = registry_.size();
    if (access_count_.load(std::memory_order_relaxed) >
        step_ * tuning_interval_ * cache_count) {
      bool reactivate = false;
      for (auto &kv : cache_stats_) {
        std::pair<uint64, uint64> move_count = kv.first->GetMoveCount();
        kv.first->ResetMoveCount();
        uint64 promotions = move_count.first, demotions = move_count.second;
        LOG(INFO) << "\"" << kv.first->GetName() << "\" promotions: " << promotions << ", demotions:" << demotions;
        uint64 prev_promotions = kv.second.prev_promotion, prev_demotions = kv.second.prev_demotion;
        // skip if there is no promotion
        if (prev_promotions != 0) {
          int64 diff = prev_promotions - promotions;
          double relative_diff = (std::fabs((double)diff)) / prev_promotions;
          if (relative_diff > 0.2) {
            reactivate = true;
            LOG(INFO) << "\"" << kv.first->GetName() << "\" promotion diff: " << relative_diff << ", reactivating sampling";
          }
        }
        if (prev_demotions != 0) {
          int64 diff = prev_demotions - demotions;
          double relative_diff = (std::fabs((double)diff)) / prev_demotions;
          if (relative_diff > 0.2) {
            reactivate = true;
            LOG(INFO) << "\"" << kv.first->GetName() << "\" demotion diff: " << relative_diff << ", reactivating sampling";
          }
        }
        kv.second.prev_promotion = promotions;
        kv.second.prev_demotion = demotions;
      }
      if (reactivate) {
        notune_counter_ = 0;
        sampling_active_.store(true, std::memory_order_release);
      }
      if (SamplingActive()) {
        LOG(INFO) << "access count: " << access_count_ << ", do tune";
        Tune(total_size_, tuning_unit_);
        
      } else {
        LOG(INFO) << "access count: " << access_count_ << ", tuning not active"; 
      }
      step_ = std::round(access_count_.load(std::memory_order_relaxed) /
                          (tuning_interval_ * cache_count)) + 1;
    }
    Env::Default()->SleepForMicroseconds(1000000);
  }
  num_active_threads_.fetch_sub(1, std::memory_order_relaxed);
  LOG(INFO) << "Tuning thread exit";
}

void CacheManager::IncreaseNanos(uint64_t lru_nano, uint64_t profiler_nano) {
  lru_nanos.fetch_add(lru_nano, std::memory_order_relaxed);
  profiler_nanos.fetch_add(profiler_nano, std::memory_order_relaxed);
}

bool CacheManager::SamplingActive() const {
  return sampling_active_.load(std::memory_order_relaxed);
}

CacheManager::CacheManager()
    : thread_pool_(std::make_unique<thread::ThreadPool>(
          Env::Default(), ThreadOptions(), "CACHE_MANAGER", 1, false)),
      access_count_(0),
      lru_nanos(0),
      profiler_nanos(0),
      sampling_active_(true) {
  ReadInt64FromEnvVar("CACHE_TUNING_INTERVAL", 100000,
                      reinterpret_cast<int64*>(&tuning_interval_));
  ReadInt64FromEnvVar("CACHE_TOTAL_SIZE", 32 * 1024 * 1024,
                      reinterpret_cast<int64*>(&total_size_));
  ReadInt64FromEnvVar("CACHE_MIN_SIZE", 2048 * 128 * 8,
                      reinterpret_cast<int64*>(&min_size_));
  ReadInt64FromEnvVar("CACHE_TUNING_UNIT", 8 * 128,
                      reinterpret_cast<int64*>(&tuning_unit_));
  std::string tuning_strategy_name;
  ReadStringFromEnvVar("CACHE_TUNING_STRATEGY", "min_mc_random_greedy",
                       &tuning_strategy_name);
  tuning_strategy_.reset(
      CacheTuningStrategyCreator::Create(tuning_strategy_name));
  num_active_threads_ = 0;
  ReadBoolFromEnvVar("CACHE_PROFLER_CLEAR", true, &clear_stat_);
  ReadInt64FromEnvVar("CACHE_STABLE_STEPS", 5, reinterpret_cast<int64*>(&notune_threshold_));
}

}  // namespace embedding
}  // namespace tensorflow
