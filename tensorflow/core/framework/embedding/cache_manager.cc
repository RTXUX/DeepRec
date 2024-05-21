#include "tensorflow/core/framework/embedding/cache_manager.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include "tensorflow/core/framework/embedding/cache_profiler.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"

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
  CacheProp* prop = new CacheProp();
  prop->profiler = &cache;
  registry_.insert(std::make_pair(cache.GetName(), std::unique_ptr<CacheProp>(prop)));
  registry2_.insert(std::make_pair(&cache, prop));

  if (min_size_specified_) {
    prop->min_size = min_size_;
  }

  __sync_fetch_and_add(&total_size_, cache.GetCacheSize());
  
  LOG(INFO) << "Registering Cache \"" << cache.GetName() << "\": "
            << "size=" << (double)cache.GetCacheSize() / (1024*1024) << "MB, entry_size=" << cache.GetCacheEntrySize()
            << ", total_size=" << (double)total_size_ / (1024*1024) << "MB";

  if (num_active_threads_ < 1) {
    StartThread();
  }
}

void CacheManager::UnregisterCache(const std::string& name) {
  mutex_lock lock(mu_);
  CacheMRCProfiler *cache = registry_.find(name)->second->profiler;
  registry_.erase(name);
  registry2_.erase(cache);
}

void CacheManager::Tune(size_t total_size, size_t unit) {
  mutex_lock lock(mu_);
  if (!sampling_active_.load(std::memory_order_relaxed)) return;
  std::vector<CacheProp*> caches;
  for (auto& kv : registry_) {
    caches.emplace_back(kv.second.get());
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
  lru_nanos.store(0, std::memory_order_relaxed);
  profiler_nanos.store(0, std::memory_order_relaxed);
}

void CacheManager::DoTune(size_t total_size,
                          std::vector<CacheProp*> props, size_t unit) {
  std::map<CacheMRCProfiler*, CacheItem> items;
  uint64_t orig_mc_sum = 0;

  for (auto prop : props) {
    CacheMRCProfiler* cache = prop->profiler;
    const size_t bucket_size = cache->GetBucketSize();
    const size_t size = cache->GetCacheSize();
    const size_t entry_size = cache->GetCacheEntrySize();
    const size_t min_size = prop->min_size;
    const size_t num_entries = size / entry_size;
    std::vector<double> mrc = cache->GetMRC(num_entries * 10);
    const double mr = InterpolateMRC(mrc, bucket_size, num_entries);
    const uint64_t vc = (uint64_t)mrc[mrc.size() - 1];
    const uint64_t mc = vc * mr;
    const double actual_hr = cache->GetHitRate();
    const uint64_t actual_hc = (uint64_t)(actual_hr * vc);
    LOG(INFO) << "Cache \"" << cache->GetName()
              << "\" visit count=" << vc
              << ", estimated hit count=" << vc - mc
              << ", actual hit count=" << actual_hc << ", relative error="
              << (double)(int64_t)(vc - mc - actual_hc) / actual_hc;
    orig_mc_sum += mc;
    items.emplace(std::piecewise_construct, std::forward_as_tuple(cache),
                  std::forward_as_tuple(bucket_size, size, size, entry_size, min_size, vc,
                                        mc, mr, std::move(mrc)));
    if (clear_stat_) {
      cache->ResetProfiling();
      cache->ResetStat();
    }
  }

  bool success = tuning_strategy_->DoTune(total_size, items, unit);
  
  if (success) {
    for (auto& kv : items) {
      kv.first->SetCacheSize(kv.second.new_size);
    }
    notune_counter_ = 0;
    for (auto prop: props) {
      prop->profiler->StopSamplingAndReleaseResource();
    }
    sampling_active_.store(false, std::memory_order_release);
    LOG(INFO) << "Stopped sampling";
    return;
  } else {
    notune_counter_++;
  }

  if (notune_counter_ > notune_threshold_) {
    sampling_active_.store(false, std::memory_order_release);
    for (auto prop : props) {
      prop->profiler->StopSamplingAndReleaseResource();
    }
    LOG(INFO) << notune_counter_ << "continuous tuning did not succeed, stop sampling!";
  }

  LOG(INFO) << "Tuning Done";
}

void CacheManager::Access(size_t size) {
  access_count_.fetch_add(1, std::memory_order_relaxed);
  const size_t target_size = total_size_ * 16;
  if (access_size_.fetch_add(size, std::memory_order_relaxed) + size >= target_size) {
    if (!access_size_lock_.test_and_set(std::memory_order_relaxed)) {
      should_tune_.store(true, std::memory_order_relaxed);
      access_size_.fetch_sub(target_size, std::memory_order_relaxed);
      access_size_lock_.clear(std::memory_order_relaxed);
    }
  }
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
    if (should_tune_.load(std::memory_order_relaxed)) {
      should_tune_.store(false, std::memory_order_relaxed);
      bool reactivate = false;
      for (auto &kv : registry2_) {
        CacheStat& stat = kv.second->stat;
        std::pair<uint64, uint64> move_count = kv.first->GetMoveCount();
        kv.first->ResetMoveCount();
        uint64 promotions = move_count.first, demotions = move_count.second;
        LOG(INFO) << "\"" << kv.first->GetName() << "\" promotions: " << promotions << ", demotions:" << demotions;
        uint64 prev_promotions = stat.prev_promotion, prev_demotions = stat.prev_demotion;
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
        stat.prev_promotion = promotions;
        stat.prev_demotion = demotions;
      }
      
      if (SamplingActive()) {
        LOG(INFO) << "access count: " << access_count_ << ", do tune";
        Tune(total_size_, tuning_unit_);
      } else {
        LOG(INFO) << "access count: " << access_count_ << ", tuning not active"; 
      }
      LOG(INFO) << "LRU Time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::nanoseconds(lru_nanos.load()))
                   .count()
            << "ms, Profiler Time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::nanoseconds(profiler_nanos.load()))
                   .count()
            << "ms";
      step_ = std::round(access_count_.load(std::memory_order_relaxed) /
                          (tuning_interval_ * cache_count)) + 1;

      reactivate = false;
      if (reactivate) {
        notune_counter_ = 0;
        for (auto &kv: registry_) {
          kv.second->profiler->StartSampling();
        }
        sampling_active_.store(true, std::memory_order_release);
      }
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

void CacheManager::NotifyBatchSize(CacheMRCProfiler* profiler, size_t batch_size) {
  if (min_size_specified_) return;
  CacheProp& prop = *registry2_[profiler];
  if (batch_size <= prop.max_batch_size) return;
  mutex_lock lock(prop.mu);
  if (batch_size <= prop.max_batch_size) return;
  prop.max_batch_size = batch_size;
  prop.min_size = 2 * batch_size;
  LOG(INFO) << "Setting min_size of " << profiler->GetName() << " to " << prop.min_size;
}

CacheManager::CacheManager()
    : thread_pool_(std::make_unique<thread::ThreadPool>(
          Env::Default(), ThreadOptions(), "CACHE_MANAGER", 1, false)),
      access_count_(0),
      lru_nanos(0),
      profiler_nanos(0),
      sampling_active_(true),
      should_tune_(false),
      access_size_(0),
      total_size_(0) {
  ReadInt64FromEnvVar("CACHE_TUNING_INTERVAL", 100000,
                      reinterpret_cast<int64*>(&tuning_interval_));
  if (ReadInt64FromEnvVar("CACHE_MIN_SIZE", 0,
                      reinterpret_cast<int64*>(&min_size_)) == Status::OK() && min_size_ != 0) {
    min_size_specified_ = true;
    LOG(INFO) << "Using min_size " << min_size_;
  } else {
    min_size_specified_ = false;
    min_size_ = 0;
    LOG(INFO) << "min_size: auto";
  }
  ReadInt64FromEnvVar("CACHE_BLOCKS", 0, reinterpret_cast<int64*>(&num_cache_blocks_));
  ReadInt64FromEnvVar("CACHE_TUNING_UNIT", 0,
                      reinterpret_cast<int64*>(&tuning_unit_));
  if (num_cache_blocks_ == 0 && tuning_unit_ == 0) {
    num_cache_blocks_ = 16384;
    LOG(INFO) << "CACHE_BLOCKS not set, using default " << num_cache_blocks_;
  }
  if (tuning_unit_ == 0){
    tuning_unit_ = total_size_ / num_cache_blocks_;
    LOG(INFO) << "CACHE_TUNING_UNIT not set, using fixed block number " << num_cache_blocks_ << ", now unit is " << tuning_unit_;
  };
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
