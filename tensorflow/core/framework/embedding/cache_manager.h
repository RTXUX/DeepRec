#ifndef DEEPREC_CACHE_MANAGER_H
#define DEEPREC_CACHE_MANAGER_H

#include <memory>
#include <map>
#include <random>

#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/embedding/cache_profiler.h"

namespace tensorflow {
namespace embedding {

static inline double InterpolateMRC(const std::vector<double> &mrc, size_t bucket_size, size_t target) {
  double bucket = (double) target / bucket_size;
  size_t bucket_int = std::floor(bucket);
  if (bucket_int >= mrc.size() - 2) {
    return mrc[mrc.size() - 2];
  }
  if (mrc.size() == 2) {
    return mrc[0];
  }
  double interpolated_mr = mrc[bucket_int] + (bucket - (double) bucket_int) * (mrc[bucket_int + 1] - mrc[bucket_int]);
  return interpolated_mr;
}

class MockTunableCache : public TunableCache {
public:
    explicit MockTunableCache(size_t numEntries);

    size_t GetCacheSize() const override;

    void SetCacheSize(size_t new_size) override;

    size_t GetCacheEntrySize() const override;

private:
    size_t num_entries_;
};

class CacheItem {
public:
    size_t bucket_size;
    size_t orig_size;
    size_t new_size;
    size_t entry_size;
    double mr;
    std::vector<double> mrc;

    CacheItem(size_t bucketSize, size_t origSize, size_t newSize, size_t entrySize, double mr,
              const std::vector<double> &mrc);

    CacheItem();
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
      RandomApportion(parts, total_size_);
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
      for (auto cache: caches) {
        cache->ResetProfiling();
        cache->ResetStat();
      }
    }

    void DoTune(size_t total_size, std::vector<CacheMRCProfiler<K> *> caches, size_t unit) {

      std::map<CacheMRCProfiler<K> *, CacheItem> items;
      double orig_mr_sum = 0.0;

      for (auto cache: caches) {

        const size_t bucket_size = cache->GetBucketSize();
        const size_t size = cache->GetCacheSize();
        const size_t entry_size = cache->GetCacheEntrySize();
        const size_t num_entries = size / entry_size;
        std::vector<double> mrc = cache->GetMRC(size * 10);
        const double mr = InterpolateMRC(mrc, bucket_size, num_entries);
        const double actual_hr = cache->GetHitRate();
        LOG(INFO) << "Cache \"" << cache->GetName() << "\" estimated hit rate=" << 1.0 - mr << ", actual hit rate="
                  << actual_hr << ", error=" << (1.0 - mr) - actual_hr;
        orig_mr_sum += mr;
        items.emplace(std::piecewise_construct, std::forward_as_tuple(cache),
                      std::forward_as_tuple(bucket_size, size, size, entry_size, mr, std::move(mrc)));
        cache->ResetProfiling();
      }

      // do random apportion and compute new MR
      {
        std::vector<size_t> parts(items.size());
        RandomApportion(parts, total_size);
        size_t i = 0;
        for (auto &item: items) {
          size_t new_size = parts[i++];
          size_t new_entries = new_size / item.second.entry_size;
          item.second.new_size = new_size;
          item.second.mr = InterpolateMRC(item.second.mrc, item.second.bucket_size, new_entries);
        }
      }

      while (true) {
        double max_gain = 0.0, min_loss = 0.0, gain_new_mr = 0.0, loss_new_mr = 0.0;
        CacheMRCProfiler<K> *max_gain_cache = nullptr, *min_loss_cache = nullptr;
        for (auto &item: items) {
          const size_t current_size = item.second.new_size;
          const size_t new_entries = (item.second.new_size + unit) / item.second.entry_size;
          const double new_mr = InterpolateMRC(item.second.mrc, item.second.bucket_size, new_entries);
          const double gain = item.second.mr - new_mr;
          if (gain > max_gain || max_gain_cache == nullptr) {
            max_gain = gain;
            max_gain_cache = item.first;
            gain_new_mr = new_mr;
          }
        }

        for (auto &item: items) {
          if (item.first == max_gain_cache) continue;
          const size_t current_size = item.second.new_size;
          if (current_size <= min_size_ + unit) {
            continue;
          }
          const ssize_t new_entries = (item.second.new_size - unit) / item.second.entry_size;

          const double new_mr = InterpolateMRC(item.second.mrc, item.second.bucket_size, new_entries);
          const double loss = new_mr - item.second.mr;
          if (loss < min_loss || min_loss_cache == nullptr) {
            min_loss = loss;
            min_loss_cache = item.first;
            loss_new_mr = new_mr;
          }
        }

        if (max_gain <= min_loss || max_gain_cache == nullptr || min_loss_cache == nullptr) break;

        items[max_gain_cache].new_size += unit;
        items[max_gain_cache].mr = gain_new_mr;
        items[min_loss_cache].new_size -= unit;
        items[min_loss_cache].mr = loss_new_mr;
      }

      double new_mr_sum = 0.0;
      for (auto &item: items) {
        new_mr_sum += item.second.mr;
      }
      LOG(INFO) << "orig MRs=" << orig_mr_sum << ", new MRs=" << new_mr_sum << ", diff=" << orig_mr_sum - new_mr_sum;
      if (new_mr_sum >= orig_mr_sum) {
        LOG(INFO) << "new MRs not less than original MRs, not tuning cache";
        return;
      }

      for (auto &item: items) {
        LOG(INFO) << "Change size of \"" << item.first->GetName() << "\" to " << item.second.new_size;
        item.first->SetCacheSize(item.second.new_size);
      }
    }

    void RandomApportion(std::vector<size_t> &parts, size_t total) {
      const size_t resv_size = parts.size() * min_size_;
      const size_t part_size = total - resv_size;
      if (resv_size >= total) {
        LOG(FATAL) << "Not enough size to partition";
      }
      const size_t num_parts = parts.size();
      std::random_device rd;
      std::default_random_engine re(rd());
      std::uniform_real_distribution<double> uniform(0, 1);
      std::uniform_int_distribution<size_t> pick(0, num_parts - 1);
      std::vector<double> apportion(num_parts);
      double normalize_sum = 0.0;
      for (auto &part: apportion) {
        const double sample = uniform(re);
        part = -std::log(sample);
        normalize_sum += part;
      }
      for (auto &part: apportion) {
        part /= normalize_sum;
      }
      size_t sum_apportion = 0;
      for (size_t i = 0; i < num_parts; ++i) {
        auto part = (size_t) std::round(apportion[i] * part_size);
        sum_apportion += part;
        parts[i] = part;
      }
      ssize_t remaining = part_size - sum_apportion;
      ssize_t step = remaining > 0 ? 1 : -1;
      while (remaining != 0) {
        auto picked_part = pick(re);
        if ((ssize_t) parts[picked_part] + step > 0) {
          parts[picked_part] += step;
          remaining -= step;
        }
      }
      for (auto &part: parts) {
        part += min_size_;
      }
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

private:
    mutex mu_;
    std::atomic<uint64> num_active_threads_;
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
    std::unique_ptr<thread::ThreadPool> thread_pool_;
    std::map<std::string, CacheMRCProfiler<K> *> registry_;

    std::atomic<uint64> access_count_;
    uint64 tuning_interval_;
    uint64 step_ = 1;

    size_t total_size_;
    size_t min_size_;
    size_t tuning_unit_;

    static const size_t total_cache_size = 1 * 1024 * 1024 * 1024;

    explicit CacheManager() : thread_pool_(
            std::make_unique<thread::ThreadPool>(Env::Default(), ThreadOptions(), "CACHE_MANAGER", 1, false)),
                              access_count_(0) {
      ReadInt64FromEnvVar("CACHE_TUNING_INTERVAL", 100000, reinterpret_cast<int64 *>(&tuning_interval_));
      ReadInt64FromEnvVar("CACHE_TOTAL_SIZE", 32 * 1024 * 1024, reinterpret_cast<int64 *>(&total_size_));
      ReadInt64FromEnvVar("CACHE_MIN_SIZE", 2048 * 128 * 8, reinterpret_cast<int64 *>(&min_size_));
      ReadInt64FromEnvVar("CACHE_TUNING_UNIT", 8 * 128, reinterpret_cast<int64 *>(&tuning_unit_));
      num_active_threads_ = 0;
    }
};

} // namespace embedding
} // namespace tensorflow
#endif //DEEPREC_CACHE_MANAGER_H
