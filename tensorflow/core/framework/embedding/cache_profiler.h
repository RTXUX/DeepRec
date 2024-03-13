#ifndef DEEPREC_CACHE_PROFILER_H
#define DEEPREC_CACHE_PROFILER_H

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <malloc.h>

#include "sparsehash/dense_hash_map_lockless"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace embedding {

class TunableCache {
 public:
  virtual size_t GetCacheSize() const = 0;

  virtual void SetCacheSize(size_t new_size) = 0;

  virtual size_t GetCacheEntrySize() const = 0;

  virtual double GetHitRate() const = 0;

  virtual void ResetStat() = 0;

  virtual std::pair<uint64, uint64> GetMoveCount() const = 0;

  virtual void ResetMoveCount() = 0;
};

template <typename K>
class CacheMRCProfilerFeeder {
 public:
  virtual void ReferenceKey(const K& key) = 0;

  virtual void ReferenceKeyBatch(const K* keys, size_t batch_size) = 0;
};

class CacheMRCProfiler : public virtual TunableCache {
 public:
  virtual std::vector<double> GetMRC(uint64_t max_cache_size) const = 0;

  virtual const std::string& GetName() const = 0;

  virtual void ResetProfiling() = 0;

  virtual size_t GetBucketSize() const = 0;

  virtual void StopSamplingAndReleaseResource() = 0;

  virtual void StartSampling() = 0;
};

#if DEBUG_DUMP
static std::atomic<size_t> dumped;
#endif

template <typename K>
class SamplingLRUAETProfiler : public virtual CacheMRCProfilerFeeder<K>,
                               public virtual CacheMRCProfiler {
 private:
  using HistogramMap = google::dense_hash_map_lockless<int32_t, uint32_t*>;
 public:
  explicit SamplingLRUAETProfiler(std::string name, const size_t bucket_size,
                                  const size_t max_reuse_time,
                                  const uint64_t sampling_interval,
                                  TunableCache* tunable_cache = nullptr)
      : name_(std::move(name)),
        bucket_size_(bucket_size),
        timestamp_(0),
        sampling_interval_(sampling_interval),
        samping_rate_(1.0 / sampling_interval),
        rand_(std::random_device()()),
        distrib_(1, sampling_interval * 2 - 1),
        distrib_real_(0.0, 1.0),
        run_lock_(false),
        run_(0),
        tunable_cache_(tunable_cache) {
    ResetLastAccessMap();
    ResetHistogram();
    #if DEBUG_DUMP
    dumped.store(0, std::memory_order_relaxed);
    #endif
  }

  void ReferenceKey(const K& key) override {
    // if resetting, just skip
    if (run_lock_.load(std::memory_order_acquire)) return;
    // indicate we are running
    run_.fetch_add(1, std::memory_order_acquire);
    if (run_lock_.load(std::memory_order_acquire)) {
      run_.fetch_sub(1, std::memory_order_release);
      return;
    };
    DoReferenceKey(key);
    run_.fetch_sub(1, std::memory_order_release);
  }

  void ReferenceKeyBatch(const K* keys, const size_t batch_size) override {
    if (run_lock_.load(std::memory_order_acquire)) return;
    // indicate we are running
    run_.fetch_add(1, std::memory_order_acquire);
    if (run_lock_.load(std::memory_order_acquire)) {
      run_.fetch_sub(1, std::memory_order_release);
      return;
    };
    for (size_t i = 0; i < batch_size; ++i) {
      DoReferenceKey(keys[i]);
    }
    run_.fetch_sub(1, std::memory_order_release);
  }

  const std::string& GetName() const override { return name_; }

  void ResetProfiling() override {
    // avoid new thread to enter profiling
    run_lock_.store(true, std::memory_order_acquire);
    // wait until no thread profiling
    while (run_.load(std::memory_order_acquire) != 0);
    timestamp_.store(0, std::memory_order_relaxed);
    ResetHistogram();
    ResetLastAccessMap();
    malloc_trim(0);
    sample_time_ = timestamp_.load(std::memory_order_relaxed);
    run_lock_.store(false, std::memory_order_release);
  }

  size_t GetBucketSize() const override { return bucket_size_; }

  std::vector<double> GetMRC(uint64_t max_cache_size) const override {
    if (run_lock_.load(std::memory_order_acquire)) {
      return {1.0, timestamp_.load(std::memory_order_relaxed)};
    }
    // prevent releasing
    std::atomic<uint>& run__ = const_cast<std::atomic<uint>&>(run_);
    run__.fetch_add(1, std::memory_order_acquire);
    uint64_t cold_miss = 0;
    
    std::vector<std::pair<int32_t, uint32_t>> reuse_time_hist;
    reuse_time_hist.reserve(reuse_time_hist_->size_lockless());
    reuse_time_hist.emplace_back(0, 0);
    for (auto iter = reuse_time_hist_->cbegin(); iter != reuse_time_hist_->cend(); ++iter) {
      if (iter->first == 0) {
        cold_miss = *iter->second;
      } else {
        reuse_time_hist.emplace_back(iter->first, *iter->second);
      }
    }
    const size_t num_elem = reuse_time_hist.size();
    const uint64_t timestamp = timestamp_.load(std::memory_order_relaxed);
    uint64_t reuse_time_sum = 0;
    
    if (sampling_interval_ != 1) {
      cold_miss = 0;
      for (auto iter = last_access_map_->cbegin();
         iter != last_access_map_->cend(); ++iter) {
        if (*(iter->second) != 0) {
          cold_miss += 1;
        }
      }
    }
    reuse_time_sum += cold_miss;
    const size_t max_dist = sampling_interval_ == 1 ? last_access_map_->size_lockless() : timestamp;
    std::sort(reuse_time_hist.begin(), reuse_time_hist.end());

    size_t last_index = 0;
    std::vector<uint64_t> prefix_sum;
    prefix_sum.reserve(num_elem);
    prefix_sum.emplace_back(0);
    for (size_t i = 1; i < num_elem; ++i) {
      prefix_sum.emplace_back(prefix_sum[last_index] + reuse_time_hist[i].second);
      reuse_time_sum += reuse_time_hist[i].second;
      last_index = i;
    }

    // calculate CCDF
    std::vector<double> prob_greater;
    prob_greater.reserve(num_elem);
    prob_greater.emplace_back(1.0);
    for (size_t i = 1; i < num_elem; ++i) {
      prob_greater.emplace_back(((double)(reuse_time_sum - prefix_sum[i])) /
                                (double)reuse_time_sum);
    }

    // integrate CCDF and calculate MRC
    uint64_t cache_size = 0;
    double integral = 0;
    double prev_integ = 0;
    const size_t num_mrc_elem = max_cache_size / bucket_size_ + 1;
    std::vector<double> result;
    result.reserve(num_mrc_elem + 1);

    size_t t = 0;
    uint64_t c = 0;
    while (t < prob_greater.size() - 1) {
      while (integral > c && c < num_mrc_elem) {
        const double mr = prob_greater[t - 1] + (prob_greater[t] - prob_greater[t - 1]) * ((double)c - prev_integ) / (integral - prev_integ);
        result.emplace_back(mr);
        ++c;
      }
      if (c >= num_mrc_elem) {
        break;
      }
      prev_integ = integral;
      const int32_t k1 = reuse_time_hist[t].first;
      const int32_t k2 = reuse_time_hist[t + 1].first;
      integral += (prob_greater[t] + prob_greater[t + 1]) * (k2 - k1) / 2.0;
      t += 1;
    }

    while (result.size() > 2) {
      const size_t s = result.size() - 1;
      if (result[s] == result[s - 1]) {
        result.pop_back();
      } else {
        break;
      }
    }
    result[0] = 1.0;

    result.emplace_back(timestamp);
    
    run__.fetch_sub(1, std::memory_order_release);
    return result;
  }

  size_t GetCacheSize() const override {
    return tunable_cache_->GetCacheSize();
  }

  void SetCacheSize(size_t new_size) override {
    tunable_cache_->SetCacheSize(new_size);
  }

  size_t GetCacheEntrySize() const override {
    return tunable_cache_->GetCacheEntrySize();
  }

  double GetHitRate() const override { return tunable_cache_->GetHitRate(); }

  void ResetStat() override { tunable_cache_->ResetStat(); }

  void StopSamplingAndReleaseResource() {
    // avoid new thread to enter profiling
    run_lock_.store(true, std::memory_order_acquire);
    // wait until no thread profiling
    while (run_.load(std::memory_order_acquire) != 0);
    timestamp_.store(0, std::memory_order_relaxed);
    for (auto iter = reuse_time_hist_->cbegin();
         iter != reuse_time_hist_->cend(); ++iter) {
      delete iter->second;
    }
    reuse_time_hist_ = nullptr;
    for (auto iter = last_access_map_->cbegin();
         iter != last_access_map_->cend(); ++iter) {
      delete iter->second;
    }
    last_access_map_ = nullptr;
    malloc_trim(0);
    sample_time_ = timestamp_.load(std::memory_order_relaxed);
  }

  void StartSampling() {
    if (!run_lock_.load(std::memory_order_acquire)) {
      // already started
      return;
    }
    ResetHistogram();
    ResetLastAccessMap();
    run_lock_.store(false, std::memory_order_release);
  }

  virtual ~SamplingLRUAETProfiler() {
    for (auto iter = last_access_map_->cbegin();
         iter != last_access_map_->cend(); ++iter) {
      delete iter->second;
    }
    last_access_map_ = nullptr;

    for (auto iter = reuse_time_hist_->cbegin();
         iter != reuse_time_hist_->cend(); ++iter) {
      delete iter->second;
    }
    reuse_time_hist_ = nullptr;
  }

 protected:
  void IncreaseHistogram(uint64_t time) {
    if (time == 0) {
      AtomicBucketIncrease(0);
      return;
    }
    size_t bucket = (time - 1) / bucket_size_ + 1;
    AtomicBucketIncrease(bucket);
  }

  void AtomicBucketIncrease(int32_t bucket) {
    if (!reuse_time_hist_) {
      return;
    }
    auto found = reuse_time_hist_->find_wait_free(bucket);
    if (found.first == HIST_EMPTY_KEY || found.first == HIST_DELETED_KEY) {
      uint32_t * value_ptr = new uint32_t(1);
      auto inserted = reuse_time_hist_->insert_lockless({bucket, value_ptr});
      if (inserted.first->second != value_ptr) {
        delete value_ptr;
        __sync_add_and_fetch(inserted.first->second, 1);
      }
    } else {
      __sync_add_and_fetch(found.second, 1);
    }
  }

  void DoReferenceKey(const K& key) {
    static thread_local std::mt19937_64 rand;
    
    int64_t reuse_dist = 0;
    
    uint64_t timestamp = timestamp_.fetch_add(1, std::memory_order_relaxed) + 1;
    auto iter = last_access_map_->find_wait_free(const_cast<K&>(key));
    // not found and we need to sample
    if (iter.first == EMPTY_KEY || iter.first == DELETED_KEY || *(iter.second) == 0) {
      if (distrib_real_(rand) <= samping_rate_) {
        if (iter.first == EMPTY_KEY || iter.first == DELETED_KEY || sampling_interval_ == 1) {
          // new sample
          uint64_t* value_ptr = new uint64_t(timestamp);
          auto inserted = last_access_map_->insert_lockless({key, value_ptr});
          if (inserted.first->second != value_ptr) {
            delete value_ptr;
            return;
          }
        } else {
          // existing key
          uint64_t* value_ptr = iter.second;
          __sync_bool_compare_and_swap(value_ptr, 0, timestamp);
        }
        reuse_dist = 0;
      } else return;
    } else {
      uint64_t old_ts = *(iter.second);
      reuse_dist = timestamp - old_ts;
      if (sampling_interval_ == 1) {
        __sync_val_compare_and_swap(iter.second, old_ts, timestamp);
      } else {
        __sync_val_compare_and_swap(iter.second, old_ts, 0);
      } 
    }
    if (reuse_dist > 0 || (reuse_dist == 0 && sampling_interval_ == 1))
      IncreaseHistogram(reuse_dist);
  }

  void ResetLastAccessMap() {
    uint64_t count = 0;
    if (last_access_map_) {
      for (auto iter = last_access_map_->cbegin();
           iter != last_access_map_->cend(); ++iter) {
        delete iter->second;
	      ++count;
      }
      // Print Last Access Map Info
      VLOG(2) << "map info size:" << count
                << ", bucket_count:" << last_access_map_->bucket_count()
                << ", load_factor:" << last_access_map_->load_factor()
                << ", max_load_factor:" << last_access_map_->max_load_factor()
                << ", min_load_factor:" << last_access_map_->min_load_factor();

      VLOG(2) << "Resetting Access Map: " << count;
    }
    last_access_map_.reset(new google::dense_hash_map_lockless<K, uint64_t*>());
    last_access_map_->max_load_factor(1.5f);
    last_access_map_->min_load_factor(0.5f);
    last_access_map_->set_empty_key_and_value(EMPTY_KEY, 0);
    last_access_map_->set_counternum(16);
    last_access_map_->set_deleted_key(DELETED_KEY);
  }

  void ResetHistogram() {
    uint64_t count = 0;
    if (reuse_time_hist_) {
      for (auto iter = reuse_time_hist_->cbegin();
           iter != reuse_time_hist_->cend(); ++iter) {
        delete iter->second;
	      ++count;
      }
      // Print Last Histogram
      VLOG(2) << "map info size:" << count
                << ", bucket_count:" << reuse_time_hist_->bucket_count()
                << ", load_factor:" << reuse_time_hist_->load_factor()
                << ", max_load_factor:" << reuse_time_hist_->max_load_factor()
                << ", min_load_factor:" << reuse_time_hist_->min_load_factor();

      VLOG(2) << "Resetting Histogram: " << count;
    }
    reuse_time_hist_.reset(new HistogramMap());
    reuse_time_hist_->max_load_factor(1.5f);
    reuse_time_hist_->min_load_factor(0.5f);
    reuse_time_hist_->set_empty_key_and_value(HIST_EMPTY_KEY, 0);
    reuse_time_hist_->set_counternum(16);
    reuse_time_hist_->set_deleted_key(HIST_DELETED_KEY);
  }

  std::pair<uint64, uint64> GetMoveCount() const {
    return tunable_cache_->GetMoveCount();
  }

  void ResetMoveCount() {
    tunable_cache_->ResetMoveCount();
  }

 private:

  std::string name_;
  size_t bucket_size_;
  std::unique_ptr<HistogramMap> reuse_time_hist_;
  std::unique_ptr<google::dense_hash_map_lockless<K, uint64_t*>>
      last_access_map_;
  std::atomic<uint64_t> timestamp_;
  volatile uint64_t sample_time_{0};
  std::atomic<bool> sample_lock_{false};
  std::atomic<bool> run_lock_;
  std::atomic<uint> run_;
  uint64_t sampling_interval_;
  double samping_rate_;
  std::uniform_int_distribution<uint64_t> distrib_;
  std::uniform_real_distribution<double> distrib_real_;
  std::mt19937 rand_;
  TunableCache* tunable_cache_;
  mutex mu_;
  static const K EMPTY_KEY = -1;
  static const int32_t HIST_EMPTY_KEY = -3;
  static const int32_t HIST_DELETED_KEY = -4;
  static const K DELETED_KEY = -2;
};

}  // namespace embedding
}  // namespace tensorflow

#endif  // DEEPREC_CACHE_PROFILER_H
