#ifndef DEEPREC_CACHE_PROFILER_H
#define DEEPREC_CACHE_PROFILER_H

#include <utility>
#include <vector>
#include <string>
#include "sparsehash/dense_hash_map_lockless"
#include <random>
#include "tensorflow/core/framework/embedding/cache.h"

namespace tensorflow {
namespace embedding {

class TunableCache {
public:
    virtual size_t GetCacheSize() const = 0;

    virtual void SetCacheSize(size_t new_size) = 0;

    virtual size_t GetCacheEntrySize() const = 0;

    virtual double GetHitRate() const = 0;

    virtual void ResetStat() = 0;
};

template<typename K>
class CacheMRCProfiler : public virtual TunableCache {
public:
    virtual void ReferenceKey(const K &key) = 0;

    virtual void ReferenceKeyBatch(const K *keys, size_t batch_size) = 0;

    virtual std::vector<double> GetMRC(uint64_t max_cache_size) const = 0;

    virtual const std::string &GetName() const = 0;

    virtual void ResetProfiling() = 0;

    virtual size_t GetBucketSize() const = 0;
};

template<typename K>
class SamplingLRUAETProfiler : public virtual CacheMRCProfiler<K> {
public:
    explicit SamplingLRUAETProfiler(std::string name,
                                    const size_t bucket_size,
                                    const size_t max_reuse_time,
                                    const uint64_t sampling_interval,
                                    TunableCache *tunable_cache = nullptr)
            : CacheMRCProfiler<K>(),
              name_(std::move(name)),
              bucket_size_(bucket_size),
              max_reuse_time_(max_reuse_time),
              reuse_time_hist_(max_reuse_time / bucket_size + 3, 0),
              timestamp_(0),
              sampling_interval_(sampling_interval),
              rand_(std::random_device()()),
              distrib_(1, sampling_interval * 2 - 1),
              run_lock_(false),
              run_(0),
              tunable_cache_(tunable_cache) {
      ResetLastAccessMap();
    }

    void ReferenceKey(const K &key) override {
      // if resetting, just skip
      while (run_lock_.load(std::memory_order_acquire));
      // indicate we are running
      run_.fetch_add(1, std::memory_order_acquire);
      DoReferenceKey(key);
      run_.fetch_sub(1, std::memory_order_release);
    }

    void ReferenceKeyBatch(const K *keys, const size_t batch_size) override {
      while (run_lock_.load(std::memory_order_acquire));
      // indicate we are running
      run_.fetch_add(1, std::memory_order_acquire);
      for (size_t i = 0; i < batch_size; ++i) {
        DoReferenceKey(keys[i]);
      }
      run_.fetch_sub(1, std::memory_order_release);
    }

    const std::string &GetName() const override {
      return name_;
    }

    void ResetProfiling() override {
      // avoid new thread to enter profiling
      run_lock_.store(true, std::memory_order_acquire);
      // wait until no thread profiling
      while (run_.load(std::memory_order_acquire) != 0);
      timestamp_.store(0, std::memory_order_relaxed);
      reuse_time_hist_.clear();
      reuse_time_hist_.resize(max_reuse_time_ / bucket_size_ + 3);
      ResetLastAccessMap();
      sample_time_ = timestamp_.load(std::memory_order_relaxed);
      run_lock_.store(false, std::memory_order_release);
    }

    size_t GetBucketSize() const override {
      return bucket_size_;
    }

    std::vector<double> GetMRC(uint64_t max_cache_size) const override {
      // TODO: potential race condition when resetting
      const size_t num_elem = reuse_time_hist_.size();
      std::vector<uint64_t> reuse_time_hist(reuse_time_hist_.cbegin(), reuse_time_hist_.cend());
      const uint64_t timestamp = timestamp_.load(std::memory_order_relaxed);
      uint64_t reuse_time_sum = 0;
      if (sampling_interval_ != 1) {
        reuse_time_sum += last_access_map_->size_lockless();
      } else {
        reuse_time_sum += reuse_time_hist[0];
      }
      size_t last_index = 0;
      std::vector<uint64_t> prefix_sum;
      prefix_sum.reserve(num_elem);
      prefix_sum.emplace_back(0);
      for (size_t i = 1; i < num_elem; ++i) {
        prefix_sum.emplace_back(prefix_sum[last_index] + reuse_time_hist[i]);
        reuse_time_sum += reuse_time_hist[i];
        last_index = i;
      }
      // calculate CCDF
      std::vector<double> prob_greater;
      prob_greater.reserve(num_elem - 1);
      prob_greater.emplace_back(1.0);
      for (size_t i = 1; i < num_elem - 1; ++i) {
        prob_greater.emplace_back(((double) (reuse_time_sum - prefix_sum[i])) / (double) reuse_time_sum);
      }
      // prefix_sum.clear();
      // integrate CCDF and calculate MRC
      uint64_t cache_size = 0;
      double integral = 0;
      double prev_integ = 0;
      const size_t num_mrc_elem = max_cache_size / bucket_size_ + 1;
      std::vector<double> result;
      result.reserve(num_mrc_elem + 1);
      for (uint64_t t = 0; t < num_elem - 1; ++t) {
        if (integral >= cache_size) {
          // linear interpolation
          double mr = prob_greater[t - 1] +
                      (prob_greater[t] - prob_greater[t - 1]) * ((cache_size - prev_integ) / (integral - prev_integ));
          result.emplace_back(mr);
          cache_size += bucket_size_;
          if (cache_size > max_cache_size) break;
        }
        prev_integ = integral;
        const double increment =
                (prob_greater[t] + prob_greater[std::min(t + 1, prob_greater.size() - 1)]) / 2 * bucket_size_;
        if (increment == 0.0) {
          break;
        }
        integral += increment;
        // integral += prob_greater[t];
      }
      result.emplace_back(timestamp);
      result[0] = 1.0;
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

    double GetHitRate() const override {
      return tunable_cache_->GetHitRate();
    }

    void ResetStat() override {
      tunable_cache_->ResetStat();
    }

    virtual ~SamplingLRUAETProfiler() {
      for (auto iter = last_access_map_->cbegin(); iter != last_access_map_->cend(); ++iter) {
        delete iter->second;
      }
      last_access_map_.reset(nullptr);
    }

protected:
    void IncreaseHistogram(uint64_t time) {
      if (time > max_reuse_time_) {
        __sync_add_and_fetch(&reuse_time_hist_[reuse_time_hist_.size() - 1], 1);
        return;
      }
      if (time == 0) {
        __sync_add_and_fetch(&reuse_time_hist_[0], 1);
        return;
      }
      size_t bucket = (time - 1) / bucket_size_ + 1;
      __sync_add_and_fetch(&reuse_time_hist_[bucket], 1);
    }

    void DoReferenceKey(const K &key) {
      uint64_t timestamp = timestamp_.fetch_add(1, std::memory_order_relaxed) + 1;
      auto iter = last_access_map_->find_wait_free(const_cast<K &>(key));
      // not found and we need to sample
      if (iter.first == EMPTY_KEY || iter.first == DELETED_KEY) {
        if (timestamp >= sample_time_) {
          bool unlocked = false;
          // avoid concurrent update of next sample time
          if (sample_lock_.compare_exchange_strong(unlocked, true, std::memory_order_acquire)) {
            uint64_t next = distrib_(rand_);
            sample_time_ = timestamp_.load(std::memory_order_relaxed) + next;
            sample_lock_.store(false, std::memory_order_release);
          }
          if (sampling_interval_ == 1) {
            IncreaseHistogram(0);
          }
          last_access_map_->insert_lockless({key, new uint64_t(timestamp)});
        }
      } else {
        uint64_t reuse_time = timestamp - *(iter.second);
        IncreaseHistogram(reuse_time);

        *(iter.second) = timestamp;
      }
    }

    void ResetLastAccessMap() {
      if (last_access_map_) {
        for (auto iter = last_access_map_->cbegin(); iter != last_access_map_->cend(); ++iter) {
          delete iter->second;
        }
      }
      last_access_map_.reset(new google::dense_hash_map_lockless<K, uint64_t *>());
      last_access_map_->max_load_factor(0.8f);
      last_access_map_->set_empty_key_and_value(EMPTY_KEY, 0);
      last_access_map_->set_counternum(16);
      last_access_map_->set_deleted_key(DELETED_KEY);
    }

private:
    std::string name_;
    size_t bucket_size_;
    size_t max_reuse_time_;
    std::vector<uint64_t> reuse_time_hist_;
    std::unique_ptr<google::dense_hash_map_lockless<K, uint64_t *>> last_access_map_;
    std::atomic<uint64_t> timestamp_;
    volatile uint64_t sample_time_{0};
    std::atomic<bool> sample_lock_{false};
    std::atomic<bool> run_lock_;
    std::atomic<uint> run_;
    uint64_t sampling_interval_;
    std::uniform_int_distribution<uint64_t> distrib_;
    std::mt19937 rand_;
    TunableCache *tunable_cache_;
    inline static constexpr K EMPTY_KEY = -1;
    inline static constexpr K DELETED_KEY = -2;
};

} // namespace embedding
} // namespace tensorflow

#endif //DEEPREC_CACHE_PROFILER_H
