#ifndef DEEPREC_CACHE_TUNING_STRATEGY_H
#define DEEPREC_CACHE_TUNING_STRATEGY_H

#include <memory>
#include <map>
#include <random>

#include "tensorflow/core/framework/embedding/cache_profiler.h"

namespace tensorflow {
namespace embedding {

class CacheItem {
public:
  size_t bucket_size;
  size_t orig_size;
  size_t new_size;
  size_t entry_size;
  uint64_t vc;
  uint64_t mc;
  double mr;
  std::vector<double> mrc;

  CacheItem(size_t bucketSize, size_t origSize, size_t newSize, size_t entrySize, uint64_t vc, uint64_t mc, double mr,
            const std::vector<double> &mrc);

  CacheItem();
};

template<typename K>
class CacheTuningStrategy {
public:
  virtual bool
  DoTune(size_t total_size, std::map<CacheMRCProfiler<K> *, CacheItem> &caches, size_t unit, size_t min_size) = 0;
};

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

static void RandomApportion(std::vector<size_t> &parts, size_t total, size_t min_size) {
  const size_t resv_size = parts.size() * min_size;
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
    part += min_size;
  }
}


template<typename K>
class MinimalizeMissCountRandomGreedyTuningStrategy : public CacheTuningStrategy<K> {
public:
  bool
  DoTune(size_t total_size, std::map<CacheMRCProfiler<K> *, CacheItem> &caches, size_t unit, size_t min_size) override {
    uint64_t orig_mc_sum = 0;

    for (auto &kv: caches) {
      CacheMRCProfiler<K> *cache = kv.first;
      CacheItem &item = kv.second;
      orig_mc_sum += item.mc;
    }

    // do random apportion and compute new MR
    {
      std::vector<size_t> parts(caches.size());
      RandomApportion(parts, total_size, min_size);
      size_t i = 0;
      for (auto &item: caches) {
        size_t new_size = parts[i++];
        size_t new_entries = new_size / item.second.entry_size;
        item.second.new_size = new_size;
        item.second.mr = InterpolateMRC(item.second.mrc, item.second.bucket_size, new_entries);
        item.second.mc = item.second.mr * item.second.vc;
      }
    }

    while (true) {
      uint64_t max_gain = 0.0, min_loss = 0.0, gain_new_mc = 0.0, loss_new_mc = 0.0;
      CacheMRCProfiler<K> *max_gain_cache = nullptr, *min_loss_cache = nullptr;
      for (auto &item: caches) {
        const size_t current_size = item.second.new_size;
        const size_t new_entries = (item.second.new_size + unit) / item.second.entry_size;
        const double new_mr = InterpolateMRC(item.second.mrc, item.second.bucket_size, new_entries);
        const uint64_t new_mc = new_mr * item.second.vc;
        const uint64_t gain = item.second.mc - new_mc;
        if (gain > max_gain || max_gain_cache == nullptr) {
          max_gain = gain;
          max_gain_cache = item.first;
          gain_new_mc = new_mc;
        }
      }

      for (auto &item: caches) {
        if (item.first == max_gain_cache) continue;
        const size_t current_size = item.second.new_size;
        if (current_size <= min_size + unit) {
          continue;
        }
        const ssize_t new_entries = (item.second.new_size - unit) / item.second.entry_size;

        const double new_mr = InterpolateMRC(item.second.mrc, item.second.bucket_size, new_entries);
        const uint64_t new_mc = new_mr * item.second.vc;
        const uint64_t loss = new_mc - item.second.mc;
        if (loss < min_loss || min_loss_cache == nullptr) {
          min_loss = loss;
          min_loss_cache = item.first;
          loss_new_mc = new_mc;
        }
      }

      if (max_gain <= min_loss || max_gain_cache == nullptr || min_loss_cache == nullptr) break;

      caches[max_gain_cache].new_size += unit;
      caches[max_gain_cache].mc = gain_new_mc;
      caches[min_loss_cache].new_size -= unit;
      caches[min_loss_cache].mc = loss_new_mc;
    }

    uint64_t new_mc_sum = 0;
    for (auto &item: caches) {
      new_mc_sum += item.second.mc;
    }
    LOG(INFO) << "orig MCs=" << orig_mc_sum << ", new MCs=" << new_mc_sum << ", diff="
              << (int64_t) (orig_mc_sum - new_mc_sum);
    if (new_mc_sum >= orig_mc_sum) {
      LOG(INFO) << "new MCs not less than original MCs, not tuning cache";
      return false;
    }

    return true;
  }
};

template<typename K>
class MinimalizeMissCountLocalGreedyTuningStrategy : public CacheTuningStrategy<K> {
public:
  bool
  DoTune(size_t total_size, std::map<CacheMRCProfiler<K> *, CacheItem> &caches, size_t unit, size_t min_size) override {
    uint64_t orig_mc_sum = 0;

    for (auto &kv: caches) {
      CacheMRCProfiler<K> *cache = kv.first;
      CacheItem &item = kv.second;
      orig_mc_sum += item.mc;
    }

    for (auto &item: caches) {
      size_t entries = item.second.orig_size / item.second.entry_size;
      item.second.new_size = item.second.orig_size;
      item.second.mr = InterpolateMRC(item.second.mrc, item.second.bucket_size, entries);
      item.second.mc = item.second.mr * item.second.vc;
    }

    while (true) {
      uint64_t max_gain = 0.0, min_loss = 0.0, gain_new_mc = 0.0, loss_new_mc = 0.0;
      CacheMRCProfiler<K> *max_gain_cache = nullptr, *min_loss_cache = nullptr;
      for (auto &item: caches) {
        const size_t current_size = item.second.new_size;
        const size_t new_entries = (item.second.new_size + unit) / item.second.entry_size;
        const double new_mr = InterpolateMRC(item.second.mrc, item.second.bucket_size, new_entries);
        const uint64_t new_mc = new_mr * item.second.vc;
        const uint64_t gain = item.second.mc - new_mc;
        if (gain > max_gain || max_gain_cache == nullptr) {
          max_gain = gain;
          max_gain_cache = item.first;
          gain_new_mc = new_mc;
        }
      }

      for (auto &item: caches) {
        if (item.first == max_gain_cache) continue;
        const size_t current_size = item.second.new_size;
        if (current_size <= min_size + unit) {
          continue;
        }
        const ssize_t new_entries = (item.second.new_size - unit) / item.second.entry_size;

        const double new_mr = InterpolateMRC(item.second.mrc, item.second.bucket_size, new_entries);
        const uint64_t new_mc = new_mr * item.second.vc;
        const uint64_t loss = new_mc - item.second.mc;
        if (loss < min_loss || min_loss_cache == nullptr) {
          min_loss = loss;
          min_loss_cache = item.first;
          loss_new_mc = new_mc;
        }
      }

      if (max_gain <= min_loss || max_gain_cache == nullptr || min_loss_cache == nullptr) break;

      caches[max_gain_cache].new_size += unit;
      caches[max_gain_cache].mc = gain_new_mc;
      caches[min_loss_cache].new_size -= unit;
      caches[min_loss_cache].mc = loss_new_mc;
    }

    uint64_t new_mc_sum = 0;
    for (auto &item: caches) {
      new_mc_sum += item.second.mc;
    }
    LOG(INFO) << "orig MCs=" << orig_mc_sum << ", new MCs=" << new_mc_sum << ", diff="
              << (int64_t) (orig_mc_sum - new_mc_sum);
    if (new_mc_sum >= orig_mc_sum) {
      LOG(INFO) << "new MCs not less than original MCs, not tuning cache";
      return false;
    }

    return true;
  }
};

template<typename K>
class MinimalizeMissCountDynamicProgrammingTuningStrategy: public CacheTuningStrategy<K> {
public:
  bool
  DoTune(size_t total_size, std::map<CacheMRCProfiler<K> *, CacheItem> &caches, size_t unit, size_t min_size) override {
    const size_t total_units = total_size / unit;
    const size_t min_unit = std::ceil((double)min_size / unit);
    size_t orig_mc_sum = 0;
    size_t low_sum = 0, high_sum = 0;
    std::vector<CacheItem*> items;
    for (auto &cache: caches) {
      items.emplace_back(&cache.second);
      orig_mc_sum += cache.second.mc;
    }

    const size_t num_caches = items.size();
    std::vector<std::vector<size_t>> miss(num_caches);
    std::vector<size_t> offset(num_caches);
    std::vector<std::vector<size_t>> target(num_caches);
    size_t last_low_sum;
    size_t last_max;
    for (size_t i = 0; i < num_caches; ++i) {
      CacheItem& item = *items[i];
      size_t max_j = 0;
      const size_t entry_size = item.entry_size;
      const size_t bucket_size = item.bucket_size;
      const size_t vc = item.vc;
      last_low_sum = low_sum;
      low_sum += min_unit;
      const std::vector<double>& mrc = items[i]->mrc;
      const size_t hi = total_units;
      high_sum += hi;
      std::vector<size_t>& miss_i = miss[i], &target_i = target[i];
      const size_t j_upper = std::min(total_units, high_sum);
      miss_i.resize(j_upper -  low_sum + 1);
      target_i.resize(j_upper -  low_sum + 1);
      if (i == 0) {
        for (size_t j = low_sum; j <= j_upper; ++j) {
          miss_i[j - low_sum] = std::round(InterpolateMRC(mrc, bucket_size, j * unit / entry_size) * vc);
        }
        max_j = j_upper;
        offset[i] = low_sum;
      } else {
        const std::vector<size_t>& miss_l = miss[i - 1];
        for (size_t j  = low_sum; j <= j_upper; ++j) {
          miss_i[j - low_sum] = UINT64_MAX;
          const size_t k_upper = std::min(j - last_low_sum, hi);
          for (size_t k = min_unit; k <= k_upper; ++k) {
            const size_t c_miss = std::round(InterpolateMRC(mrc, bucket_size, k * unit / entry_size) * vc);
            const size_t p_miss = miss_l[std::min(last_max, j - k) - offset[i - 1]];
            const size_t n_miss = c_miss + p_miss;
            if (miss_i[j - low_sum] > n_miss) {
              miss_i[j - low_sum] = n_miss;
              target_i[j - low_sum] = k;
              if (j > max_j) {
                max_j = j;
              }
            }
          }
        }
        offset[i] = low_sum;
      }

      last_max = max_j;
    }

    size_t t = total_units;
    const size_t new_mc_sum = miss[num_caches - 1][t - offset[num_caches - 1]];
    if (new_mc_sum >= orig_mc_sum) {
      LOG(INFO) << "Dynamic Programming: new MC sum not less than orig mc_sum";
      return false;
    }
    for (ssize_t i = (ssize_t)num_caches - 1; i >= 0; --i) {
      if (i > 0) {
        const size_t alloc = target[i][t - offset[i]];
        items[i]->new_size = alloc * unit;
        t -= alloc;
      } else {
        items[i]->new_size = t * unit;
      }
    }
    return true;
  }
};

template<typename K>
class MinimalizeMissRateDynamicProgrammingTuningStrategy: public CacheTuningStrategy<K> {
public:
  bool
  DoTune(size_t total_size, std::map<CacheMRCProfiler<K> *, CacheItem> &caches, size_t unit, size_t min_size) override {
    const size_t total_units = total_size / unit;
    const size_t min_unit = std::ceil((double)min_size / unit);
    double orig_mr_sum = 0;
    size_t low_sum = 0, high_sum = 0;
    std::vector<CacheItem*> items;
    for (auto &cache: caches) {
      items.emplace_back(&cache.second);
      orig_mr_sum += cache.second.mr;
    }

    const size_t num_caches = items.size();
    std::vector<std::vector<double>> miss(num_caches);
    std::vector<size_t> offset(num_caches);
    std::vector<std::vector<size_t>> target(num_caches);
    size_t last_low_sum;
    size_t last_max;
    for (size_t i = 0; i < num_caches; ++i) {
      CacheItem& item = *items[i];
      size_t max_j = 0;
      const size_t entry_size = item.entry_size;
      const size_t bucket_size = item.bucket_size;
      last_low_sum = low_sum;
      low_sum += min_unit;
      const std::vector<double>& mrc = items[i]->mrc;
      const size_t hi = total_units;
      high_sum += hi;
      std::vector<double>& miss_i = miss[i];
      std::vector<size_t>& target_i = target[i];
      const size_t j_upper = std::min(total_units, high_sum);
      miss_i.resize(j_upper -  low_sum + 1);
      target_i.resize(j_upper -  low_sum + 1);
      if (i == 0) {
        for (size_t j = low_sum; j <= j_upper; ++j) {
          miss_i[j - low_sum] = InterpolateMRC(mrc, bucket_size, j * unit / entry_size);
        }
        max_j = j_upper;
        offset[i] = low_sum;
      } else {
        const std::vector<double>& miss_l = miss[i - 1];
        for (size_t j  = low_sum; j <= j_upper; ++j) {
          miss_i[j - low_sum] = UINT64_MAX;
          const size_t k_upper = std::min(j - last_low_sum, hi);
          for (size_t k = min_unit; k <= k_upper; ++k) {
            const double c_miss = InterpolateMRC(mrc, bucket_size, k * unit / entry_size);
            const double p_miss = miss_l[std::min(last_max, j - k) - offset[i - 1]];
            const double n_miss = c_miss + p_miss;
            if (miss_i[j - low_sum] > n_miss) {
              miss_i[j - low_sum] = n_miss;
              target_i[j - low_sum] = k;
              if (j > max_j) {
                max_j = j;
              }
            }
          }
        }
        offset[i] = low_sum;
      }

      last_max = max_j;
    }

    size_t t = total_units;
    const double new_mr_sum = miss[num_caches - 1][t - offset[num_caches - 1]];
    if (new_mr_sum >= orig_mr_sum) {
      LOG(INFO) << "Dynamic Programming: new MR sum not less than orig MR sum";
      return false;
    }
    for (ssize_t i = (ssize_t)num_caches - 1; i >= 0; --i) {
      if (i > 0) {
        const size_t alloc = target[i][t - offset[i]];
        items[i]->new_size = alloc * unit;
        t -= alloc;
      } else {
        items[i]->new_size = t * unit;
      }
    }
    return true;
  }
};

template<typename K>
class CacheTuningStrategyCreator {
public:
  static CacheTuningStrategy<K> *Create(const std::string &type) {
    if (type == "min_mc_random_greedy") {
      LOG(INFO) << "Use MinimalizeMissCountRandomGreedyTuningStrategy";
      return new MinimalizeMissCountRandomGreedyTuningStrategy<K>();
    } else if (type == "min_mc_local_greedy") {
      LOG(INFO) << "Use MinimalizeMissCountLocalGreedyTuningStrategy";
      return new MinimalizeMissCountLocalGreedyTuningStrategy<K>();
    } else if (type == "min_mc_dp") {
      LOG(INFO) << "Use MinimalizeMissCountDynamicProgrammingTuningStrategy";
      return new MinimalizeMissCountDynamicProgrammingTuningStrategy<K>();
    } else if (type == "min_mr_dp") {
      LOG(INFO) << "Use MinimalizeMissRateDynamicProgrammingTuningStrategy";
      return new MinimalizeMissRateDynamicProgrammingTuningStrategy<K>();
    } else {
      LOG(INFO) << "CacheTuningStrategyCreator: \"" << type
                << "\" not valid, using default \"min_mc_random_greedy\" strategy";
      return new MinimalizeMissCountRandomGreedyTuningStrategy<K>();
    }
  }
};

} // namespace embedding
} // namespace tensorflow


#endif //DEEPREC_CACHE_TUNING_STRATEGY_H
