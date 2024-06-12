/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
======================================================================*/
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_FACTORY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_FACTORY_H_

#include <string>
#include "cache.h"
#include "tensorflow/core/framework/embedding/cache_manager.h"
#include "tensorflow/core/framework/embedding/cache_profiler.h"
#include "tensorflow/core/framework/embedding/profiled_cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace embedding {

template<typename K, typename C>
class Creator {
  public:
    [[noreturn]] static C Create(const std::string& name, int64 capacity, int num_threads, int way) {
      LOG(FATAL) << "Not implemented: " << typeid(Create).name();
    }
};

template<typename K>
class Creator<K, LRUCache<K>> {
  public:
    static LRUCache<K> Create(const std::string& name, int64 capacity, int num_threads, int way) {
      return { name };
    }
};

template<typename K>
class Creator<K, LFUCache<K>> {
  public:
    static LFUCache<K> Create(const std::string& name, int64 capacity, int num_threads, int way) {
      return { name };
    }
};

template<typename K>
class Creator<K, BlockLockLFUCache<K>> {
  public:
    static BlockLockLFUCache<K> Create(const std::string& name, int64 capacity, int num_threads, int way) {
      return { capacity, way, num_threads };
    }
};
class CacheFactory {
 public:
    template<typename K>
    static BatchCache<K> *
    Create(CacheStrategy cache_strategy, ProfilingStrategy profiling_strategy, std::string name, int64 capacity, int num_threads, TunableCache *tunable_cache = nullptr) {
      switch (cache_strategy) {
        case CacheStrategy::LRU:
          LOG(INFO) << " Use Storage::LRU in multi-tier EmbeddingVariable "
                    << name;
          return CreateCache<K, LRUCache<K>>(profiling_strategy, name, capacity, num_threads, 0, tunable_cache);
        case CacheStrategy::LFU:
          LOG(INFO) << " Use Storage::LFU in multi-tier EmbeddingVariable "
                    << name;
          return CreateCache<K, LFUCache<K>>(profiling_strategy, name, capacity, num_threads, 0, tunable_cache);
        case CacheStrategy::B64LFU:
          return CreateCache<K, BlockLockLFUCache<K>>(profiling_strategy, name, capacity, num_threads, 64, tunable_cache);
        case CacheStrategy::B8LFU:
          return CreateCache<K, BlockLockLFUCache<K>>(profiling_strategy, name, capacity, num_threads, 8, tunable_cache);
        default:
          LOG(FATAL) << " Invalid Cache strategy";
          return nullptr;
      }
  }

  template<typename K, typename Base>
  static Base*
  CreateCache(const ProfilingStrategy profiling_strategy, const std::string& name, int64 capacity, int num_threads, int way, TunableCache *tunable_cache = nullptr) {
    switch (profiling_strategy) {
      case ProfilingStrategy::NONE:
        return new Base(std::move(Creator<K, Base>::Create(name, capacity, num_threads, way)));
      case ProfilingStrategy::AET: {
        size_t bucket_size;
        size_t max_reuse_dist;
        uint64_t sampling_interval;
        ReadInt64FromEnvVar("CACHE_PROFILER_BUCKET_SIZE", 10, reinterpret_cast<int64 *>(&bucket_size));
        ReadInt64FromEnvVar("CACHE_PROFILER_MAX_REUSE_DIST", 100000, reinterpret_cast<int64 *>(&max_reuse_dist));
        ReadInt64FromEnvVar("CACHE_PROFILER_SAMPLING_INTERVAL", 1, reinterpret_cast<int64 *>(&sampling_interval));
        ProfiledCacheProxy<K, Base> *proxy_cache = new ProfiledCacheProxy<K, Base>(name, bucket_size, max_reuse_dist, sampling_interval, std::move(Creator<K, Base>::Create(name, capacity, num_threads, way)), tunable_cache);
        if (tunable_cache != nullptr) {
          CacheManager::GetInstance().RegisterCache(*proxy_cache->GetProfiler());
        }
        return proxy_cache;
      }
    }
  }

  
};



} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_FACTORY_H_
