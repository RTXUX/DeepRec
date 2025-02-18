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

#include "cache.h"
#include "tensorflow/core/framework/embedding/profiled_cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace embedding {
class CacheFactory {
 public:
    template<typename K>
    static BatchCache<K> *
    Create(CacheStrategy cache_strategy, std::string name, TunableCache *tunable_cache = nullptr) {
      int64 shard_shift;
      switch (cache_strategy) {
        case CacheStrategy::LRU:
          LOG(INFO) << " Use Storage::LRU in multi-tier EmbeddingVariable "
                    << name;
          return new LRUCache<K>(name);
        case CacheStrategy::LFU:
          LOG(INFO) << " Use Storage::LFU in multi-tier EmbeddingVariable "
                    << name;
          return new LFUCache<K>(name);
        case CacheStrategy::ProfiledLRU:
          LOG(INFO) << " Use Storage::ProfiledLRU in multi-tier EmbeddingVariable "
                    << name;
          size_t bucket_size;
          size_t max_reuse_dist;
          uint64_t sampling_interval;
          ReadInt64FromEnvVar("CACHE_PROFILER_BUCKET_SIZE", 10, reinterpret_cast<int64 *>(&bucket_size));
          ReadInt64FromEnvVar("CACHE_PROFILER_MAX_REUSE_DIST", 100000, reinterpret_cast<int64 *>(&max_reuse_dist));
          ReadInt64FromEnvVar("CACHE_PROFILER_SAMPLING_INTERVAL", 1, reinterpret_cast<int64 *>(&sampling_interval));
          ProfiledLRUCache<K> *cache;
          cache = new ProfiledLRUCache<K>(name, bucket_size, max_reuse_dist, sampling_interval, tunable_cache);
          if (tunable_cache != nullptr) {
            CacheManager::GetInstance().RegisterCache(*cache->GetProfiler());
          }
          return cache;
        case CacheStrategy::ShardedLRU:
          LOG(INFO) << " Use Storage::ShardedLRU in multi-tier EmbeddingVariable " << name;
          ReadInt64FromEnvVar("CACHE_SHARD_SHIFT", 0, &shard_shift);
          return new ShardedLRUCache<K>(name, shard_shift);
        case CacheStrategy::ProfiledShardedLRU:
          LOG(INFO) << " Use Storage::ProfiledShardedLRU in multi-tier EmbeddingVariable " << name;
          ReadInt64FromEnvVar("CACHE_SHARD_SHIFT", 0, &shard_shift);
          ReadInt64FromEnvVar("CACHE_PROFILER_BUCKET_SIZE", 10, reinterpret_cast<int64 *>(&bucket_size));
          ReadInt64FromEnvVar("CACHE_PROFILER_MAX_REUSE_DIST", 100000, reinterpret_cast<int64 *>(&max_reuse_dist));
          ReadInt64FromEnvVar("CACHE_PROFILER_SAMPLING_INTERVAL", 1, reinterpret_cast<int64 *>(&sampling_interval));
          ProfiledShardedLRUCache<K> *pscache;
          pscache = new ProfiledShardedLRUCache<K>(name, bucket_size, max_reuse_dist, sampling_interval, shard_shift, tunable_cache);
          if (tunable_cache != nullptr) {
            CacheManager::GetInstance().RegisterCache(*pscache->GetProfiler());
          }
          return pscache;
        default:
          LOG(INFO) << " Invalid Cache strategy, \
                       use LFU in multi-tier EmbeddingVariable "
                    << name;
          return new LFUCache<K>(name);
      }
  }
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_FACTORY_H_
