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
#include <cstdint>
#include <memory>
#include <random>
#include "tensorflow/core/kernels/embedding_variable_test.h"
#include "sparsehash/dense_hash_map_lockless"

namespace tensorflow {
namespace embedding {
static constexpr int64 EMPTY_KEY = -1;
static constexpr int64 DELETED_KEY = -2;
static float PerfMemory(const std::vector<size_t>& num_elements) {
  std::uniform_int_distribution<int64_t> distrib(0, INT64_MAX - 1);
  std::mt19937_64 rand_engine((std::random_device())());
  using LockLessHashMap = google::dense_hash_map_lockless<int64_t, uint64_t*>;
  std::vector<std::unique_ptr<LockLessHashMap>> maps;
  size_t total_elements = 0;
  double start_mem = getResident() * getpagesize();
  for (const size_t num_element: num_elements) {
    double sub_start_mem = getResident() * getpagesize();
    std::unique_ptr<LockLessHashMap> map = std::make_unique<LockLessHashMap>();
    map->max_load_factor(1.5f);
    map->set_empty_key_and_value(EMPTY_KEY, 0);
    map->set_counternum(16);
    map->set_deleted_key(DELETED_KEY);
    for (size_t i = 0; i < num_element; ++i) {
      uint64_t *value_ptr = new uint64_t;
      *value_ptr = distrib(rand_engine);
      map->insert_lockless({distrib(rand_engine), value_ptr});
    }
    LOG(INFO)   << "bucket_count:" << map->bucket_count()
                << ", load_factor:" << map->load_factor()
                << ", max_load_factor:" << map->max_load_factor()
                << ", min_load_factor:" << map->min_load_factor();
    maps.emplace_back(std::move(map));
    double sub_end_mem = getResident() * getpagesize();
    LOG(INFO) << "elements=" << num_element << ", mem_mb=" << (sub_end_mem - sub_start_mem)/(1024 * 1024);
    total_elements += num_element;
  }
  double end_mem = getResident() * getpagesize();
  double used_mb = (end_mem - start_mem)/(1024 * 1024);
  LOG(INFO) << "[TestMemory]Use Memory: " << used_mb << ", Elements: " << total_elements;
  return used_mb;
}

TEST(LocklessHashMapMemoryTest, TestMemory) {
  PerfMemory({4631374, 78980, 816514, 2497});
}
} //namespace embedding
} //namespace tensorflow
