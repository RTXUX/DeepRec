#include "tensorflow/core/framework/embedding/cache_manager.h"

namespace tensorflow {
namespace embedding {

size_t MockTunableCache::GetCacheSize() const {
  return num_entries_ * GetCacheEntrySize();
}

void MockTunableCache::SetCacheSize(size_t new_size) {
  num_entries_ = new_size / GetCacheEntrySize();
}

size_t MockTunableCache::GetCacheEntrySize() const {
  return 8;
}

MockTunableCache::MockTunableCache(size_t size) : num_entries_(size / GetCacheEntrySize()) {}

CacheItem::CacheItem(size_t bucketSize, size_t origSize, size_t newSize, size_t entrySize, uint64_t vc, uint64_t mc,
                     double mr,
                     const std::vector<double> &mrc) : bucket_size(bucketSize), orig_size(origSize),
                                                       new_size(newSize), entry_size(entrySize), vc(vc), mc(mc), mr(mr),
                                                       mrc(mrc) {}

CacheItem::CacheItem() {}
} // namespace embedding
} // namespace tensorflow
