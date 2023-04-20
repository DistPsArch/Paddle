/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_HETERPS

#include <ThreadPool.h>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <vector>

#ifdef PADDLE_WITH_PSLIB
#include "common_value.h"  // NOLINT
#endif

#ifdef PADDLE_WITH_PSCORE
#include "paddle/fluid/distributed/ps/table/depends/feature_value.h"
#endif

#include "paddle/fluid/distributed/ps/thirdparty/round_robin.h"
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {

struct FixedChunk {
  size_t chunk_elements;
  char*  chunk_data;
};

template<typename _Tp>
class FixedMempool {
public:
  FixedMempool() {
    default_chunk_elements_ = 0;
    cur_chunk_elements_ = 0;
    cur_chunk_ = nullptr;
    cur_index_ = 0;
    chunk_list_index_ = 0;
  }
  ~FixedMempool() {
    for (auto iter : chunk_list_) {
      free(iter.chunk_data);
    }
    chunk_list_.clear();
  }
  //设置默认块大小
  void set_default_chunk_elemts(size_t chunk_elements) {
    default_chunk_elements_ = chunk_elements;
  } 
  //批量获取元素
  _Tp* get_batch_element(size_t batch_num) {
    //调用方一定是第一次获取
    if (unlikely(cur_chunk_ != NULL)) {
      abort();
    }
    if (chunk_list_.size() != 0) {
      if (chunk_list_[0].chunk_elements < batch_num) {
        free(chunk_list_[0].chunk_data);
        posix_memalign((void**)&(chunk_list_[0].chunk_data), alignof(_Tp), sizeof(_Tp) * batch_num);
        chunk_list_[0].chunk_elements = batch_num;
      }
    } else {
      FixedChunk tmp;
      tmp.chunk_elements = batch_num;
      posix_memalign((void**)&(tmp.chunk_data), alignof(_Tp), sizeof(_Tp) * batch_num);
      chunk_list_.push_back(tmp);
    }
    cur_chunk_ = (_Tp*)(chunk_list_[0].chunk_data);
    cur_chunk_elements_ = chunk_list_[0].chunk_elements;
    chunk_list_index_ = 1;
    cur_index_ = batch_num;
    return cur_chunk_;
  }
  //获得一个元素
  _Tp* get_one_element() {
    //没有可用元素，分配一个chunk
    if (unlikely(cur_chunk_ == NULL || cur_index_ >= cur_chunk_elements_)) {
      if (chunk_list_index_ < chunk_list_.size()) {
        cur_chunk_ = (_Tp*)chunk_list_[chunk_list_index_].chunk_data;
        cur_chunk_elements_ = chunk_list_[chunk_list_index_].chunk_elements;
      } else {
        FixedChunk tmp;
        tmp.chunk_elements = default_chunk_elements_;
        posix_memalign((void**)&(tmp.chunk_data), alignof(_Tp), sizeof(_Tp) * default_chunk_elements_);
        chunk_list_.push_back(tmp);
        cur_chunk_ = (_Tp*)tmp.chunk_data;
        cur_chunk_elements_ = default_chunk_elements_;
      }
      chunk_list_index_++;
      cur_index_ = 0;
    }
    size_t pre = cur_index_;
    cur_index_++;
    return cur_chunk_ + pre;
  }
  //占用的内存大小
  size_t get_size() {
    size_t ret = 0;
    for (auto& iter : chunk_list_) {
      ret += sizeof(_Tp) * iter.chunk_elements;
    }
    return ret;
  }
  void reset() {
    cur_chunk_ = NULL;
    cur_index_ = 0;
    chunk_list_index_ = 0;
  }
  void release_memory() {
    reset();
    for (auto iter : chunk_list_) {
      free(iter.chunk_data);
    }
    chunk_list_.clear();
  }
  int use_chunk() {
    return chunk_list_index_;
  }
private:
  size_t default_chunk_elements_; //默认创建块大小
  size_t cur_chunk_elements_; //当前正在操作块的块大小
  _Tp* cur_chunk_; //当前块数据
  std::vector<FixedChunk> chunk_list_; //块集合
  size_t cur_index_; //当前块索引
  size_t chunk_list_index_;
};

struct DupUnit {
  uint64_t key;
  DupUnit* next;
};

class KeyDupUnit {
public:
  KeyDupUnit() {
    buckets_ = nullptr;
    local_bucket_ = 0;
    local_bucket_max_ = 0;
    uniq_keys_num_ = 0;
  }
  void reset(size_t shard_size = 0, bool is_free = false) {
    if (shard_size == 0) {
      shard_size = local_bucket_;
    }
    local_bucket_ = shard_size;
    if (local_bucket_ > local_bucket_max_) {
      mempool_.release_memory();
    } else if (is_free && 2 * local_bucket_ < local_bucket_max_) {
      local_bucket_max_ = 0;
      mempool_.release_memory();
    } else {
      mempool_.reset();
    }
    mempool_.set_default_chunk_elemts(shard_size / 6 + 1);
    local_bucket_max_ = std::max(local_bucket_max_, local_bucket_);
    uniq_keys_num_ = 0;
    buckets_ = mempool_.get_batch_element(local_bucket_);
    memset(buckets_, 0, sizeof(DupUnit) * local_bucket_);
  }
  void batch_add_keys(const std::vector<uint64_t>& keys) {
    for (auto key : keys) {
      if (unlikely(key == 0)) {
        continue;
      }
      auto local_id = key % local_bucket_;
      if (buckets_[local_id].key == 0) {
        buckets_[local_id].key = key;
        uniq_keys_num_++;
        continue;
      }
      auto* pre_element = buckets_ + local_id;
      auto* cur_element = buckets_ + local_id;
      while (cur_element != NULL && cur_element->key != key) {
        pre_element = cur_element;
        cur_element = cur_element->next;
      }
      if (cur_element == NULL) {
        cur_element = mempool_.get_one_element();
        cur_element->key = key;
        cur_element->next = NULL;
        pre_element->next = cur_element;
        uniq_keys_num_++;
      }
    }
  }
  size_t get_uniq_key_size() {
    return uniq_keys_num_;
  }
  void trans_to_array(uint64_t* key_start) {
    for (size_t i = 0; i < local_bucket_; i++) {
      if (buckets_[i].key == 0) {
        continue;
      }
      *key_start++ = buckets_[i].key;
      auto cur_element = buckets_[i].next;
      while (cur_element != nullptr) {
        *key_start++ = cur_element->key;
        cur_element = cur_element->next;
      }
    }
  }

  void print_detail(int shard_id) {
    int size_4 = 0;
    int size_16 = 0;
    int size_64 = 0;
    int size_256 = 0;
    int size_1024 = 0;
    int size_4096 = 0;
    int size_16384 = 0;
    int size_max = 0;
    for (size_t i = 0; i < local_bucket_; i++) {
      int bucket_size = 0;
      if (buckets_[i].key == 0) {
        continue;
      }
      bucket_size++;
      auto cur_element = buckets_[i].next;
      while (cur_element != nullptr) {
        cur_element = cur_element->next;
        bucket_size++;
      }
      size_max = std::max(bucket_size, size_max);
      if (bucket_size >= 16384) {
        size_16384++;
      } else if (bucket_size >= 4096) {
        size_4096++;
      } else if (bucket_size >= 1024) {
        size_1024++;
      } else if (bucket_size >= 256) {
        size_256++;
      } else if (bucket_size >= 64) {
        size_64++;
      } else if (bucket_size >= 16) {
        size_16++;
      } else {
        size_4++;
      }
    }
    int use_chunk = mempool_.use_chunk();
    VLOG(0) << "lxch. shard_id: " << shard_id
            << "  bucket_size: " << local_bucket_
            << "  uniq_size: " << uniq_keys_num_
            << "  use_chunk: " << use_chunk
            << "  max_bucket: " << size_max
            << "  size_4: " << size_4
            << "  size_16: " << size_16
            << "  size_64: " << size_64
            << "  size_256: " << size_256
            << "  size_1024: " << size_1024
            << "  size_4096: " << size_4096
            << "  size_16384: " << size_16384;
  }

  size_t get_bucket_size() {
    return local_bucket_;   
  }
private:
  FixedMempool<DupUnit> mempool_; //内存分配器
  DupUnit* buckets_; //bucket
  size_t local_bucket_; //本地bucket个数
  size_t local_bucket_max_;
  size_t uniq_keys_num_; //去重后的key数量 
};

#define ratio_level  10000
class KeyDup {
public:
  KeyDup() {
    for (size_t i = 0; i < ratio_level; i++) {
      dup_ratio_[i] = NULL;
    }
  };
  ~KeyDup() {
    clear();
  }
  void init(size_t shard_size, size_t bucket_size, size_t avg_feagsign_per_ins, size_t total_ins) {
    shard_size_ = shard_size;
    shard_ = new KeyDupUnit[shard_size_];
    bucket_size = get_bucket_prime(bucket_size);
    for (size_t i = 0; i < shard_size_; i++) {
      shard_[i].reset(bucket_size);
    }
    avg_feagsign_per_ins_ = avg_feagsign_per_ins;
    for (size_t i = 0; i < ratio_level; i++) {
      dup_ratio_[i] = new double[shard_size];
      for (size_t j = 0; j < shard_size; j++) {
        dup_ratio_[i][j] = 0.0;
      }
    }
    total_ins_ = total_ins;
  }
  void batch_add_keys(size_t shard_id, const std::vector<uint64_t>& keys) {
    shard_[shard_id].batch_add_keys(keys);
  }
  size_t get_uniq_key_size(size_t shard_id) {
    return shard_[shard_id].get_uniq_key_size();
  }
  void trans_to_array(size_t shard_id, uint64_t* key_start) {
    return shard_[shard_id].trans_to_array(key_start);
  }
  void reset(size_t total_ins) {
    //先更新去重比例信息
    {
      int level = get_ratio_level(total_ins_);
      for (size_t i = 0; i < shard_size_; i++) {
        auto uniqkeys = shard_[i].get_uniq_key_size();
        double ratio = uniqkeys / 1.0 / total_ins_;
        if (fabs(dup_ratio_[level][i]) < 0.00001) {
          dup_ratio_[level][i] = ratio;
        } else {
          dup_ratio_[level][i] = (dup_ratio_[level][i] + ratio) / 2;
        }
      }
    }

    //重置本次的桶大小
    {
      static bool is_free = true;
      total_ins_ = total_ins;
      int level = get_ratio_level(total_ins_);
      for (size_t i = 0; i < shard_size_; i++) {
        size_t bucket_size = shard_[i].get_bucket_size();
        size_t uniqkeys = 0;
        if (fabs(dup_ratio_[level][i]) < 0.00001) {
          uniqkeys = total_ins_ * avg_feagsign_per_ins_ / 40 * 1.3 / shard_size_;
        } else {
          uniqkeys = size_t(dup_ratio_[level][i] * total_ins_) + 1;
        }
        double factor = uniqkeys / 1.0 / bucket_size;
        if (factor > 4) {
          size_t new_bucket_size = (size_t)(uniqkeys / 2);
          if (new_bucket_size < 10000) new_bucket_size = 10000;
          new_bucket_size = get_bucket_prime(new_bucket_size);
          shard_[i].reset(new_bucket_size, is_free);
        } else if (factor < 0.5) { //不做真正内存的缩小操作
          size_t new_bucket_size = (size_t)(uniqkeys / 2);
          if (new_bucket_size < 10000) new_bucket_size = 10000;
          new_bucket_size = get_bucket_prime(new_bucket_size);
          shard_[i].reset(new_bucket_size, is_free);
        } else {
          shard_[i].reset();
        }
      }
      is_free = false;
    }
  }
  void print_detail() {
    for (size_t i = 0; i < shard_size_; i++) {
      shard_[i].print_detail(i);
    }
  }
  bool is_init() {
    return shard_size_ != 0;
  }
private:
  void clear() {
    if (shard_ != NULL) {
      delete []shard_;
    }
    for (size_t i = 0; i < ratio_level; i++) {
      if (dup_ratio_[i] != NULL) {
        delete dup_ratio_[i];
      }
    }
  }
  int get_ratio_level(size_t ins) {
    ins = ins / 10000;
    int ii = 0;
    size_t tmp_ins = ins;
    while (tmp_ins != 0) {
      ii++;
      tmp_ins = tmp_ins / 2;
    }
    return ii;
  }
  size_t get_bucket_prime(size_t size) {
    static size_t ins[] = {11, 23, 47, 101, 223, 449, 853, 1621, 3137, 6343,
                          12611, 31891, 81971, 158699, 200257, 254659, 319849,
                          400217, 502961, 607681, 808351, 1002241, 1209287,
                          1502143, 1801363, 2109109, 2432471, 3217813, 4104031, 4801177,
                          5920529, 7093813, 8241551, 9345451, 10606157, 11800037,
                          13078657, 14439059, 16038709, 17643971, 19068559, 20622059,
                          22030751, 23823259, 25170881};
    if (size > 25170881) {
      return 25170881;
    }
    for (size_t i = 0; i < sizeof(ins)/sizeof(size_t); i++) {
      if (size < ins[i]) {
        return ins[i];
      }
    }
    return 25170881;
  }
  KeyDupUnit* shard_ = NULL;
  size_t avg_feagsign_per_ins_;
  size_t total_ins_;
  double* dup_ratio_[ratio_level];
  size_t shard_size_ = 0;
};

class HeterContext {
 public:
  virtual ~HeterContext() {
    if (!multi_mf_dim_) {
      for (size_t i = 0; i < mutex_.size(); ++i) {
        delete mutex_[i];
      }
      mutex_.clear();
    } else {
      for (size_t i = 0; i < dim_mutex_.size(); ++i) {
        for (size_t j = 0; j < dim_mutex_[i].size(); j++) {
          delete dim_mutex_[i][j];
        }
        dim_mutex_[i].clear();
      }
    }
  }
  Scope* scope_{nullptr};
  std::vector<std::vector<FeatureKey>> feature_keys_;
  std::vector<std::vector<std::vector<FeatureKey>>> feature_dim_keys_;
  std::vector<std::vector<std::vector<FeatureKey>>> device_task_keys_;

#ifdef PADDLE_WITH_PSLIB
  std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>> value_ptr_;
  std::vector<std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>>>
      device_task_ptr_;
  std::vector<std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>>>
      value_dim_ptr_;
  std::vector<std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>>>
      device_dim_ptr_;
#endif
#ifdef PADDLE_WITH_PSCORE
  std::vector<std::vector<paddle::distributed::FixedFeatureValue*>> value_ptr_;
  std::vector<std::vector<std::vector<paddle::distributed::FixedFeatureValue*>>>
      value_dim_ptr_;
  std::vector<std::vector<std::vector<paddle::distributed::FixedFeatureValue*>>>
      device_task_ptr_;
  std::vector<std::vector<std::vector<paddle::distributed::FixedFeatureValue*>>>
      device_dim_ptr_;
#endif
  std::vector<std::vector<FeatureKey>> device_keys_;
  std::vector<std::vector<std::vector<FeatureKey>>> device_dim_keys_;
  std::vector<std::mutex*> mutex_;
  std::vector<std::vector<std::mutex*>> dim_mutex_;
  int multi_mf_dim_ = 0;

  uint32_t shard_num_ = 37;
  uint64_t size() {
    uint64_t total_size = 0;
    for (auto& keys : feature_keys_) {
      total_size += keys.size();
    }
    return total_size;
  }
  void SetShardNum(uint32_t shard_num) { shard_num_ = shard_num; }
  uint32_t ShardNum() { return shard_num_; }
  void init(int shard_num, int device_num) {
    shard_num_ = shard_num;
    feature_keys_.resize(shard_num_);
    value_ptr_.resize(shard_num_);
    device_task_ptr_.resize(shard_num_);
    device_task_keys_.resize(shard_num_);
    for (size_t i = 0; i < device_task_ptr_.size(); i++) {
      device_task_ptr_[i].resize(device_num);
      device_task_keys_[i].resize(device_num);
    }

    device_keys_.resize(device_num);
    mutex_.resize(device_num);
    for (size_t i = 0; i < mutex_.size(); ++i) {
      mutex_[i] = new std::mutex();
    }
  }

  void init(int shard_num, int device_num, int dim_num) {
    shard_num_ = shard_num;
    feature_keys_.resize(shard_num_);
    feature_dim_keys_.resize(shard_num_);
    value_ptr_.resize(shard_num_);
    value_dim_ptr_.resize(shard_num_);
    device_task_ptr_.resize(shard_num_);
    device_task_keys_.resize(shard_num_);
    for (size_t i = 0; i < device_task_ptr_.size(); i++) {
      device_task_ptr_[i].resize(device_num);
      device_task_keys_[i].resize(device_num);
    }
    for (size_t i = 0; i < feature_dim_keys_.size(); i++) {
      feature_dim_keys_[i].resize(dim_num);
      value_dim_ptr_[i].resize(dim_num);
    }
    device_keys_.resize(device_num);

    device_dim_keys_.resize(device_num);
    device_dim_ptr_.resize(device_num);
    mutex_.resize(device_num);
    dim_mutex_.resize(device_num);
    for (size_t i = 0; i < mutex_.size(); ++i) {
      mutex_[i] = new std::mutex();
    }
    for (size_t i = 0; i < dim_mutex_.size(); ++i) {
      dim_mutex_[i].resize(dim_num);
      for (int j = 0; j < dim_num; j++) {
        dim_mutex_[i][j] = new std::mutex();
      }
    }
    multi_mf_dim_ = dim_num;
  }

  void Reset() {
    if (!multi_mf_dim_) {
      for (size_t i = 0; i < feature_keys_.size(); ++i) {
        feature_keys_[i].clear();
      }
      for (size_t i = 0; i < value_ptr_.size(); ++i) {
        value_ptr_[i].clear();
      }
      for (size_t i = 0; i < device_keys_.size(); ++i) {
        device_keys_[i].clear();
      }
      for (size_t i = 0; i < device_task_ptr_.size(); ++i) {
        for (size_t j = 0; j < device_task_ptr_[i].size(); ++j) {
          device_task_ptr_[i][j].clear();
          device_task_keys_[i][j].clear();
        }
      }
    } else {
      VLOG(3) << "Reset gpu task with dynamic mf dimention";
      for (size_t i = 0; i < feature_dim_keys_.size(); i++) {
        for (size_t j = 0; j < feature_dim_keys_[i].size(); j++) {
          feature_dim_keys_[i][j].clear();
        }
      }
      for (size_t i = 0; i < value_dim_ptr_.size(); i++) {
        for (size_t j = 0; j < value_dim_ptr_[i].size(); j++) {
          value_dim_ptr_[i][j].clear();
        }
      }

      for (size_t i = 0; i < device_dim_keys_.size(); i++) {
        for (size_t j = 0; j < device_dim_keys_[i].size(); j++) {
          device_dim_keys_[i][j].clear();
        }
      }
      for (size_t i = 0; i < device_dim_ptr_.size(); i++) {
        for (size_t j = 0; j < device_dim_ptr_[i].size(); j++) {
          device_dim_ptr_[i][j].clear();
        }
      }
    }
  }
  void batch_add_keys(
      const std::vector<std::unordered_set<uint64_t>>& thread_keys) {
    assert(thread_keys.size() == feature_keys_.size());

    for (uint32_t i = 0; i < shard_num_; i++) {
      int idx = 0;
      idx = feature_keys_[i].size();
      feature_keys_[i].resize(feature_keys_[i].size() + thread_keys[i].size());
      std::copy(thread_keys[i].begin(), thread_keys[i].end(),
                feature_keys_[i].begin() + idx);
    }
  }

  void batch_add_keys(int shard_num,
                      const robin_hood::unordered_set<uint64_t>& shard_keys) {
    int idx = feature_keys_[shard_num].size();
    feature_keys_[shard_num].resize(feature_keys_[shard_num].size() +
                                    shard_keys.size());
    std::copy(shard_keys.begin(), shard_keys.end(),
              feature_keys_[shard_num].begin() + idx);
  }

  void batch_add_keys(int shard_num, int dim_id,
                      const robin_hood::unordered_set<uint64_t>& shard_keys) {
    int idx = feature_dim_keys_[shard_num][dim_id].size();
    feature_dim_keys_[shard_num][dim_id].resize(
        feature_dim_keys_[shard_num][dim_id].size() + shard_keys.size());
    std::copy(shard_keys.begin(), shard_keys.end(),
              feature_dim_keys_[shard_num][dim_id].begin() + idx);
  }

  void UniqueKeys() {
    std::vector<std::thread> threads;
    auto unique_func = [this](int i) {
      auto& cur_keys = feature_keys_[i];
      std::sort(cur_keys.begin(), cur_keys.end());
      std::vector<FeatureKey>::iterator it;
      it = std::unique(cur_keys.begin(), cur_keys.end());
      cur_keys.resize(std::distance(cur_keys.begin(), it));
    };
    auto unique_dynamic_mf_func = [this](int i, int j) {
      auto& cur_keys = feature_dim_keys_[i][j];
      std::sort(cur_keys.begin(), cur_keys.end());
      std::vector<FeatureKey>::iterator it;
      it = std::unique(cur_keys.begin(), cur_keys.end());
      cur_keys.resize(std::distance(cur_keys.begin(), it));
    };
    if (!multi_mf_dim_) {
      for (uint32_t i = 0; i < shard_num_; i++) {
        threads.push_back(std::thread(unique_func, i));
      }
    } else {
      for (uint32_t i = 0; i < shard_num_; i++) {
        for (int j = 0; j < multi_mf_dim_; j++) {
          threads.push_back(std::thread(unique_dynamic_mf_func, i, j));
        }
      }
      VLOG(3) << "heter_context unique keys with dynamic mf dimention";
    }
    for (std::thread& t : threads) {
      t.join();
    }
  }
  uint16_t pass_id_;
};

}  // end namespace framework
}  // end namespace paddle
#endif
