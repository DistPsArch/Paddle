// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_HETERPS

#include <algorithm>
#include <deque>

#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_PSLIB
void AfsWrapper::init(const std::string& fs_name, const std::string& fs_user,
                      const std::string& pass_wd, const std::string& conf) {
  int ret = afs_handler_.init(fs_name.c_str(), fs_user.c_str(), pass_wd.c_str(),
                              conf.c_str());
  if (ret != 0) {
    LOG(ERROR) << "AFS Init Error";
  }
}

int AfsWrapper::remove(const std::string& path) {
  return afs_handler_.remove(path);
}

int AfsWrapper::mkdir(const std::string& path) {
  return afs_handler_.mkdir(path);
}

std::vector<std::string> AfsWrapper::list(const std::string& path) {
  return afs_handler_.list(path);
}

int AfsWrapper::exist(const std::string& path) {
  return afs_handler_.exist(path);
}

int AfsWrapper::upload(const std::string& local_file,
                       const std::string& afs_file) {
  return afs_handler_.upload_file(local_file, afs_file);
}

int AfsWrapper::download(const std::string& local_file,
                         const std::string& afs_file) {
  return afs_handler_.download_file(local_file, afs_file);
}

int AfsWrapper::touchz(const std::string& path) {
  return afs_handler_.touchz(path);
}

std::string AfsWrapper::cat(const std::string& path) {
  return afs_handler_.cat(path);
}

int AfsWrapper::mv(const std::string& old_path, const std::string& dest_path) {
  return afs_handler_.mv(old_path, dest_path);
}
#endif

std::shared_ptr<PSGPUWrapper> PSGPUWrapper::s_instance_ = NULL;
bool PSGPUWrapper::is_initialized_ = false;
std::mutex PSGPUWrapper::ins_mutex;
#ifdef PADDLE_WITH_PSLIB
void PSGPUWrapper::InitAfsApi(const std::string& fs_name,
                              const std::string& fs_user,
                              const std::string& pass_wd,
                              const std::string& conf) {
  int ret = afs_handler_.init(fs_name.c_str(), fs_user.c_str(), pass_wd.c_str(),
                              conf.c_str());
  if (ret != 0) {
    VLOG(0) << "AFS Init Error";
  }
  use_afs_api_ = 1;
}
#endif
void PSGPUWrapper::PreBuildTask(std::shared_ptr<HeterContext> gpu_task) {
  VLOG(3) << "PSGPUWrapper::BuildGPUPSTask begin";
  platform::Timer timeline;
  timeline.Start();
  int device_num = heter_devices_.size();
//   if (!multi_mf_dim_) {
//    gpu_task->init(thread_keys_shard_num_, device_num);
//  } else {
    gpu_task->init(thread_keys_shard_num_, device_num, multi_mf_dim_);
//  }

  std::vector<std::thread> threads;

  // data should be in input channel
 // if (!multi_mf_dim_) {
 //   thread_keys_.resize(thread_keys_thread_num_);
 //   for (int i = 0; i < thread_keys_thread_num_; i++) {
 //     thread_keys_[i].resize(thread_keys_shard_num_);
 //   }
//  } else {

    thread_dim_keys_.resize(thread_keys_thread_num_);
    for (int i = 0; i < thread_keys_thread_num_; i++) {
      thread_dim_keys_[i].resize(thread_keys_shard_num_);
      for (int j = 0; j < thread_keys_shard_num_; j++) {
        thread_dim_keys_[i][j].resize(multi_mf_dim_);
      }
    }

 // }

  dataset_mutex_.lock();
  Dataset* cur_dataset = dataset_pipe_.front();
  dataset_pipe_.pop();
  dataset_mutex_.unlock();
  std::string data_set_name = std::string(typeid(*cur_dataset).name());

  double* add_time_info = new double[thread_keys_shard_num_ * multi_mf_dim_];
  for (int i = 0; i < thread_keys_shard_num_ * multi_mf_dim_; i++) {
    add_time_info[i] = 0;
  }
  double* dispatch_time_info = new double[16*2];
  static KeyDup dup_ins;
  if (data_set_name.find("SlotRecordDataset") != std::string::npos) {
    platform::Timer timeline_1;
    timeline_1.Start();
    SlotRecordDataset* dataset = dynamic_cast<SlotRecordDataset*>(cur_dataset);
    auto input_channel = dataset->GetInputChannel();
    VLOG(3) << "buildtask::inputslotchannle size: "
            << input_channel->Size();
    const std::deque<SlotRecord>& vec_data = input_channel->GetData();
    size_t total_len = vec_data.size();
    //自适应调节map的bucket大小
    if (!dup_ins.is_init()) {
      size_t feasign_size = 0;
      for (auto iter = vec_data.begin(); iter != vec_data.end() && iter != vec_data.begin() + 500; iter++) {
        feasign_size += (*iter)->slot_uint64_feasigns_.slot_values.size();
      }
      size_t avg_size = feasign_size / 500;
      size_t uniq_size = total_len * avg_size / 40 * 1.3;
      uniq_size = uniq_size / thread_keys_shard_num_ / multi_mf_dim_;
      if (uniq_size < 10000) uniq_size = 10001;
      uniq_size = uniq_size / 10 * 10 + 1;
      dup_ins.init(thread_keys_shard_num_ * multi_mf_dim_, uniq_size, avg_size, total_len);
    } else {
      dup_ins.reset(total_len);
    }
    timeline_1.Pause();
    auto lxch_step_1 = timeline_1.ElapsedSec();
    timeline_1.Start();

    const uint32_t commit_id = 5000;
    std::atomic<uint64_t> task_doing_num(0);
    const uint64_t max_task_limit = 6000;
    auto& l_dup_ins = dup_ins;
    auto fill_func = [this, &task_doing_num, add_time_info](std::shared_ptr< std::vector<uint64_t> > task, size_t shard_id) -> void {
      platform::Timer timeline_info;
      timeline_info.Start();
      l_dup_ins.batch_add_keys(shard_id, *task);
      task_doing_num.fetch_sub(1);
      timeline_info.Pause();
      add_time_info[shard_id] += timeline_info.ElapsedUS();
    };
    
    auto first_uniq_func = [this, &task_doing_num, &fill_func, &vec_data, dispatch_time_info](int begin_index, int end_index, int dispach_i) -> void {
      platform::Timer timeline_info;
      platform::Timer timeline_info_1;
      double tt_tt = 0;
      timeline_info.Start();
      std::vector< std::shared_ptr< std::vector<uint64_t> > > cache_key;
      cache_key.resize(thread_keys_shard_num_ * multi_mf_dim_);
      for (auto &iter : cache_key) {
        iter = std::shared_ptr< std::vector<uint64_t> >(new std::vector<uint64_t>());
        iter->reserve(commit_id + 200);
      }
      auto iter_start = vec_data.begin() + begin_index;
      auto iter_end = iter_start + (end_index - begin_index);
      int uniq_thread_size = uniq_thread_pool_.size();
      for (auto iter = iter_start; iter != iter_end; iter++) {
        const auto& ins = *iter;
        const auto& feasign_v = ins->slot_uint64_feasigns_.slot_values;
        const auto& slot_offset = ins->slot_uint64_feasigns_.slot_offsets;
        
        for (size_t slot_idx = 0; slot_idx < slot_offset_vector_.size(); slot_idx++) {
          for (size_t j = slot_offset[slot_offset_vector_[slot_idx]]; j < slot_offset[slot_offset_vector_[slot_idx] + 1]; j++) {
            int dim_id = slot_index_vec_[slot_idx];
            int shard_id = feasign_v[j] % thread_keys_shard_num_;
            int uniq_index = shard_id * multi_mf_dim_ + dim_id;
            cache_key[uniq_index]->emplace_back(feasign_v[j]);
            if (cache_key[uniq_index]->size() >= commit_id) {
              timeline_info_1.Start();
              while(task_doing_num.load() >= max_task_limit) {
                usleep(500);
                continue;
              }
              timeline_info_1.Pause();
              tt_tt += timeline_info_1.ElapsedUS();
              task_doing_num.fetch_add(1);
              uniq_thread_pool_[shard_id % uniq_thread_size]->enqueue(fill_func, cache_key[uniq_index], uniq_index);
              cache_key[uniq_index] = std::shared_ptr< std::vector<uint64_t> >(new std::vector<uint64_t>());
              cache_key[uniq_index]->reserve(commit_id + 200);
            }
          }
        }
      };

      for (size_t i = 0; i < cache_key.size(); i++) {
        if (cache_key[i]->size() == 0) {
          continue;
        }
        timeline_info_1.Start();
        while(task_doing_num.load() >= max_task_limit) {
          usleep(1000);
          continue;
        }
        timeline_info_1.Pause();
        tt_tt += timeline_info_1.ElapsedUS();
        size_t shard_id = i / multi_mf_dim_;
        task_doing_num.fetch_add(1);
        uniq_thread_pool_[shard_id % uniq_thread_size]->enqueue(fill_func, cache_key[i], i);
      }
      timeline_info.Pause();
      dispatch_time_info[dispach_i*2] = timeline_info.ElapsedUS();
      dispatch_time_info[dispach_i*2 + 1] = tt_tt;
    };

    //开始做数据分发
    const int thread_num = 16;
    size_t per_thread_len = total_len / thread_num + 1;
    std::vector<std::thread> threads;
    for (int i = 0; i < thread_num; i++) {
      size_t start_index = i * per_thread_len;
      size_t end_index = std::min(start_index + per_thread_len, total_len);
      if (start_index < end_index) {
        threads.push_back( std::thread(first_uniq_func, start_index, end_index, i));
      }
    }
    for (auto& t : threads) {
      t.join();
    }

    while (task_doing_num.load() != 0) {
      usleep(1000);
    }
    timeline_1.Pause();
    auto lxch_step_2 = timeline_1.ElapsedSec();
    timeline_1.Start();

    auto func_uniq = [this, &gpu_task] (int shard_id, int dim_id) -> void {
      auto& result_vec = gpu_task->feature_dim_keys_[shard_id][dim_id];
      int dup_shard_id = shard_id * multi_mf_dim_ + dim_id;
      size_t uniq_size = l_dup_ins.get_uniq_key_size(dup_shard_id);
      //加一个0进来，因为去重集合会忽略到0的feasign，就可能导致结果集里面没有，即使多个0，也无所谓
      if (shard_id == 0 && dim_id + 1 == (int)index_dim_vec_.size() && 0) {
        result_vec.resize(uniq_size + 1);
        result_vec[0] = 0;
        l_dup_ins.trans_to_array(dup_shard_id, result_vec.data() + 1);
      } else {
        result_vec.resize(uniq_size);
        l_dup_ins.trans_to_array(dup_shard_id, result_vec.data());
      }
      
      std::sort(result_vec.begin(), result_vec.end());
    };
    std::vector<std::future<void>> task_futures;
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        task_futures.emplace_back(uniq_thread_pool_[i % uniq_thread_pool_.size()]->enqueue(func_uniq, i, j));
      }
    }
    for (auto& f : task_futures) {
      f.wait();
    }
    task_futures.clear();

    timeline_1.Pause();
    auto lxch_step_3 = timeline_1.ElapsedSec();
    VLOG(0) << "dedup keys time  init: " << lxch_step_1 
            << "  dedup:" << lxch_step_2 
            << "  sort:" << lxch_step_3;

  } else {
    CHECK(false);
  }
  
  timeline.Pause();
  VLOG(0) << "GpuPs task unique cost " << timeline.ElapsedSec() << " seconds.";
  /*
  for (int i = 0; i < 16; i++) {
    VLOG(0) << "lxch dispatch thread_id: " << i
            << "  all_time: " << dispatch_time_info[2*i]
            << "  wait_time: " << dispatch_time_info[2*i+1];
  }
  for (int i = 0; i < thread_keys_shard_num_ * multi_mf_dim_; i++) {
    VLOG(0) << "lxch add shard_id: " << i
            << " all_time: " << add_time_info[i];
  }
  dup_ins.print_detail();
  */
  delete []add_time_info;
  delete []dispatch_time_info;
}

void PSGPUWrapper::BuildPull(std::shared_ptr<HeterContext> gpu_task) {
  platform::Timer timeline;
  std::vector<std::future<void>> task_futures;
  int device_num = heter_devices_.size();
  auto& local_keys = gpu_task->feature_keys_;
  auto& local_ptr = gpu_task->value_ptr_;

  static std::vector<std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>>> value_dim_ptr;
  value_dim_ptr.resize(thread_keys_shard_num_);
  for (size_t i = 0; i < value_dim_ptr.size(); i++) {
    value_dim_ptr[i].resize(multi_mf_dim_);
    for (size_t j = 0; j < value_dim_ptr[i].size(); j++) {
      value_dim_ptr[i][j].clear();
    }
  }
  

  auto& local_dim_keys = gpu_task->feature_dim_keys_;
  auto& local_dim_ptr = value_dim_ptr;

  // auto& device_keys = gpu_task->device_keys_;
  // auto& device_vals = gpu_task->device_values_;
  auto& device_dim_keys = gpu_task->device_dim_keys_;
  auto& device_dim_ptr = gpu_task->device_dim_ptr_;
  auto& device_dim_mutex = gpu_task->dim_mutex_;
  if (multi_mf_dim_) {
    for (size_t dev = 0; dev < device_dim_keys.size(); dev++) {
      device_dim_keys[dev].resize(multi_mf_dim_);
      device_dim_ptr[dev].resize(multi_mf_dim_);
    }
  }
  // auto& device_mutex = gpu_task->mutex_;

  std::vector<std::thread> threads(thread_keys_shard_num_);
#ifdef PADDLE_WITH_PSLIB
  auto fleet_ptr = FleetWrapper::GetInstance();
#endif
#ifdef PADDLE_WITH_PSCORE
  auto fleet_ptr = paddle::distributed::FleetWrapper::GetInstance();
#endif


#if (defined PADDLE_WITH_PSLIB) && (defined PADDLE_WITH_HETERPS)
  // get day_id: day nums from 1970
  struct std::tm b;
  b.tm_year = year_ - 1900;
  b.tm_mon = month_ - 1;
  b.tm_mday = day_;
  b.tm_min = b.tm_hour = b.tm_sec = 0;
  std::time_t seconds_from_1970 = std::mktime(&b);
  int day_id = seconds_from_1970 / 86400;
  fleet_ptr->pslib_ptr_->_worker_ptr->set_day_id(table_id_, day_id);
#endif
  fleet_ptr->pslib_ptr_->_worker_ptr->acquire_table_mutex(0);

  timeline.Start();
  auto ptl_func = [this, &local_keys, &local_ptr, &fleet_ptr, &gpu_task](int i) {
    size_t key_size = local_keys[i].size();
    int32_t status = -1;
#ifdef PADDLE_WITH_PSLIB
    // auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse_ptr(
    //    reinterpret_cast<char**>(local_ptr[i].data()), this->table_id_,
    //    local_keys[i].data(), key_size);
    int32_t cnt = 0;
    while (true) {
      auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse_ptr(i,
          reinterpret_cast<char**>(local_ptr[i].data()), this->table_id_,
          local_keys[i].data(), key_size, gpu_task->pass_id_);
      bool flag = true;

      tt.wait();

      try {
        status = tt.get();
      } catch (const std::future_error& e) {
        VLOG(0) << "Caught a future_error with code" << e.code()
                << ", Message:" << e.what();
      }
      if (status != 0) {
        VLOG(0) << "fleet pull sparse failed, status[" << status << "]";
        sleep(sleep_seconds_before_fail_exit_);
        flag = false;
        cnt++;
      }
      if (cnt > 3) {
        VLOG(0) << "fleet pull sparse failed, retry 3 times";
        exit(-1);
      }

      if (flag) {
        break;
      }
    }
#endif
#ifdef PADDLE_WITH_PSCORE
    int32_t cnt = 0;
    while (true) {
      auto tt = fleet_ptr->worker_ptr_->PullSparsePtr(
          reinterpret_cast<char**>(local_ptr[i].data()), this->table_id_,
          local_keys[i].data(), key_size);
      bool flag = true;

      tt.wait();

      try {
        status = tt.get();
      } catch (const std::future_error& e) {
        VLOG(0) << "Caught a future_error with code" << e.code()
                << ", Message:" << e.what();
      }
      if (status != 0) {
        VLOG(0) << "fleet pull sparse failed, status[" << status << "]";
        sleep(sleep_seconds_before_fail_exit_);
        flag = false;
        cnt++;
      }
      if (cnt > 3) {
        VLOG(0) << "fleet pull sparse failed, retry 3 times";
        exit(-1);
      }

      if (flag) {
        break;
      }
    }
#endif
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(300);
      exit(-1);
    } else {
      VLOG(3) << "FleetWrapper Pull sparse to local done with table size: "
              << local_keys[i].size();
    }
  };

  auto ptl_dynamic_mf_func = [this, &local_dim_keys, &local_dim_ptr,
                              &fleet_ptr, &gpu_task](int i, int j) {
#ifdef PADDLE_WITH_PSLIB
    size_t key_size = local_dim_keys[i][j].size();
    int32_t status = -1;
    int32_t cnt = 0;
    local_dim_ptr[i][j].resize(key_size);
    VLOG(0) << "GpuPs shard: " << i << "mf dim: " << index_dim_vec_[j]
            << " key len: " << key_size;
    while (true) {
      auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse_ptr(i,
          reinterpret_cast<char**>(local_dim_ptr[i][j].data()), this->table_id_,
          local_dim_keys[i][j].data(), key_size, gpu_task->pass_id_);
      bool flag = true;

      tt.wait();

      try {
        status = tt.get();
      } catch (const std::future_error& e) {
        VLOG(0) << "Caught a future_error with code" << e.code()
                << ", Message:" << e.what();
      }
      if (status != 0) {
        VLOG(0) << "fleet pull sparse failed, status[" << status << "]";
        sleep(sleep_seconds_before_fail_exit_);
        flag = false;
        cnt++;
      }
      if (cnt > 3) {
        VLOG(0) << "fleet pull sparse failed, retry 3 times";
        exit(-1);
      }

      if (flag) {
        break;
      }
    }
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(300);
      exit(-1);
    } else {
      VLOG(3) << "FleetWrapper Pull sparse to local done with table size: "
              << local_dim_keys[i][j].size();
    }
#endif
  };
  if (!multi_mf_dim_) {
    for (size_t i = 0; i < threads.size(); i++) {
      threads[i] = std::thread(ptl_func, i);
    }
  } else {
    threads.resize(thread_keys_shard_num_ * multi_mf_dim_);

    std::vector<std::future<void>> task_futures;
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        //threads[i * multi_mf_dim_ + j] = std::thread(ptl_dynamic_mf_func, i, j);
        task_futures.emplace_back(pull_thread_pool_[i]->enqueue(ptl_dynamic_mf_func, i, j));
      }
    }
    for (auto& f : task_futures) {
      f.wait();
    }
    task_futures.clear();
  }
  if (!multi_mf_dim_) {
    for (std::thread& t : threads) {
      t.join();
    }
  }
  fleet_ptr->pslib_ptr_->_worker_ptr->release_table_mutex(0);
  timeline.Pause();
  VLOG(0) << "pull sparse from CpuPS into GpuPS cost " << timeline.ElapsedSec()
          << " seconds.";
  if (multi_node_) {
    auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
    if (!gloo_wrapper->IsInitialized()) {
      VLOG(0) << "GLOO is not inited";
      gloo_wrapper->Init();
    }
    gloo_wrapper->Barrier();
  }

  timeline.Start();
  std::vector<std::vector<std::pair<uint64_t, char*>>> pass_values;

  bool record_status = false;
#ifdef PADDLE_WITH_PSLIB
  uint16_t pass_id = 0;
  if (multi_node_) {
    record_status = fleet_ptr->pslib_ptr_->_worker_ptr->take_sparse_record(
        table_id_, pass_id, pass_values);
  }
#endif
  auto& device_task_keys = gpu_task->device_task_keys_;
  auto& device_task_ptrs = gpu_task->device_task_ptr_;

  /*
  auto build_pull_dynamic_mf_func = [this, device_num, &local_dim_keys,
                                &local_dim_ptr, &device_dim_keys,
                                &device_dim_ptr,
                                &device_dim_mutex](int i, int j) {
#ifdef PADDLE_WITH_PSLIB
    std::vector<std::vector<FeatureKey>> task_keys(device_num);
    std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>> task_ptrs(
        device_num);
    for (size_t k = 0; k < local_dim_keys[i][j].size(); k++) {
      int shard = local_dim_keys[i][j][k] % device_num;
      task_keys[shard].push_back(local_dim_keys[i][j][k]);
      task_ptrs[shard].push_back(local_dim_ptr[i][j][k]);
    }
    // allocate local keys to devices
    for (int dev = 0; dev < device_num; dev++) {

        device_dim_mutex[dev][j]->lock();

        int len = task_keys[dev].size();
        int cur = device_dim_keys[dev][j].size();
        device_dim_keys[dev][j].resize(device_dim_keys[dev][j].size() +
                                         len);
        device_dim_ptr[dev][j].resize(device_dim_ptr[dev][j].size() + len);
        for (int k = 0; k < len; ++k) {
          device_dim_keys[dev][j][cur + k] = task_keys[dev][k];
          device_dim_ptr[dev][j][cur + k] = task_ptrs[dev][k];
        }
        device_dim_mutex[dev][j]->unlock();
      
    }
#endif
  };
  */
  std::vector<std::shared_ptr<std::atomic<uint32_t>> > device_keys_num;
  device_keys_num.resize(device_num * multi_mf_dim_);
  for (size_t i = 0; i < device_keys_num.size(); i++) {
    device_keys_num[i].reset(new std::atomic<uint32_t>());
    device_keys_num[i]->store(0);
  }
  std::atomic<uint32_t> barrir_sum;
  barrir_sum.store(thread_keys_shard_num_ * multi_mf_dim_);
  auto build_pull_dynamic_mf_func = [this, device_num, &local_dim_keys,
                                &local_dim_ptr, &device_dim_keys,
                                &device_dim_ptr,
                                &device_dim_mutex,
                                &device_keys_num,
                                &barrir_sum](int i, int j) {
#ifdef PADDLE_WITH_PSLIB
    //先统计大小
    std::vector<uint32_t> local_keys_num;
    local_keys_num.resize(device_num * multi_mf_dim_);
    for (size_t i = 0; i < local_keys_num.size(); i++) {
      local_keys_num[i] = 0;
    }
    for (size_t k = 0; k < local_dim_keys[i][j].size(); k++) {
      uint32_t shard = local_dim_keys[i][j][k] % device_num;
      uint32_t shard_idx = shard * multi_mf_dim_ + j;
      local_keys_num[shard_idx]++;
    }
    for (size_t i = 0; i < local_keys_num.size(); i++) {
      device_keys_num[i]->fetch_add(local_keys_num[i]);
    }
    barrir_sum.fetch_sub(1);
    while (barrir_sum.load() != 0) {
      usleep(200);
    }

    auto fill_func = [] (std::mutex* mutex, std::vector<FeatureKey>& keys, 
                          std::vector<paddle::ps::DownpourFixedFeatureValue*>& values, 
                          std::vector<std::pair<FeatureKey, paddle::ps::DownpourFixedFeatureValue*>>& src_kv,
                          uint32_t keys_sum) -> void {
      mutex->lock();
      if (keys.size() == 0) {
        keys.reserve(keys_sum);
        values.reserve(keys_sum);
      }
      int len = src_kv.size();
      for (int i = 0; i < len; i++) {
        keys.push_back(src_kv[i].first);
        values.push_back(src_kv[i].second);
      }
      mutex->unlock();
    };

    //回填结果
    std::vector<std::vector<std::pair<FeatureKey, paddle::ps::DownpourFixedFeatureValue*>>> task_kv(device_num);
    for (size_t k = 0; k < local_dim_keys[i][j].size(); k++) {
      uint32_t shard = local_dim_keys[i][j][k] % device_num;
      task_kv[shard].push_back(std::make_pair(local_dim_keys[i][j][k], local_dim_ptr[i][j][k]));
      if (task_kv[shard].size() > 5000) {
        uint32_t shard_idx = shard * multi_mf_dim_ + j;
        uint32_t shard_sum = device_keys_num[shard_idx]->load();
        fill_func(device_dim_mutex[shard][j], device_dim_keys[shard][j], device_dim_ptr[shard][j],
                  task_kv[shard], shard_sum);
        task_kv[shard].clear();
      }
    }
    for (int dev = 0; dev < device_num; dev++) {
      if (task_kv[dev].size() > 0) {
        uint32_t shard_idx = dev * multi_mf_dim_ + j;
        uint32_t shard_sum = device_keys_num[shard_idx]->load();
        fill_func(device_dim_mutex[dev][j], device_dim_keys[dev][j], device_dim_ptr[dev][j],
                  task_kv[dev], shard_sum);
        task_kv[dev].clear();
      }
    }
#endif
  };

  auto build_func = [device_num, record_status, &pass_values, &local_keys,
                     &local_ptr, &device_task_keys, &device_task_ptrs](int i) {
    auto& task_keys = device_task_keys[i];
#ifdef PADDLE_WITH_PSLIB
    auto& task_ptrs = device_task_ptrs[i];
#endif

#ifdef PADDLE_WITH_PSCORE
    auto& task_ptrs = device_task_ptrs[i];
#endif

    for (size_t j = 0; j < local_keys[i].size(); j++) {
      int shard = local_keys[i][j] % device_num;
      task_keys[shard].push_back(local_keys[i][j]);
      task_ptrs[shard].push_back(local_ptr[i][j]);
    }
#ifdef PADDLE_WITH_PSLIB
    if (record_status) {
      size_t local_keys_size = local_keys.size();
      size_t pass_values_size = pass_values.size();
      for (size_t j = 0; j < pass_values_size; j += local_keys_size) {
        auto& shard_values = pass_values[j];
        for (size_t pair_idx = 0; pair_idx < pass_values[j].size();
             pair_idx++) {
          auto& cur_pair = shard_values[pair_idx];
          int shard = cur_pair.first % device_num;
          task_keys[shard].push_back(cur_pair.first);
          task_ptrs[shard].push_back(
              (paddle::ps::DownpourFixedFeatureValue*)cur_pair.second);
        }
      }
    }
#endif
  };
  if (!multi_mf_dim_) {
    for (int i = 0; i < thread_keys_shard_num_; i++) {
      task_futures.emplace_back(hbm_thread_pool_[i]->enqueue(build_func, i));
    }
    for (auto& f : task_futures) {
      f.wait();
    }
    task_futures.clear();
    VLOG(0) << "GpuPs build hbmps done";
  }


  for (int i = 0; i < thread_keys_shard_num_; i++) {
    for (int j = 0; j < multi_mf_dim_; j++) {
      threads[i * multi_mf_dim_ + j] =
          std::thread(build_pull_dynamic_mf_func, i, j);
    }
  }
  for (std::thread& t : threads) {
    t.join();
  }
  
  timeline.Pause();
  VLOG(0) << "GpuPs prepare for build hbm cost " << timeline.ElapsedSec()
          << " seconds.";
}

void PSGPUWrapper::BuildGPUTask(std::shared_ptr<HeterContext> gpu_task) {
  int device_num = heter_devices_.size();
  platform::Timer timeline;
  timeline.Start();

  std::vector<size_t> feature_keys_count(device_num);
  size_t size_max = 0;
  if (!multi_mf_dim_) {
    for (int i = 0; i < device_num; i++) {
      feature_keys_count[i] = gpu_task->device_keys_[i].size();
      VLOG(0) << i << " card contains feasign nums: " << feature_keys_count[i];
      size_max = std::max(size_max, feature_keys_count[i]);
    }
  } else {
    for (int i = 0; i < device_num; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        feature_keys_count[i] += gpu_task->device_dim_ptr_[i][j].size();
        VLOG(1) << i << " card with dynamic mf dim: " << index_dim_vec_[j] << " dim index: " << j << " contains feasign nums: "
              << gpu_task->device_dim_ptr_[i][j].size();
      }
      VLOG(1) << i << " card with dynamic mf contains feasign nums total: "
              << feature_keys_count[i];
      size_max = std::max(size_max, feature_keys_count[i]);
    }
  }
  if (HeterPs_) {
    delete HeterPs_;
    HeterPs_ = nullptr;
  }
  if (size_max <= 0) {
    VLOG(0) << "Skip build gpu ps cause feasign nums = " << size_max;
    return;
  }
  std::vector<std::thread> threads(device_num);
  auto* accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();

  HeterPs_ = HeterPsBase::get_instance(size_max, resource_, accessor_type_, optimizer_type_);
  HeterPs_->set_nccl_comm_and_size(inner_comms_, inter_comms_, node_size_);
  HeterPs_->set_sparse_sgd(optimizer_config_);
  HeterPs_->set_embedx_sgd(optimizer_config_);

  // auto build_func = [this, &gpu_task, &feature_keys_count](int i) {
  //  VLOG(3) << "building table: " << i;
  //  this->HeterPs_->build_ps(i, gpu_task->device_keys_[i].data(),
  //                           gpu_task->device_values_[i].data(),
  //                           feature_keys_count[i], 500000, 2);
  //  if (feature_keys_count[i] > 0) {
  //    HeterPs_->show_one_table(i);
  //  }
  //};

  // multi-thread process
  auto build_dymf_mem_pool = [this, &gpu_task, &accessor_wrapper_ptr](int i, int j) {
    this->HeterPs_->set_multi_mf_dim(multi_mf_dim_, max_mf_dim_);
    int mf_dim = this->index_dim_vec_[j];
    size_t feature_value_size =
        accessor_wrapper_ptr->GetFeatureValueSize(mf_dim);
    VLOG(3) << "build dymf mem pool with device:" << i << " dim:" << mf_dim << " feature_value_size:" << feature_value_size;
    auto& device_dim_keys = gpu_task->device_dim_keys_[i][j];
    auto& device_dim_ptrs = gpu_task->device_dim_ptr_[i][j];
    size_t len = device_dim_keys.size();
    CHECK(len == device_dim_ptrs.size());
    this->mem_pools_[i * this->multi_mf_dim_ + j] = new MemoryPool(len, feature_value_size);
  };

  auto build_dymf_hbm_pool = [this, &gpu_task, &accessor_wrapper_ptr](int i, int j) {

    auto& device_dim_keys = gpu_task->device_dim_keys_[i][j];
    size_t len = device_dim_keys.size();
    int mf_dim = this->index_dim_vec_[j];
    size_t feature_value_size =
        accessor_wrapper_ptr->GetFeatureValueSize(mf_dim);
    auto& mem_pool = this->mem_pools_[i * this->multi_mf_dim_ + j];
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    this->hbm_pools_[i * this->multi_mf_dim_ + j] = new HBMMemoryPool(mem_pool);
    auto& cur_pool = this->hbm_pools_[i * this->multi_mf_dim_ + j];

    this->HeterPs_->build_ps(i, device_dim_keys.data(),
                             cur_pool->mem(), len, feature_value_size,
                             500000, 2);
    if (device_dim_keys.size() > 0) {
      VLOG(3) << "show table: " << i << " table kv size: " << device_dim_keys.size() << "dim: " << mf_dim << " len: " << len;
      HeterPs_->show_one_table(i);
    }
    delete mem_pool;
  };

  int thread_num = 16;
  auto build_dynamic_mf_func = [this, &gpu_task, thread_num, &accessor_wrapper_ptr](int i, int j, int z) {
    int mf_dim = this->index_dim_vec_[j];
    VLOG(3) << "building table: " << i << "with mf dim: " << mf_dim;

    auto& device_dim_keys = gpu_task->device_dim_keys_[i][j];
    auto& device_dim_ptrs = gpu_task->device_dim_ptr_[i][j];

    size_t len = device_dim_keys.size();
    CHECK(len == device_dim_ptrs.size());

    auto& mem_pool = this->mem_pools_[i * this->multi_mf_dim_ + j];

    // ============ add for multi-thread ================ 
    int len_per_thread = len / thread_num;
    int remain = len % thread_num;
    int left = -1, right = -1;
    
    int real_len = len_per_thread;
    if (z < remain) real_len++;
    
    if (z < remain) {
      left = z * (len_per_thread + 1);
      right = left + real_len;
    } else {
      left = remain * (len_per_thread + 1) + (z - remain) * len_per_thread;
      right = left + real_len;
    }
    // ============ add for multi-thread ================ 

    for (int k = left; k < right; k++) {
      float* val = (float*)mem_pool->mem_address(k);
      accessor_wrapper_ptr->BuildFill(val, device_dim_ptrs[k], cpu_accessor_, mf_dim, accessor_type_);
    }
  };
    threads.resize(device_num * multi_mf_dim_);
    for (int i = 0; i < device_num; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        threads[i + j * device_num] = std::thread(build_dymf_mem_pool, i, j);
      }
    }
    for (std::thread& t : threads) {
      t.join();
    }
    threads.clear();
    // multi-thread process
    threads.resize(device_num * multi_mf_dim_ * thread_num);
    for (int i = 0; i < device_num; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        for (int k = 0; k < thread_num; k++) {
          threads[(i + j * device_num) * thread_num + k] = std::thread(build_dynamic_mf_func, i, j, k);
        }
      }
    }
    for (std::thread& t : threads) {
      t.join();
    }
    threads.clear();
    threads.resize(device_num * multi_mf_dim_);
    for (int i = 0; i < device_num; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        threads[i + j * device_num] = std::thread(build_dymf_hbm_pool, i, j);
      }
    }
    for (std::thread& t : threads) {
      t.join();
    }
    threads.clear();
  timeline.Pause();
  VLOG(0) << "GpuPs build table total costs: " << timeline.ElapsedSec()
          << " s.";
}

void PSGPUWrapper::LoadIntoMemory(bool is_shuffle) {
  platform::Timer timer;
  VLOG(3) << "Begin LoadIntoMemory(), dataset[" << dataset_ << "]";
  timer.Start();
  dataset_->LoadIntoMemory();
  timer.Pause();
  VLOG(0) << "LoadIntoMemory cost: " << timer.ElapsedSec() << "s";

  if (dataset_->GetMemoryDataSize() > 0) {
    // local shuffle
    if (is_shuffle) {
      dataset_->LocalShuffle();
    }
    InitSlotInfo();
    std::shared_ptr<HeterContext> gpu_task = gpu_task_pool_.Get();
    gpu_task->Reset();
    gpu_task->pass_id_ = (uint16_t)(dataset_->GetPassID());

    dataset_mutex_.lock();
    dataset_pipe_.push(dataset_);
    dataset_mutex_.unlock();

    data_ready_channel_->Put(gpu_task);
  }
  
  VLOG(3) << "End LoadIntoMemory(), dataset[" << dataset_ << "]";
}

void PSGPUWrapper::start_build_thread() {
  running_ = true;
  VLOG(3) << "start build CPU ps thread.";
  pre_build_threads_ = std::thread([this] { pre_build_thread(); });
  // buildpull_threads_ = std::thread([this] { build_pull_thread(); });
}

void PSGPUWrapper::pre_build_thread() {
  // prebuild: process load_data
  while (running_) {
    std::shared_ptr<HeterContext> gpu_task = nullptr;
    if (!data_ready_channel_->Get(gpu_task)) {
      continue;
    }
    VLOG(3) << "thread PreBuildTask start.";
    platform::Timer timer;
    timer.Start();
    // build cpu ps data process
    PreBuildTask(gpu_task);
    timer.Pause();
    VLOG(0) << "thread PreBuildTask end, cost time: " << timer.ElapsedSec()
            << "s";
    buildcpu_ready_channel_->Put(gpu_task);
  }
  VLOG(3) << "build cpu thread end";
}
/*
void PSGPUWrapper::build_pull_thread() {
  while (running_) {
    std::shared_ptr<HeterContext> gpu_task = nullptr;
    if (!buildcpu_ready_channel_->Get(gpu_task)) {
      continue;
    }
    VLOG(3) << "thread build pull start.";
    platform::Timer timer;
    timer.Start();
    // build cpu ps data process
    BuildPull(gpu_task);
    timer.Pause();
    VLOG(1) << "thread BuildPull end, cost time: " << timer.ElapsedSec() << "s";
    buildpull_ready_channel_->Put(gpu_task);
  }
  VLOG(3) << "build cpu thread end";
}
*/

void PSGPUWrapper::build_task() {
  // build_task: build_pull + build_gputask
  std::shared_ptr<HeterContext> gpu_task = nullptr;
  // train end, gpu free
  if (!gpu_free_channel_->Get(gpu_task)) {
    return;
  }
  // ins and pre_build end
  if (!buildcpu_ready_channel_->Get(gpu_task)) {
    return;
  }

  VLOG(0) << "BuildPull start.";
  platform::Timer timer;
  timer.Start();
  BuildPull(gpu_task);
  BuildGPUTask(gpu_task);
  timer.Pause();
  VLOG(0) << " BuildGPUTask end, cost time: " << timer.ElapsedSec()
          << "s";

  current_task_ = gpu_task;
}

void PSGPUWrapper::BeginPass() {
  platform::Timer timer;
  timer.Start();
  if (current_task_) {
    PADDLE_THROW(
        platform::errors::Fatal("[BeginPass] current task is not ended."));
  }
  build_task();
  timer.Pause();

  if (current_task_ == nullptr) {
    PADDLE_THROW(platform::errors::Fatal(
        "[BeginPass] after build_task, current task is not null."));
  }

  VLOG(0) << "BeginPass end, cost time: " << timer.ElapsedSec() << "s";
}

void PSGPUWrapper::EndPass() {

  if (!current_task_) {
    PADDLE_THROW(
        platform::errors::Fatal("[EndPass] current task has been ended."));
  }

  platform::Timer timer;
  timer.Start();
  size_t keysize_max = 0;

  // in case of feasign_num = 0, skip dump_to_cpu
  if (!multi_mf_dim_) {
    for (size_t i = 0; i < heter_devices_.size(); i++) {
      keysize_max =
          std::max(keysize_max, current_task_->device_keys_[i].size());
    }
  } else {
    for (size_t i = 0; i < heter_devices_.size(); i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        keysize_max =
            std::max(keysize_max, current_task_->device_dim_keys_[i][j].size());
      }
    }
  }

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();

  int thread_num = 8; 
  auto dump_pool_to_cpu_func = [this, thread_num, &accessor_wrapper_ptr](int i, int j, int z) {

    PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(this->resource_->dev_id(i)));

    auto& hbm_pool = this->hbm_pools_[i * this->multi_mf_dim_ + j];
    auto& device_keys = this->current_task_->device_dim_keys_[i][j];
    size_t len = device_keys.size();
    // ============ add for multi-thread ================ 
    int len_per_thread = len / thread_num;
    int remain = len % thread_num;
    int left = -1, right = -1;
    
    int real_len = len_per_thread;
    if (z < remain) real_len++;
    
    if (z < remain) {
      left = z * (len_per_thread + 1);
      right = left + real_len;
    } else {
      left = remain * (len_per_thread + 1) + (z - remain) * len_per_thread;
      right = left + real_len;
    }
    // ============ add for multi-thread ================ 
    int mf_dim = this->index_dim_vec_[j];
    // size_t feature_value_size =
    //    TYPEALIGN(8, sizeof(FeatureValue) + ((mf_dim + 1) * sizeof(float)));
    size_t feature_value_size =
        accessor_wrapper_ptr->GetFeatureValueSize(mf_dim);

    VLOG(3) << "dump pool to cpu table: " << i << "with mf dim: " << mf_dim
            << " key_len :" << len
            << " feature_value_size:" << feature_value_size;
    char* test_build_values =
        (char*)malloc(feature_value_size * real_len);
    uint64_t offset = left * feature_value_size;

    cudaMemcpy(test_build_values, hbm_pool->mem() + offset,
               feature_value_size * real_len, cudaMemcpyDeviceToHost);
    CHECK(len == hbm_pool->capacity());
    uint64_t unuse_key = std::numeric_limits<uint64_t>::max();
    for (int i = left; i < right; ++i) {
      if (device_keys[i] == unuse_key) {
        continue;
      }
      size_t local_offset = (i - left) * feature_value_size;
      float* gpu_val = (float*)(test_build_values + local_offset);
      accessor_wrapper_ptr->DumpFill(gpu_val, cpu_accessor_, mf_dim, accessor_type_);
    }
    free(test_build_values);
  };
  
  if (multi_mf_dim_) {
    VLOG(0) << "dynamic mf dump pool: multi_mf_dim_: " << multi_mf_dim_;
    size_t device_num = heter_devices_.size();
    std::vector<std::thread> threads(device_num * multi_mf_dim_ * thread_num);
    for (size_t i = 0; i < device_num; i++) {
      for (int j = 0; j < multi_mf_dim_; j++) {
        for (int k = 0; k < thread_num; k++) {
          threads[(i + j * device_num) * thread_num + k] = std::thread(dump_pool_to_cpu_func, i, j, k);
        }
      }
    }
    for (std::thread& t : threads) {
      t.join();
    }

  }
  if (keysize_max != 0) {
    HeterPs_->end_pass();
  }

  for (size_t i = 0; i < hbm_pools_.size(); i++) {
    delete hbm_pools_[i];
  }

  gpu_task_pool_.Push(current_task_);
  current_task_ = nullptr;
  gpu_free_channel_->Put(current_task_);
  timer.Pause();
  VLOG(0) << "EndPass end, cost time: " << timer.ElapsedSec() << "s";
}

void PSGPUWrapper::PullSparse(const paddle::platform::Place& place,
                              const int table_id,
                              const std::vector<const uint64_t*>& keys,
                              const std::vector<float*>& values,
                              const std::vector<int64_t>& slot_lengths,
                              const int hidden_size) {
/*
  VLOG(3) << "Begine Gpu Ps PullSparse";
  platform::Timer all_timer;
  platform::Timer pull_gpups_timer;
  all_timer.Start();
  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  auto buf = memory::Alloc(place, total_length * sizeof(FeatureValue));
  FeatureValue* total_values_gpu = reinterpret_cast<FeatureValue*>(buf->ptr());
  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GpuPs now."));
  } else if (platform::is_gpu_place(place)) {
    VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
    int device_id = place.GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
    LoDTensor& total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys = reinterpret_cast<uint64_t*>(
        total_keys_tensor.mutable_data<int64_t>({total_length, 1}, place));

    // construct slot_level lod info
    auto slot_lengths_lod = slot_lengths;
    for (size_t i = 1; i < slot_lengths_lod.size(); i++) {
      slot_lengths_lod[i] += slot_lengths_lod[i - 1];
    }
    auto buf_key = memory::Alloc(place, keys.size() * sizeof(uint64_t*));
    auto buf_length =
        memory::Alloc(place, slot_lengths.size() * sizeof(int64_t));
    uint64_t** gpu_keys = reinterpret_cast<uint64_t**>(buf_key->ptr());
    int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
    cudaMemcpy(gpu_keys, keys.data(), keys.size() * sizeof(uint64_t*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_len, slot_lengths_lod.data(),
               slot_lengths.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    this->CopyKeys(place, gpu_keys, total_keys, gpu_len,
                   static_cast<int>(slot_lengths.size()),
                   static_cast<int>(total_length));
    VLOG(3) << "Begin call PullSparseGPU in GPUPS, dev: " << devid_2_index
            << " len: " << total_length;
    pull_gpups_timer.Start();
    HeterPs_->pull_sparse(devid_2_index, total_keys, total_values_gpu,
                          static_cast<int>(total_length));
    pull_gpups_timer.Pause();
    VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
            << "]";
    this->CopyForPull(place, gpu_keys, values, total_values_gpu, gpu_len,
                      static_cast<int>(slot_lengths.size()), hidden_size,
                      total_length);
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GpuPs: PullSparse Only Support CUDAPlace Now."));
  }
  all_timer.Pause();
  VLOG(3) << "GpuPs PullSparse total costs: " << all_timer.ElapsedSec()
          << " s, of which GPUPS costs: " << pull_gpups_timer.ElapsedSec()
          << " s";
  VLOG(3) << "End PullSparse";
*/
}

void PSGPUWrapper::PullSparse(const paddle::platform::Place& place,
                              const int table_id,
                              const std::vector<const uint64_t*>& keys,
                              const std::vector<float*>& values,
                              const std::vector<int64_t>& slot_lengths,
                              const std::vector<int>& slot_dim, // dimension for each slot
                              const int hidden_size) {
  VLOG(3) << "Begine Gpu Ps PullSparse";
  platform::Timer all_timer;
  platform::Timer pull_gpups_timer;
  all_timer.Start();

  size_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);

  size_t feature_value_size = 0;
  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  feature_value_size = accessor_wrapper_ptr->GetFeatureValueSize(max_mf_dim_);

  VLOG(3) << "PullSparse, total_length:" << total_length << " featurevalue size:" << feature_value_size;

  auto buf = memory::Alloc(place, total_length * feature_value_size);
  float* total_values_gpu = reinterpret_cast<float*>(buf->ptr());

  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GpuPs now."));
  } else if (platform::is_gpu_place(place)) {
    VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
    int device_id = place.GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
    LoDTensor& total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys = reinterpret_cast<uint64_t*>(
        total_keys_tensor.mutable_data<int64_t>({int64_t(total_length), 1}, place));
    // construct slot_level lod info
    auto slot_lengths_lod = slot_lengths;
    for (size_t i = 1; i < slot_lengths_lod.size(); i++) {
      slot_lengths_lod[i] += slot_lengths_lod[i - 1];
    }
    auto buf_key = memory::Alloc(place, keys.size() * sizeof(uint64_t*));
    auto buf_length =
        memory::Alloc(place, slot_lengths.size() * sizeof(int64_t));
    uint64_t** gpu_keys = reinterpret_cast<uint64_t**>(buf_key->ptr());
    int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
    cudaMemcpy(gpu_keys, keys.data(), keys.size() * sizeof(uint64_t*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_len, slot_lengths_lod.data(),
               slot_lengths.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    auto buf_dim =
        memory::Alloc(place, slot_dim.size() * sizeof(int));
    int* gpu_dim = reinterpret_cast<int*>(buf_dim->ptr());
    cudaMemcpy(gpu_dim, slot_dim.data(),
               slot_dim.size() * sizeof(int), cudaMemcpyHostToDevice);

    this->CopyKeys(place, gpu_keys, total_keys, gpu_len,
                   static_cast<int>(slot_lengths.size()),
                   static_cast<int>(total_length));

    VLOG(3) << "Begin call PullSparseGPU in GPUPS, dev: " << devid_2_index
            << " len: " << total_length;
    
    pull_gpups_timer.Start();
    HeterPs_->pull_sparse(devid_2_index, total_keys, total_values_gpu,
                          total_length);
    
    VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
            << "]";

    accessor_wrapper_ptr->CopyForPull(place,
                                      gpu_keys,
                                      values,
                                      total_values_gpu,
                                      gpu_len,
                                      static_cast<int>(slot_lengths.size()),
                                      hidden_size,
                                      total_length,
                                      gpu_dim,
                                      feature_value_size);


    pull_gpups_timer.Pause();

  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GpuPs: PullSparse Only Support CUDAPlace Now."));
  }
  all_timer.Pause();
  time_1 += all_timer.ElapsedSec();
  time_2 += pull_gpups_timer.ElapsedSec();
  VLOG(3) << "GpuPs PullSparse total costs: " << all_timer.ElapsedSec()
            << " s, of which pullsparse costs: " << pull_gpups_timer.ElapsedSec()
            << " s";
  VLOG(3) << "End PullSparse";

}

void PSGPUWrapper::PushSparseGrad(const paddle::platform::Place& place,
                                  const int table_id,
                                  const std::vector<const uint64_t*>& keys,
                                  const std::vector<const float*>& grad_values,
                                  const std::vector<int64_t>& slot_lengths,
                                  const int hidden_size, const int batch_size) {
  VLOG(3) << "Begin GPUPS PushSparseGrad";
  platform::Timer all_timer;
  platform::Timer push_gpups_timer;
  all_timer.Start();
  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t grad_value_size = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);

  auto buf = memory::Alloc(place, total_length * grad_value_size);
  VLOG(3) << "Push Sparse Max mf dimention: " << max_mf_dim_ << " grad_value_size: " << grad_value_size;

  float* total_grad_values_gpu =
      reinterpret_cast<float*>(buf->ptr());

  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GPUPS now."));
  } else if (platform::is_gpu_place(place)) {
    int device_id = place.GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
    LoDTensor& cached_total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys =
        reinterpret_cast<uint64_t*>(cached_total_keys_tensor.data<int64_t>());
    VLOG(3) << "Begin copy grad tensor to gpups struct";
    accessor_wrapper_ptr->CopyForPush(place,
                                      grad_values,
                                      total_grad_values_gpu,
                                      slot_lengths,
                                      total_length,
                                      batch_size,
                                      grad_value_size,
                                      slot_vector_,
                                      slot_mf_dim_vector_);

    VLOG(3) << "Begin call PushSparseGPU in GPUPS, dev: " << devid_2_index
            << " len: " << total_length;
    push_gpups_timer.Start();
    HeterPs_->push_sparse(devid_2_index, total_keys, total_grad_values_gpu,
                          static_cast<int>(total_length));
    push_gpups_timer.Pause();
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GPUPS: PushSparseGrad Only Support CUDAPlace Now."));
  }
  all_timer.Pause();
  time_3 += all_timer.ElapsedSec();
  time_4 += push_gpups_timer.ElapsedSec();
  VLOG(3) << "PushSparseGrad total cost: " << all_timer.ElapsedSec()
          << " s, of which GPUPS cost: " << push_gpups_timer.ElapsedSec()
          << " s";
  VLOG(3) << "End PushSparseGrad";
}


#ifdef PADDLE_WITH_PSLIB
void add_sparse_optimizer(
    std::unordered_map<std::string, float>& config,  // NOLINT
    const ::paddle::SparseCommonSGDRuleParameter& sgd_param,
    const std::string& prefix = "") {
  auto optimizer_name = sgd_param.name();
  if (optimizer_name == "naive") {
    config[prefix + "optimizer_type"] = 0;
    config[prefix + "learning_rate"] = sgd_param.naive().learning_rate();
    config[prefix + "initial_range"] = sgd_param.naive().initial_range();
    if (sgd_param.naive().weight_bounds_size() == 2) {
      config[prefix + "min_bound"] = sgd_param.naive().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.naive().weight_bounds()[1];
    }
  } else if (optimizer_name == "adagrad") {
    config[prefix + "optimizer_type"] = 1;
    config[prefix + "learning_rate"] = sgd_param.adagrad().learning_rate();
    config[prefix + "initial_range"] = sgd_param.adagrad().initial_range();
    config[prefix + "initial_g2sum"] = sgd_param.adagrad().initial_g2sum();
    if (sgd_param.adagrad().weight_bounds_size() == 2) {
      config[prefix + "min_bound"] = sgd_param.adagrad().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adagrad().weight_bounds()[1];
    }
  } else if (optimizer_name == "std_adagrad") {
    config[prefix + "optimizer_type"] = 2;
    config[prefix + "learning_rate"] = sgd_param.adagrad().learning_rate();
    config[prefix + "initial_range"] = sgd_param.adagrad().initial_range();
    config[prefix + "initial_g2sum"] = sgd_param.adagrad().initial_g2sum();
    if (sgd_param.adagrad().weight_bounds_size() == 2) {
      config[prefix + "min_bound"] = sgd_param.adagrad().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adagrad().weight_bounds()[1];
    }
  } else if (optimizer_name == "adam") {
    config[prefix + "optimizer_type"] = 3;
    config[prefix + "learning_rate"] = sgd_param.adam().learning_rate();
    config[prefix + "initial_range"] = sgd_param.adam().initial_range();
    if (sgd_param.adam().weight_bounds_size() == 2) {
      config[prefix + "min_bound"] = sgd_param.adam().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adam().weight_bounds()[1];
    }
  }
}

void PSGPUWrapper::InitializeGPUServer(const std::string& fleet_desc) {
  // optimizer config for hbmps
  google::protobuf::TextFormat::ParseFromString(fleet_desc, &_ps_param);
  auto sparse_table =
      _ps_param.server_param().downpour_server_param().downpour_table_param(0);
  auto sparse_table_accessor = sparse_table.accessor();
  auto sparse_table_accessor_parameter =
      sparse_table_accessor.downpour_accessor_param();
  auto accessor_class = sparse_table_accessor.accessor_class();

  // NOTE(zhangminxu): gpups' sparse table optimizer config,
  // now only support single sparse table
  // auto sparse_table = param_.sparse_table(0);
  std::unordered_map<std::string, float> config;
  if (accessor_class == "DownpourFeatureValueAccessor" ||
      accessor_class == "DownpourCtrAccessor" ||
      accessor_class == "DownpourCtrDoubleAccessor") {

    config["nonclk_coeff"] = sparse_table_accessor_parameter.nonclk_coeff();
    config["clk_coeff"] = sparse_table_accessor_parameter.click_coeff();
    config["learning_rate"] =
        sparse_table_accessor.sparse_sgd_param().learning_rate();
    config["initial_g2sum"] =
        sparse_table_accessor.sparse_sgd_param().initial_g2sum();
    config["initial_range"] =
        sparse_table_accessor.sparse_sgd_param().initial_range();
    if (sparse_table_accessor.sparse_sgd_param().weight_bounds_size() == 2) {
      config["min_bound"] =
          sparse_table_accessor.sparse_sgd_param().weight_bounds()[0];
      config["max_bound"] =
          sparse_table_accessor.sparse_sgd_param().weight_bounds()[1];

    }
    // NOTE(zhangminxu): for DownpourCtrAccessor & DownpourCtrDoubleAccessor,
    // optimizer config for embed_w & embedx_w is the same
    config["mf_create_thresholds"] = sparse_table_accessor.embedx_threshold(); // default = 10
    config["mf_learning_rate"] = config["learning_rate"];
    config["mf_initial_g2sum"] = config["initial_g2sum"];
    config["mf_initial_range"] = config["initial_range"];
    config["mf_min_bound"] = config["min_bound"];
    config["mf_max_bound"] = config["max_bound"];
    config["mf_embedx_dim"] = sparse_table_accessor.embedx_dim(); // default = 8

  } else if (accessor_class == "DownpourSparseValueAccessor") {
    auto optimizer_name = sparse_table_accessor.sparse_commonsgd_param().name();
    if (optimizer_name == "naive") {
      config["learning_rate"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .naive()
                                    .learning_rate();
      config["initial_range"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .naive()
                                    .initial_range();
      if (sparse_table_accessor.sparse_commonsgd_param()
              .naive()
              .weight_bounds_size() == 2) {
        config["min_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .naive()
                                  .weight_bounds()[0];
        config["max_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .naive()
                                  .weight_bounds()[1];
      }
    } else if (optimizer_name == "adagrad") {
      config["learning_rate"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .adagrad()
                                    .learning_rate();
      config["initial_range"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .adagrad()
                                    .initial_range();
      config["initial_g2sum"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .adagrad()
                                    .initial_g2sum();
      if (sparse_table_accessor.sparse_commonsgd_param()
              .adagrad()
              .weight_bounds_size() == 2) {
        config["min_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .adagrad()
                                  .weight_bounds()[0];
        config["max_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .adagrad()
                                  .weight_bounds()[1];
      }
    } else if (optimizer_name == "adam") {
      config["learning_rate"] =
          sparse_table_accessor.sparse_commonsgd_param().adam().learning_rate();
      config["initial_range"] =
          sparse_table_accessor.sparse_commonsgd_param().adam().initial_range();
      if (sparse_table_accessor.sparse_commonsgd_param()
              .adam()
              .weight_bounds_size() == 2) {
        config["min_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .adam()
                                  .weight_bounds()[0];
        config["max_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                  .adam()
                                  .weight_bounds()[1];
      }
    }
  } else if (accessor_class == "DownpourUnitAccessor" ||
             accessor_class == "DownpourDoubleUnitAccessor" ||
             accessor_class == "DownpourCtrDymfAccessor" || 
             accessor_class == "DownpourCtrDoubleDymfAccessor") {
    config["nonclk_coeff"] = sparse_table_accessor_parameter.nonclk_coeff();
    config["clk_coeff"] = sparse_table_accessor_parameter.click_coeff();
    config["mf_create_thresholds"] = sparse_table_accessor.embedx_threshold();
    // optimizer config for embed_w and embedx
    add_sparse_optimizer(config, sparse_table_accessor.embed_sgd_param());
    add_sparse_optimizer(
        config, sparse_table_accessor.embedx_sgd_param(), "mf_");
    config["mf_embedx_dim"] = sparse_table_accessor.embedx_dim(); // default = 8
  }
  config["sparse_shard_num"] = sparse_table.shard_num();

  GlobalAccessorFactory::GetInstance().Init(accessor_class);

  GlobalAccessorFactory::GetInstance().GetAccessorWrapper()->Configure(
        config);
  
  InitializeGPUServer(config);
  SetCPUAccessorType(accessor_class);
}
#endif

}  // end namespace framework
}  // end namespace paddle
#endif
