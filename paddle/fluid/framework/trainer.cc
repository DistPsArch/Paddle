/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "io/fs.h"

namespace paddle {
namespace framework {

class MemRegion {
        public:
            MemRegion() {
                _cap = 200 * 1024 * 1024; // 200 MB
                _buf = (char*)malloc(_cap);
                _cur = 0;
                _file_idx = -1;
            }
            virtual ~MemRegion() {
                free(_buf);
            }
            bool buff_remain(int len) {
                if (_cap - _cur < len) {
                    return false;
                } else {
                    return true;
                }
            }
            char* acquire(int len) {
                if (_cap - _cur < len) {
                    return nullptr;
                } else {
                    char* ret =  _buf + _cur;
                    _cur += len;
                    return ret;
                }
            }
            void reset() {
                _cur = 0;
                _file_idx = -1;
            }
            int _cap;
            int _cur;
            int _file_idx;
            char* _buf;
};

void TrainerBase::SetScope(Scope* root_scope) { root_scope_ = root_scope; }

void TrainerBase::ParseDumpConfig(const TrainerDesc& desc) {
  dump_fields_path_ = desc.dump_fields_path();
  need_dump_field_ = false;
  need_dump_param_ = false;
  if (dump_fields_path_ == "") {
    VLOG(2) << "dump_fields_path_ is empty";
    return;
  }
  auto& file_list = dataset_ptr_->GetFileList();
  if (file_list.size() == 0) {
    VLOG(2) << "file_list is empty";
    return;
  }

  dump_converter_ = desc.dump_converter();
  if (desc.dump_fields_size() != 0) {
    need_dump_field_ = true;
    dump_fields_.resize(desc.dump_fields_size());
    for (int i = 0; i < desc.dump_fields_size(); ++i) {
      dump_fields_[i] = desc.dump_fields(i);
    }
  }

  if (desc.dump_param_size() != 0) {
    need_dump_param_ = true;
    dump_param_.resize(desc.dump_param_size());
    for (int i = 0; i < desc.dump_param_size(); ++i) {
      dump_param_[i] = desc.dump_param(i);
    }
  }
}

void TrainerBase::DumpWork(int tid) {
#ifdef _LINUX
  int err_no = 0;
  // GetDumpPath is implemented in each Trainer
  std::string path = GetDumpPath(tid);
  auto ps_gpu_ptr = PSGPUWrapper::GetInstance();
  if (ps_gpu_ptr->UseAfsApi()) {
    auto afs_writer = ps_gpu_ptr->OpenWriter(path);
    // read batch from  queue_
    std::string out_str;
    std::shared_ptr<MemRegion> region = std::make_shared<MemRegion>();
    ChannelReader<std::string> reader(queue_.get());
    while (reader >> out_str) {
      int len = out_str.length();
      if (!region->buff_remain(len)) {
        // if (0 != afs_writer->write(out_str.data(), out_str.length(), true)) {
        if (0 != afs_writer->write(region->_buf, region->_cur, true)) {
          VLOG(0) << "Dump Work save failed!!!";
        }
        region->reset();
      }
      char* buf = region->acquire(len);
      // for safety, CHECK(buf)
      memcpy(buf, out_str.data(), len);
    }
    // write left str
    if (region->_cur) {
        if (0 != afs_writer->write(region->_buf, region->_cur, true)) {
          VLOG(0) << "Dump Work save failed!!!";
        }
        region->reset();
    } 
  } else {
    std::shared_ptr<FILE> fp = fs_open_write(path, &err_no, dump_converter_);
    // while (1) {
     std::string out_str;
     ChannelReader<std::string> reader(queue_.get());
     while (reader >> out_str) {
      // if (!queue_->Get(out_str)) {
      //  break;
      //}
      size_t write_count =
          fwrite_unlocked(out_str.data(), 1, out_str.length(), fp.get());
      if (write_count != out_str.length()) {
        VLOG(3) << "dump text failed";
        continue;
      }
      write_count = fwrite_unlocked("\n", 1, 1, fp.get());
      if (write_count != 1) {
        VLOG(3) << "dump text failed";
        continue;
      }
     }
    // }
  }
#endif
}

void TrainerBase::FinalizeDumpEnv() {
  queue_->Close();
  for (auto& th : dump_thread_) {
    th.join();
  }
  queue_.reset();
}

}  // end namespace framework
}  // end namespace paddle
