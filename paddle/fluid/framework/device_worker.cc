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

#include "paddle/fluid/framework/device_worker.h"

#include "paddle/fluid/framework/convert_utils.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {

class Scope;

void DeviceWorker::SetRootScope(Scope* root_scope) { root_scope_ = root_scope; }

void DeviceWorker::SetDataFeed(DataFeed* data_feed) {
  device_reader_ = data_feed;
}

template <typename T>
std::string PrintLodTensorType(Tensor* tensor, int64_t start, int64_t end) {
  auto count = tensor->numel();
  if (start < 0 || end > count) {
    VLOG(3) << "access violation";
    return "access violation";
  }
  // std::ostringstream os;
  std::string os;
  for (int64_t i = start; i < end; i++) {
    // os << ":" << tensor->data<T>()[i];
    // os.append(":").append(std::to_string(tensor->data<T>()[i]));
    if (i != start) os.append(" ");
    os.append(std::to_string(tensor->data<T>()[i]));
  }
  // return os.str();
  return os;
}

std::string PrintLodTensorIntType(Tensor* tensor, int64_t start, int64_t end) {
  auto count = tensor->numel();
  if (start < 0 || end > count) {
    VLOG(3) << "access violation";
    return "access violation";
  }

  // std::ostringstream os;
  std::string os;
  for (int64_t i = start; i < end; i++) {
    // os << ":" << static_cast<uint64_t>(tensor->data<int64_t>()[i]);
    if (i != start) os.append(" ");
    os.append(std::to_string(static_cast<uint64_t>(tensor->data<int64_t>()[i])));
  }
  //return os.str();
  return os;
}

std::string PrintLodTensor(Tensor* tensor, int64_t start, int64_t end) {
  std::string out_val;
  if (framework::TransToProtoVarType(tensor->dtype()) == proto::VarType::FP32) {
    out_val = PrintLodTensorType<float>(tensor, start, end);
  } else if (framework::TransToProtoVarType(tensor->dtype()) ==
             proto::VarType::INT64) {
    out_val = PrintLodTensorIntType(tensor, start, end);
  } else if (framework::TransToProtoVarType(tensor->dtype()) ==
             proto::VarType::FP64) {
    out_val = PrintLodTensorType<double>(tensor, start, end);
  } else {
    out_val = "unsupported type";
  }
  return out_val;
}

std::pair<int64_t, int64_t> GetTensorBound(LoDTensor* tensor, int index) {
  auto& dims = tensor->dims();
  if (tensor->lod().size() != 0) {
    auto& lod = tensor->lod()[0];
    return {lod[index] * dims[1], lod[index + 1] * dims[1]};
  } else {
    return {index * dims[1], (index + 1) * dims[1]};
  }
}

bool CheckValidOutput(LoDTensor* tensor, size_t batch_size) {
  auto& dims = tensor->dims();
  if (dims.size() != 2) return false;
  if (tensor->lod().size() != 0) {
    auto& lod = tensor->lod()[0];
    if (lod.size() != batch_size + 1) {
      return false;
    }
  } else {
    if (dims[0] != static_cast<int>(batch_size)) {
      return false;
    }
  }
  return true;
}

void DeviceWorker::DumpParam(const Scope& scope, const int batch_id) {
  std::ostringstream os;
  for (auto& param : *dump_param_) {
    os.str("");
    Variable* var = scope.FindVar(param);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    framework::LoDTensor cpu_tensor;
    if (platform::is_gpu_place(tensor->place())) {
      TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
      tensor = &cpu_tensor;
    }
    int64_t len = tensor->numel();
    os << "(" << batch_id << "," << param << ")"
       << PrintLodTensor(tensor, 0, len);
    writer_ << os.str();
  }
}

void DeviceWorker::InitRandomDumpConfig(const TrainerDesc& desc) {
  bool enable_random_dump = desc.enable_random_dump();
  if (!enable_random_dump) {
    dump_mode_ = 0;
  } else {
    if (desc.random_with_lineid()) {
      dump_mode_ = 1;
    } else {
      dump_mode_ = 2;
    }
  }
  dump_interval_ = desc.dump_interval();
}

void DeviceWorker::DumpField(const Scope& scope, int dump_mode,
                             int dump_interval) {  // dump_mode: 0: no random,
                                                   // 1: random with insid hash,
                                                   // 2: random with random
                                                   // number

  size_t batch_size = device_reader_->GetCurBatchSize();
  auto& ins_id_vec = device_reader_->GetInsIdVec();
  auto& ins_content_vec = device_reader_->GetInsContentVec();
  if (ins_id_vec.size() > 0) {
    batch_size = ins_id_vec.size();
  }
  std::vector<std::string> ars(batch_size);
  // std::vector<bool> hit(batch_size, false);

  std::default_random_engine engine(0);
  std::uniform_int_distribution<size_t> dist(0U, INT_MAX);
  
  std::vector<std::thread> dump_field_threads;

  size_t tensor_copy_thread_num = dump_fields_->size();
  std::vector<std::shared_ptr<LoDTensor>> cpu_tensors(tensor_copy_thread_num, nullptr);

  auto tensor_copy_logic = [this, batch_size, &scope, &cpu_tensors](size_t tid) {

    auto& field = (*dump_fields_)[tid];

    Variable* var = scope.FindVar(field);
    if (var == nullptr) {
      VLOG(0) << "Note: field[" << field
              << "] cannot be find in scope, so it was skipped.";
      return;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    if (!tensor->IsInitialized()) {
      VLOG(0) << "Note: field[" << field
              << "] is not initialized, so it was skipped.";
      return;
    }

    cpu_tensors[tid] = std::make_shared<LoDTensor>();
    framework::LoDTensor& cpu_tensor = *(cpu_tensors[tid].get());

    if (platform::is_gpu_place(tensor->place())) {
      TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
      cpu_tensor.set_lod(tensor->lod());
      tensor = &cpu_tensor;
    }
    if (!CheckValidOutput(tensor, batch_size)) {
      VLOG(0) << "Note: field[" << field << "] cannot pass check, so it was "
                                            "skipped. Maybe the dimension is "
                                            "wrong ";
      return;
    }
     // cpu_tensors[tid] = tensor;

  };

   dump_field_threads.clear();
   dump_field_threads.resize(tensor_copy_thread_num);
  
   for (size_t i = 0; i < tensor_copy_thread_num; i++) {
       dump_field_threads[i] = std::thread(tensor_copy_logic, i);
   }
   for (size_t i = 0; i < tensor_copy_thread_num; i++) {
       dump_field_threads[i].join();
   }

  size_t fill_thread_num = 32;
  auto fill_hit = [this, dump_interval, dump_mode, batch_size,
                   &ins_id_vec, &ins_content_vec, &ars, &dist, &engine, &cpu_tensors](size_t begin, size_t end) {
   
    for (size_t i = begin; i < end; i++) {
      size_t r = 0;
      if (dump_mode == 1) {
        r = XXH64(ins_id_vec[i].data(), ins_id_vec[i].length(), 0);
      } else if (dump_mode == 2) {
        r = dist(engine);
      }
      if (r % dump_interval != 0) {
        continue;
      }
      // hit[i] = true;
      // ars[i] += ins_id_vec[i];
      if (ins_id_vec.size() > 0) {
        ars[i].append(ins_id_vec[i]);
      }
      if (ins_content_vec.size() > 0) {
        ars[i].append("\t");
        ars[i].append(ins_content_vec[i]);
      }    

      // int j = 0;
      for (auto& tensor : cpu_tensors) {
        if (tensor == nullptr) {
          // j++;
          continue;
        }
        // auto& field = (*dump_fields_)[j];
      
        auto bound = GetTensorBound(tensor.get(), i);
        // ars[i] = ars[i] + "\t" + field + ":" +
        //         std::to_string(bound.second - bound.first);
        //ars[i] += PrintLodTensor(tensor.get(), bound.first, bound.second);
        // ars[i].append("\t").append(field).append(":").append(
        //          std::to_string(bound.second - bound.first));
        ars[i].append("\t");
        ars[i].append(PrintLodTensor(tensor.get(), bound.first, bound.second));
        // j++;
      }
      ars[i].append("\n");
    }

  };

  dump_field_threads.clear();
  dump_field_threads.resize(fill_thread_num); 
  size_t avg_ins = batch_size / fill_thread_num;
  size_t left_ins = batch_size % fill_thread_num;
  size_t begin, end;

  for (size_t i = 0; i < fill_thread_num; i++) {
    if (i < left_ins) {
      begin = (avg_ins + 1) * i;
      end = begin + avg_ins + 1;
    } else { // i >= left_ins
      begin = (avg_ins + 1) * left_ins + (i - left_ins) * avg_ins;
      end = begin + avg_ins;
    }
    dump_field_threads[i] = std::thread(fill_hit, begin, end);
  }

  for (size_t i = 0; i < fill_thread_num; i++) {
    dump_field_threads[i].join();
  }
  dump_field_threads.clear();

  // #pragma omp parallel for
  for (size_t i = 0; i < ars.size(); i++) {
    if (ars[i].length() == 0) {
      continue;
    }
    writer_ << ars[i];
  }
  // writer_.Flush();
}

}  // namespace framework
}  // namespace paddle
