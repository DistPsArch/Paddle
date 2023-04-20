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
#include <vector>
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_ps_base.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

template <typename GPUAccessor, template<typename T> class GPUOptimizer>
class HeterPs : public HeterPsBase {
 public:
  HeterPs() {}
  HeterPs(size_t capacity, std::shared_ptr<HeterPsResource> resource, GPUAccessor& gpu_accessor);
  virtual ~HeterPs();
  HeterPs(const HeterPs&) = delete;
  HeterPs& operator=(const HeterPs&) = delete;

  virtual void pull_sparse(int num, FeatureKey* d_keys, float* d_vals,
                           size_t len) override;
  // virtual void build_ps(int num, FeatureKey* h_keys, FeatureValue* h_vals,
  //                      size_t len, size_t chunk_size, int stream_num) override;
  virtual void build_ps(int num, FeatureKey* h_keys, char* pool,
                        size_t len, size_t feature_value_size, size_t chunk_size, int stream_num) override;
  virtual void set_nccl_comm_and_size(
      const std::vector<ncclComm_t>& inner_comms,
      const std::vector<ncclComm_t>& inter_comms, int comm_size) override;
  virtual void set_multi_mf_dim(int multi_mf_dim, int max_mf_dim) override;
  virtual void end_pass() override;
  virtual int get_index_by_devid(int devid) override;
  virtual void show_one_table(int gpu_num) override;
  virtual void push_sparse(int num, FeatureKey* d_keys,
                           float* d_grads, size_t len) override;

  void set_sparse_sgd(const OptimizerConfig& optimizer_config) override;
  void set_embedx_sgd(const OptimizerConfig& optimizer_config) override;

 private:
  std::shared_ptr<HeterComm<FeatureKey, float, float, GPUAccessor>> comm_;
  GPUOptimizer<GPUAccessor> opt_;
};

}  // end namespace framework
}  // end namespace paddle
#endif
