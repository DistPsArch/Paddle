/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <cmath>
#include <curand_kernel.h>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include <cooperative_groups.h>
#define WARP_SIZE 32

#if defined(PADDLE_WITH_CUDA)
namespace cg = cooperative_groups;

// check
#define CUDA_KERNEL_LOOP(i, n)                                  \
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void CopyKernel(T *output_values, const T *input_values, const int out_row, const int in_row, const int cols) {
    CUDA_KERNEL_LOOP(i, out_row * cols) {
      int out_row_id = i / cols;
      int col_id = i % cols;
      int in_row_id = out_row_id % in_row;
      output_values[i] = input_values[in_row_id * cols + col_id];
    }
}

template <typename T>
__global__ void CopyKernel2(T *output_values, const T *input_values, int64_t* random_values, const int out_row, const int cols) {
    CUDA_KERNEL_LOOP(i, out_row * cols) {
      int out_row_id = i / cols;
      int col_id = i % cols;
      int in_row_id = random_values[out_row_id];
      output_values[i] = input_values[in_row_id * cols + col_id];
    }
}


__global__ void InitializeKernel(int64_t* random_rematch_data, int64_t* random_label_target_data, const int64_t* label_data, const int input_rows, const int random_rematch_ratio) {
    CUDA_KERNEL_LOOP(i, input_rows * random_rematch_ratio) {
        int rematch_id = i / input_rows;
        int row_id = i % input_rows;
        if (rematch_id == 0) {
            random_rematch_data[row_id] = row_id;
            random_label_target_data[row_id] = label_data[row_id];
        } else {
            random_rematch_data[i] = 0;
            random_label_target_data[i] = 0;
        }
    }
}

// use global_random_rematch
template <typename T>
__global__ void InitializeKernel2(T *random_rematch_data, T *random_label_target_data, T* tmp_v, const int input_rows) {
    CUDA_KERNEL_LOOP(i, input_rows) {
        random_rematch_data[i] = tmp_v[i];
        random_label_target_data[i] = 0;
    }
}

/*
void custom_random_shuffle(std::vector<int>& ret) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);
    const int n = ret.size();
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    std::vector<bool> visit(n, false);
    while (!v.empty()) {
        std::shuffle(v.begin(), v.end(), rng);
        int tmp = v.back();
        v.pop_back();
        if (v.empty()) {
            std::uniform_int_distribution<int> distr(0, n-2);
            ret[tmp] = tmp;
            std::swap(ret[tmp], ret[(distr(rng) + tmp + 1) % n]);
            return;
        }
        visit[tmp] = true;
        std::shuffle(v.begin(), v.end(), rng);
        int curr = v.back();
        v.pop_back();
        v.push_back(tmp);
        ret[tmp] = curr;
        while (!visit[curr]) {
            visit[curr] = true;
            std::shuffle(v.begin(), v.end(), rng);
            ret[curr] = v.back();
            v.pop_back();
            curr = ret[curr];
        }
    }
}
*/

/*
// self_in_val: news_u2_output [ins_num, 64]
// other_in_val: news_u1_output [ins_num, 64]
// Eigen::MatrixXf vec_sim_mat(ori_ins_num, ori_ins_num);
// _calc_vector_similarity(self_in_val, other_in_val, vec_sim_mat);

void _calc_vector_similarity(
        const Eigen::MatrixXf& self_in_val,
        const Eigen::MatrixXf& other_in_val,
        Eigen::MatrixXf& vec_sim_mat) {
    
    int ins_num = self_in_val.rows();

    //先全体平方，然后每行做sum,然后开方，最后得到的shape [ins_num, 1]
    Eigen::MatrixXf a_norm = other_in_val.array().square().rowwise().sum().array().sqrt();
    Eigen::MatrixXf b_norm = self_in_val.array().square().rowwise().sum().array().sqrt();

    for (int i = 0; i < ins_num; ++i) {
        for (int j = 0; j < ins_num; ++j) {
            CHECK(a_norm(i, 0) >= 1e-5 && b_norm(i, 0) >= 1e-5);
            // 下面是计算news_u1_output 对 news_u2_output的相似度矩阵
            // 的计算公式
            vec_sim_mat(i, j) = other_in_val.row(i).dot(self_in_val.row(j))
                    / (a_norm(i, 0) * b_norm(j, 0));
            //LOG(INFO) << "vec_other(" << i << ",0)=" << other_in_val(i, 0)
            //          << ",vec_self(" << j << ",0)=" << self_in_val(j, 0)
            //          << ",vec_sim_mat(" << i << "," << j << ")=" << vec_sim_mat(i, j);

        }
    }
}

// shape（ins_num, 2)
// const Eigen::MatrixXf& weighted_sample_feature = _feature_input->value();
// fea_num个Matrix,每个都是(ins_num, ins_num)1
// std::vector<Eigen::MatrixXf> feature_match_mat(fea_num,
//        Eigen::MatrixXf(ori_ins_num, ori_ins_num));
// _calc_feature_match(weighted_sample_feature, feature_match_mat);
// 这个feature_match_mat矩阵就是统计在每个特征上，样本与样本间在这个特征上的值是否相同
// 
void _calc_feature_match(
        const Eigen::MatrixXf& weighted_sample_feature,
        std::vector<Eigen::MatrixXf>& feature_match_mat) {

    int ins_num = weighted_sample_feature.rows();
    int fea_num = weighted_sample_feature.cols();
   
    // 比如对于 i = 0, j = 0， 也就是第一个ins, 第一个特征
    //  
    for (int i = 0; i < fea_num; ++i) {
        // 对于第j个ins,从他自己开始k = j，如果weighted_sample_feature(k, i) == weighted_sample_feature(j,i)的话
        // 也就是对于每个ins,从他自己往下数，如果有某个ins的第i个feature的值和他一样的话
        // 那么，在第i个特征上，j,k和k,j的值都是1.0,否则为0.0
        for (int j = 0; j < ins_num; ++j) {

            for (int k = j; k < ins_num; ++k) {

                if (weighted_sample_feature(j, i)
                        == weighted_sample_feature(k, i)) {
                    // 那么，在第i个特征上，j,k和k,j的值都是1.0,否则为0.0
                    feature_match_mat[i](j, k) = 1.0;
                    feature_match_mat[i](k, j) = 1.0;
                } else {
                    feature_match_mat[i](j, k) = 0.0;
                    feature_match_mat[i](k, j) = 0.0;
                }
                //LOG(INFO) << i << "th feature_match_mat(" << j << "," << k << ")="
                //    << feature_match_mat[i](j, k);

            }
        }
    }

}

// Eigen::MatrixXf vec_sim_mat(ori_ins_num, ori_ins_num);
// std::vector<Eigen::MatrixXf> feature_match_mat(fea_num,
//        Eigen::MatrixXf(ori_ins_num, ori_ins_num));
// Eigen::MatrixXf sample_weight_mat(ori_ins_num, ori_ins_num);
// _calc_sample_weight(vec_sim_mat, feature_match_mat, sample_weight_mat);
void _calc_sample_weight(
        const float vec_sim_max,
        const float vec_sim_base,
        const float fea_match_base, 
        const Eigen::MatrixXf& vec_sim_mat,
        const std::vector<Eigen::MatrixXf>& feature_match_mat,
        Eigen::MatrixXf& sample_weight_mat) {

    int ins_num = vec_sim_mat.rows();
    int fea_num = feature_match_mat.size();

    for (int i = 0; i < ins_num; ++i) {
        for (int j = 0; j < ins_num; ++j) {
            // 这是cosine相似度吗
            double cosine = vec_sim_mat(i, j);
            // 自己不作为负样本
            // 这个sample_weight是采样负样本的权重呀
            // 也就是这个对作为一个负样本的权重
            if (i == j) {
                sample_weight_mat(i, j) = 0.0;
                continue;
            }
            // 与自己距离过近的点以很小概率作为负样本
            // 两个样本的cosine相似度如果很大的话，也就是很相似的两个样本
            // 以很小的概率作为负样本
            if (cosine > vec_sim_max) {
                cosine = -cosine;
            }
            double w = 1.0;
            // 在安全距离外，距离越近负采样概率越大
            // 1000.0的cosine次方
            w *= std::pow(vec_sim_base, cosine);
            // 增加feature与自身不match的点的负采样概率
            // 在一跳孪生网络中，_fea_match_base都是1.0,所以下面的逻辑实际上都是乘以1
            for (int k = 0; k < fea_num; ++k) {
                w *= std::pow(fea_match_base, 1.0 - feature_match_mat[k](i, j));
            }
            // 原来这个sample_weight_mat二维矩阵的意义是<i,j>样本对作为负样本的概率，自身<i,i>不作为负样本
            sample_weight_mat(i, j) = w;
        }
    }
}

// Eigen::MatrixXf sample_weight_mat(ori_ins_num, ori_ins_num);
// 这个函数是计算样本对<i,j>作为负样本的概率
// rematch_ratio - 1 = 3
// random_val: [ins_num * 4, 1], 前面前面ins_num行，每行的值为ins的id, 后面每行都是0
// _do_weighted_rematch(sample_weight_mat, rematch_ratio - 1, random_val);

void _do_weighted_rematch(
        const Eigen::MatrixXf& sample_weight_mat,
        int sample_num,
        Eigen::MatrixXf& random_val) {

    //CHECK(sample_weight_mat.rows() == sample_weight_mat.cols());
    int ins_num = sample_weight_mat.rows();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < ins_num; ++i) {

        const Eigen::VectorXf& sample_weight = sample_weight_mat.row(i); // 拿出一行
        std::vector<double> cdf(ins_num, sample_weight(0)); // cdf是一个vector，ins_num个，每个的初始值是<i, 0>做负样本的概率
        for (int j = 1; j < ins_num; ++j) {
            //LOG(INFO) << "sample_weight_mat(" << i << "," << j << ")=" << sample_weight(j);
            cdf[j] = cdf[j - 1] + sample_weight(j); // cdf[j] = cdf[j-1] + <i,j> 做负样本的概率,是个累加的过程
        }

        // 不知道这个cdf的意义在哪里
        double sum = cdf[ins_num - 1];
        CHECK(sum >= 1e-5);
        // 每个值又除了sum
        //std::stringstream ss;
        for (int j = 0; j < ins_num; ++j) {
            cdf[j] /= sum;
            //ss << "|" << cdf[j];
        }

        //LOG(INFO) << "[" << i << "]CDF" << ss.str();
        for (int j = 0; j < sample_num; ++j) {
            int pos = ins_num - 1;
            double rand_num = dis(gen);  // uniform random 0~1
            //LOG(INFO) << "rand_num=" << rand_num;
            for (int k = 0; k < ins_num; ++k) {
                if (cdf[k] > rand_num) {
                    pos = k;
                    break;
                }
            }
            random_val(i + (j + 1) * ins_num, 0) = pos;
            //LOG(INFO) << "neg_sample_pos for [" << i << "] is " << pos;
        }


    }
}
*/

template <typename T>
__global__ void CopyKernel3(T* random_val, T* random_label_target_val, const int size) {
    CUDA_KERNEL_LOOP(i, size) {
      random_val[i] = 0;
      random_label_target_val[i] = 1;
    }
}

/*
template <typename T>
__global__ void CalcVectorSimilarityKernel(
        const int ins_num,
        const int cols,
        const T* self_in_val,
        const T* other_in_val,
        T* vec_sim_mat) {
    CUDA_KERNEL_LOOP(z, ins_num * ins_num) {
        int i = z / ins_num;
        int j = z % ins_num;
        // a_norm, b_norm
        float a_norm_i = 0.0;
        float b_norm_j = 0.0;
        vec_sim_mat[z] = 0.0;
        for (int k = 0; k < cols; k++) {
            a_norm_i += other_in_val[i * cols + k] * other_in_val[i* cols + k];
            b_norm_j += self_in_val[j * cols + k] * self_in_val[j * cols + k];
            vec_sim_mat[z] += other_in_val[i * cols + k] * self_in_val[j * cols + k];
        }
        a_norm_i = sqrt(a_norm_i);
        b_norm_j = sqrt(b_norm_j);
        // CHECK(a_norm_i >= 1e-5 && b_norm_j >= 1e-5);
        if (!(a_norm_i >= 1e-5 && b_norm_j >= 1e-5)) {
            printf("a_norm_i or b_norm_j not >= 1e-5\n");
        }
        vec_sim_mat[z] /= (a_norm_i * b_norm_j);
    }
}
*/

template <typename T>
__global__ void CalcVectorSimilarityKernel(
        const int ins_num,
        const int cols,
        const T* self_in_val,
        const T* other_in_val,
        T* vec_sim_mat) {
      
        // (ins_num * ins_num - 1) / 32 + 1 
        // const size_t z = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t q = blockIdx.x * 32 + threadIdx.x / WARP_SIZE;
        if (4 * q >= ins_num * ins_num) return;

        cg::thread_block b = cg::this_thread_block();
        cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);
        const size_t k = g.thread_rank();

        int col_per_thread = cols / WARP_SIZE;
        int remain = cols % WARP_SIZE;

        int col_size = col_per_thread;
        if (k < remain) col_size++;
  
        int left = -1, right = -1;
        if (k < remain) {
          left = k * (col_per_thread + 1);
          right = left + col_size;
        } else {
          left = remain * (col_per_thread + 1) + (k - remain) * col_per_thread;
          right = left + col_size;
        }

        for (int z = 4 * q; (z < 4 * q + 4) && (z < ins_num * ins_num); z++) {

          const size_t i = z / ins_num;
          const size_t j = z % ins_num;
          // const size_t k = threadIdx.y;

          // cg::thread_block b = cg::this_thread_block();
          // cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);
          // const size_t k = g.thread_rank();

          // a_norm, b_norm
          float a_norm_i = 0.0;
          float b_norm_j = 0.0;
          vec_sim_mat[z] = 0.0;
   
          // int col_per_thread = cols / WARP_SIZE;
          // int remain = cols % WARP_SIZE;

          // int col_size = col_per_thread;
          // if (k < remain) col_size++;
  
          // int left = -1, right = -1;
          // if (k < remain) {
          //  left = k * (col_per_thread + 1);
          //  right = left + col_size;
          // } else {
          //  left = remain * (col_per_thread + 1) + (k - remain) * col_per_thread;
          //  right = left + col_size;
          // }

          float local_a_norm_i_sum = 0.0;
          float local_b_norm_j_sum = 0.0;
          float local_vec_sim_sum = 0.0; 

          for (int t = left; t < right; t++) {
            local_a_norm_i_sum += other_in_val[i * cols + t] * other_in_val[i * cols + t];
            local_b_norm_j_sum += self_in_val[j * cols + t] * self_in_val[j * cols + t];
            local_vec_sim_sum += other_in_val[i * cols + t] * self_in_val[j * cols + t];
          }

          // reduce among threads within warp
          // shfl_down
          for (int p = 1; p < WARP_SIZE; p *= 2) {
            local_a_norm_i_sum += g.shfl_down(local_a_norm_i_sum, p);
            local_b_norm_j_sum += g.shfl_down(local_b_norm_j_sum, p);
            local_vec_sim_sum += g.shfl_down(local_vec_sim_sum, p);
            // if (g.thread_rank() < p) {
            //
            //}
          }
          // for (int k = 0; k < cols; k++) {
          //    a_norm_i += other_in_val[i * cols + k] * other_in_val[i* cols + k];
          //    b_norm_j += self_in_val[j * cols + k] * self_in_val[j * cols + k];
          //    vec_sim_mat[z] += other_in_val[i * cols + k] * self_in_val[j * cols + k];
          //}

          if (g.thread_rank() == 0) {
            a_norm_i = sqrt(local_a_norm_i_sum);
            b_norm_j = sqrt(local_b_norm_j_sum);

            // CHECK(a_norm_i >= 1e-5 && b_norm_j >= 1e-5);
            if (!(a_norm_i >= 1e-5 && b_norm_j >= 1e-5)) {
                printf("a_norm_i or b_norm_j not >= 1e-5, %f, %f, %f, %f\n", local_a_norm_i_sum, local_b_norm_j_sum, a_norm_i, b_norm_j);
            }

            vec_sim_mat[z] = local_vec_sim_sum / (a_norm_i * b_norm_j);
          }
          g.sync();
        }
}

template <typename T>
__global__ void CalcFeatureMatch(
        const int ins_num,
        const int fea_num,
        const T* weighted_sample_feature,
        T* feature_match_mat) {
    CUDA_KERNEL_LOOP(z, fea_num * ins_num * ins_num) {
        int fea_id = z / (ins_num * ins_num);
        int j = (z % (ins_num * ins_num)) / ins_num;
        int k = (z % (ins_num * ins_num)) % ins_num;
        if (k >= j) {
            if (weighted_sample_feature[j * fea_num + fea_id]
                    == weighted_sample_feature[k * fea_num + fea_id]) {
                // 那么，在第i个特征上，j,k和k,j的值都是1.0,否则为0.0
                feature_match_mat[fea_id * ins_num * ins_num + j * ins_num + k] = 1.0; // z
                feature_match_mat[fea_id * ins_num * ins_num + k * ins_num + j] = 1.0;
            } else {
                feature_match_mat[fea_id * ins_num * ins_num + j * ins_num + k] = 0.0; // z
                feature_match_mat[fea_id * ins_num * ins_num + k * ins_num + j] = 0.0;
            }
        }
    }
}

template <typename T>
__global__ void CalcSampleWeight(
        const int ins_num,
        const int fea_num,
        const double vec_sim_max,
        const double vec_sim_base,
        const double fea_match_base, 
        const T* vec_sim_mat,
        const T* feature_match_mat,
        T* sample_weight_mat) {

    CUDA_KERNEL_LOOP(z, ins_num * ins_num) {
        int i = z / ins_num;
        int j = z % ins_num;
        // double cosine = vec_sim_mat[z];
        double cosine = vec_sim_mat[z];
        if (i == j) {
            sample_weight_mat[z] = 0.0;
            continue;
        }
        if (cosine > vec_sim_max) {
            cosine = -cosine;
        }
        double w = 1.0;
        // 在安全距离外，距离越近负采样概率越大
        // 1000.0的cosine次方
        w *= powf(vec_sim_base, cosine);
        // 增加feature与自身不match的点的负采样概率
        // 在一跳孪生网络中，_fea_match_base都是1.0,所以下面的逻辑实际上都是乘以1
        for (int k = 0; k < fea_num; ++k) {
            w *= powf(fea_match_base, 1.0 - feature_match_mat[k * ins_num * ins_num + i * ins_num + j]);
        }
        // 原来这个sample_weight_mat二维矩阵的意义是<i,j>样本对作为负样本的概率，自身<i,i>不作为负样本
        sample_weight_mat[z] = w;
    }

}

template <typename T>
__global__ void DoWeightedRematch(const int ins_num,
        const T* sample_weight_mat,
        double* cdf_mat) {
/*
    CUDA_KERNEL_LOOP(i, ins_num) {

        double* cdf = cdf_mat + i * ins_num;
        const T* sample_weight_start = sample_weight_mat + i * ins_num;
        cdf[0] = sample_weight_start[0];
        for (int k = 1; k < ins_num; k++) cdf[k] = cdf[k - 1] + sample_weight_start[k];
        double sum = cdf[ins_num - 1];
        if (!(sum >= 1e-5)) { printf("sum not >= 1e-5\n"); }
        for (int k = 0; k < ins_num; k++) {cdf[k] /= sum; }
    }
*/

  // 32 threads(warp) for each ins      
  const size_t i = blockIdx.x * 32 + threadIdx.x / WARP_SIZE;
  if (i >= ins_num) return;

  // const size_t j = threadIdx.y;

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);
  const size_t j = g.thread_rank();

  int cdf_per_thread = ins_num / WARP_SIZE;
  int remain = ins_num % WARP_SIZE;
  
  int cdf_size = cdf_per_thread;

  if (j < remain) cdf_size++;

  //float cdf[cdf_size];
  double* cdf = cdf_mat + i * ins_num;
  const T* sample_weight_start = sample_weight_mat + i * ins_num;

  if (j < remain) {
    sample_weight_start += j * (cdf_per_thread + 1);
    cdf += j * (cdf_per_thread + 1); 
  }
  else {
    sample_weight_start += remain * (cdf_per_thread + 1) + (j - remain) * cdf_per_thread;
    cdf += remain * (cdf_per_thread + 1) + (j - remain) * cdf_per_thread;
  }

  // prefix sum
  cdf[0] = sample_weight_start[0];
  for (int k = 1; k < cdf_size; k++) cdf[k] = cdf[k-1] + sample_weight_start[k];

  double cdf_offset = 0.0;
  for (int i = 1; i < WARP_SIZE; i *= 2) {
    double temp_sum = g.shfl_up(cdf[cdf_size -1] + cdf_offset, i);
    if (g.thread_rank() >= i) {
      cdf_offset += temp_sum;
    }
  }

  // broadcast sum to all threads within warp
  double sum = 0.0;
  if (g.thread_rank() == WARP_SIZE - 1) {
    sum = cdf[cdf_size - 1] + cdf_offset;
    if (!(sum >= 1e-5)) {
      printf("sum not >= 1e-5\n");
    }
  }
  g.sync();
  sum = g.shfl(sum, WARP_SIZE - 1);
  
  for (int j = 0; j < cdf_size; ++j) {
    cdf[j] =  (cdf[j] + cdf_offset) / sum;
  }

}

__global__ void FillRandomVal(const int sample_num,
                         const int ins_num,
                         const double* cdf_data,
                         int64_t* random_val) {
    CUDA_KERNEL_LOOP(z, ins_num * sample_num) {  
      const int i = z / sample_num;
      const int j = z % sample_num;
      int pos = ins_num -1;      
      // "check"
      int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
      curandState state;
      curand_init(clock64(), tid_x, 0, &state);
      double rand_num = curand_uniform_double(&state);
      const double* cdf = cdf_data + i * ins_num;
      for (int k = 0; k < ins_num; ++k) {
        if (cdf[k] > rand_num) {
          pos = k;
          break;
        }
      }
      random_val[i + (j + 1) * ins_num] = pos;
    }
}

/*
template <typename T>
void WeightedRematch(const framework::ExecutionContext& ctx,
        const int fea_num,
        const int ori_ins_num,
        const int col_num,
        const int ext_ins_num,
        const float vec_sim_max,
        const float vec_sim_base,
        const float fea_match_base, 
        const T* self_in_val,
        const T* other_in_val,
        const T* weighted_sample_feature,
        int64_t* random_label_target_val,
        int64_t* random_val) {

    auto gpu_stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();
    int rematch_ratio = ext_ins_num / ori_ins_num;
    if (ori_ins_num == 1) {
        int grid_size = (ext_ins_num - 1) / 1024 + 1;
        int block_size = 1024;
        CopyKernel3<<<grid_size, block_size, 0, gpu_stream>>>(random_val, random_label_target_val, ext_ins_num);
        // for (int i = 1; i < ext_ins_num; ++i) {
        //    random_val(i, 0) = 0;  // rematch to the only ins
        //    random_label_target_val(i, 0) = 1;  // correct labels to 1
        // }
        return;
    }

    // create vec_sim_mat, feature_match_mat, sample_weight_mat
    std::vector<float> vec_sim_mat(ori_ins_num * ori_ins_num);
    std::vector<float> feature_match_mat(fea_num * ori_ins_num * ori_ins_num);
    std::vector<float> sample_weight_mat(ori_ins_num * ori_ins_num);
    std::vector<float> cdf_mat(ori_ins_num, ori_ins_num);

    paddle::framework::MixVector<T> vec_sim_mat_v(&vec_sim_mat);
    auto vec_sim_data = vec_sim_mat_v.CUDAMutableData(ctx.GetPlace());

    paddle::framework::MixVector<float> feature_match_mat_v(&feature_match_mat);
    auto feature_match_data = feature_match_mat_v.CUDAMutableData(ctx.GetPlace());

    paddle::framework::MixVector<float> sample_weight_mat_v(&sample_weight_mat);
    auto sample_weight_data = sample_weight_mat_v.CUDAMutableData(ctx.GetPlace());

    // cdf_data
    paddle::framework::MixVector<float> cdf_mat_v(&cdf_mat);
    auto cdf_data = cdf_mat_v.CUDAMutableData(ctx.GetPlace());

    int calc_sim_grid_size = (ori_ins_num * ori_ins_num - 1) / 1024 + 1;
    int calc_sim_block_size = 1024;
    CalcVectorSimilarityKernel<<<calc_sim_grid_size, calc_sim_block_size, 0, gpu_stream>>>(ori_ins_num, col_num, self_in_val, other_in_val, vec_sim_data);

    int calc_fea_grid_size = (fea_num * ori_ins_num * ori_ins_num - 1) / 1024 + 1;
    int calc_fea_block_size = 1024;
    CalcFeatureMatch<<<calc_fea_grid_size, calc_fea_block_size, 0, gpu_stream>>>(ori_ins_num, fea_num, weighted_sample_feature, feature_match_data);

    int calc_sample_grid_size = (ori_ins_num * ori_ins_num - 1) / 1024 + 1;
    int calc_sample_block_size = 1024;
    CalcSampleWeight<<<calc_sample_grid_size, calc_sample_block_size, 0, gpu_stream>>>(ori_ins_num, fea_num, vec_sim_max, vec_sim_base, fea_match_base, vec_sim_data, feature_match_data, sample_weight_data);

    const size_t grid_size_ = (ins_num - 1) / 32 + 1;
    dim3 grid_dims(grid_size_);
    dim3 block_dims(32, 32);
    DoWeightedRematch<<<grid_dims, block_dims, 0, gpu_stream>>>(ori_ins_num, rematch_ratio - 1, sample_weight_data, random_val, cdf_data);

    int fill_random_grid_size = ((rematch_ratio -1)* ori_ins_num - 1) / 1024 + 1;
    int fill_random_block_size = 1024;
    FillRandomVal<<<fill_random_grid_size, fill_random_block_size, 0, gpu_stream>>>(rematch_ratio - 1, ins_num, cdf_data, random_val);

}
*/

/*

// self_in_val: news_u2_output [ins_num, 64]
// other_in_val: news_u1_output [ins_num, 64]
// random_val: [ins_num * 4, 1], 前面前面ins_num行，每行的值为ins的id, 后面每行都是0
// _weighted_rematch(self_in_val, other_in_val, random_val);
void _weighted_rematch(
        const float vec_sim_max,
        const float vec_sim_base,
        const float fea_match_base, 
        const Eigen::MatrixXf& self_in_val,
        const Eigen::MatrixXf& other_in_val,
        const Eigen::MatrixXf& weighted_sample_feature,
        Eigen::MatrixXf& random_label_target_val,
        Eigen::MatrixXf& random_val) {

    //std::shared_ptr<MatrixOutput> weighted_sample_feature_ptr
    //    = component_table()->get_component<MatrixOutput>("weighted_sample_feature");
    //if (_weight_formula == "default") {
    //    CHECK(weighted_sample_feature_ptr != nullptr);
    //}

    // shape（ins_num, 2)
    // const Eigen::MatrixXf& weighted_sample_feature = _feature_input->value();
    // fea_num = 2 
    int fea_num = weighted_sample_feature.cols();
    // ori_ins_num = ins_num
    int ori_ins_num = self_in_val.rows();

    // ext_ins_num = ins_num * 4
    int ext_ins_num = random_val.rows();

    // rematch_ratio = 4
    int rematch_ratio = ext_ins_num / ori_ins_num;

    if (ori_ins_num == 1) {
        //LOG(WARNING) << "ins_num=" << ori_ins_num << ", TODO: correct random labels";
        //std::shared_ptr<MatrixOutput> random_label_target_ptr 
        //    = component_table()->get_component<MatrixOutput>("random_label_target");
        for (int i = 1; i < ext_ins_num; ++i) {
            random_val(i, 0) = 0;  // rematch to the only ins
            random_label_target_val(i, 0) = 1;  // correct labels to 1

        }
        return;
    }

    // 这个看上去是要求ins与ins间的similarity
    // (ins_num, ins_num)
    //
    Eigen::MatrixXf vec_sim_mat(ori_ins_num, ori_ins_num);

    //fea_num个Matrix,每个都是(ins_num, ins_num)
    std::vector<Eigen::MatrixXf> feature_match_mat(fea_num,
            Eigen::MatrixXf(ori_ins_num, ori_ins_num));
    // 这个也是shape = (ins_num, ins_num)
    Eigen::MatrixXf sample_weight_mat(ori_ins_num, ori_ins_num);

    // self_in_val: news_u2_output [ins_num, 64]
    // other_in_val: news_u1_output [ins_num, 64]
    // Eigen::MatrixXf vec_sim_mat(ori_ins_num, ori_ins_num);
    _calc_vector_similarity(self_in_val, other_in_val, vec_sim_mat);

    //上面调用完，vec_sim_mat就是计算后的相似度矩阵

    // shape（ins_num, 2)
    // const Eigen::MatrixXf& weighted_sample_feature = _feature_input->value();
    // fea_num个Matrix,每个都是(ins_num, ins_num)
    // std::vector<Eigen::MatrixXf> feature_match_mat(fea_num,
    //        Eigen::MatrixXf(ori_ins_num, ori_ins_num));
    _calc_feature_match(weighted_sample_feature, feature_match_mat);


    // Eigen::MatrixXf vec_sim_mat(ori_ins_num, ori_ins_num);
    // std::vector<Eigen::MatrixXf> feature_match_mat(fea_num,
    //        Eigen::MatrixXf(ori_ins_num, ori_ins_num));
    // 这个也是shape = (ins_num, ins_num)
    // Eigen::MatrixXf sample_weight_mat(ori_ins_num, ori_ins_num);
    // 这个函数是计算样本对<i,j>作为负样本的概率
    _calc_sample_weight(vec_sim_max, vec_sim_base, fea_match_base, vec_sim_mat, feature_match_mat, sample_weight_mat);

    // Eigen::MatrixXf sample_weight_mat(ori_ins_num, ori_ins_num);
    // 这个函数是计算样本对<i,j>作为负样本的概率
    // rematch_ratio - 1 = 3
    // random_val: [ins_num * 4, 1], 前面前面ins_num行，每行的值为ins的id, 后面每行都是0
    _do_weighted_rematch(sample_weight_mat, rematch_ratio - 1, random_val);

}

*/

#endif


namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;
using LoD = framework::LoD;
using LoDTensor = framework::LoDTensor;



template <typename T>
void WeightedRematch(const framework::ExecutionContext& ctx,
        const int fea_num,
        const int ori_ins_num,
        const int col_num,
        const int ext_ins_num,
        const double vec_sim_max,
        const double vec_sim_base,
        const double fea_match_base, 
        const T* self_in_val,
        const T* other_in_val,
        const T* weighted_sample_feature,
        int64_t* random_label_target_val,
        int64_t* random_val) {

    auto gpu_stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();
    int rematch_ratio = ext_ins_num / ori_ins_num;

    if (ori_ins_num == 1) {
        int grid_size = (ext_ins_num - 1) / 1024 + 1;
        int block_size = 1024;
        CopyKernel3<<<grid_size, block_size, 0, gpu_stream>>>(random_val, random_label_target_val, ext_ins_num);
        // for (int i = 1; i < ext_ins_num; ++i) {
        //    random_val(i, 0) = 0;  // rematch to the only ins
        //    random_label_target_val(i, 0) = 1;  // correct labels to 1
        // }
        return;
    }


    auto vec_sim_mat_v = memory::Alloc(ctx.GetPlace(), ori_ins_num * ori_ins_num * sizeof(T));
    T* vec_sim_data = reinterpret_cast<T*>(vec_sim_mat_v->ptr());

    auto feature_match_mat_v = memory::Alloc(ctx.GetPlace(), fea_num * ori_ins_num * ori_ins_num * sizeof(T));
    T* feature_match_data = reinterpret_cast<T*>(feature_match_mat_v->ptr());

    auto sample_weight_mat_v = memory::Alloc(ctx.GetPlace(), ori_ins_num * ori_ins_num * sizeof(T));
    T* sample_weight_data = reinterpret_cast<T*>(sample_weight_mat_v->ptr());

    auto cdf_mat_v = memory::Alloc(ctx.GetPlace(), ori_ins_num * ori_ins_num * sizeof(double));
    double* cdf_data = reinterpret_cast<double*>(cdf_mat_v->ptr());

    // 每个warp处理4行数据
    int calc_sim_grid_size = (ori_ins_num * ori_ins_num - 1) / (32 * 4) + 1;
    int calc_sim_block_size = 1024;
    CalcVectorSimilarityKernel<<<calc_sim_grid_size, calc_sim_block_size, 0, gpu_stream>>>(ori_ins_num, col_num, self_in_val, other_in_val, vec_sim_data);

    int calc_fea_grid_size = (fea_num * ori_ins_num * ori_ins_num - 1) / 1024 + 1;
    int calc_fea_block_size = 1024;
    CalcFeatureMatch<<<calc_fea_grid_size, calc_fea_block_size, 0, gpu_stream>>>(ori_ins_num, fea_num, weighted_sample_feature, feature_match_data);

    int calc_sample_grid_size = (ori_ins_num * ori_ins_num - 1) / 1024 + 1;
    int calc_sample_block_size = 1024;
    CalcSampleWeight<<<calc_sample_grid_size, calc_sample_block_size, 0, gpu_stream>>>(ori_ins_num, fea_num, vec_sim_max, vec_sim_base, fea_match_base, vec_sim_data, feature_match_data, sample_weight_data);

    const size_t grid_size_ = (ori_ins_num - 1) / 32 + 1;
    dim3 grid_dims(grid_size_);
    dim3 block_dims(1024);
    DoWeightedRematch<<<grid_dims, block_dims, 0, gpu_stream>>>(ori_ins_num, sample_weight_data, cdf_data);

    int fill_random_grid_size = ((rematch_ratio -1) * ori_ins_num - 1) / 1024 + 1;
    int fill_random_block_size = 1024;
    FillRandomVal<<<fill_random_grid_size, fill_random_block_size, 0, gpu_stream>>>(rematch_ratio - 1, ori_ins_num, cdf_data, random_val);

    cudaStreamSynchronize(gpu_stream);

}




template <typename T>
class WeightedRandomSampleOpGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // input
    auto* self_input_tensor = ctx.Input<framework::Tensor>("SelfInput");
    auto* other_input_tensor = ctx.Input<framework::Tensor>("OtherInput");
    auto* feature_input_tensor = ctx.Input<framework::Tensor>("FeatureInput");
    auto* label_input_tensor = ctx.Input<LoDTensor>("LabelInput");

    // output
    auto* output_tensor = ctx.Output<framework::Tensor>("Out");
    auto* random_label_target = ctx.Output<framework::Tensor>("RandomLabelTarget");
    auto* random_rematch_tensor = ctx.Output<framework::Tensor>("RandomRematch");


    int self_input_rows = self_input_tensor->dims()[0];
    int self_input_cols = self_input_tensor->dims()[1];

    // attr
    bool do_random = ctx.Attr<bool>("do_random");
    bool use_global_random_rematch = ctx.Attr<bool>("do_random");
    int random_rematch_ratio = ctx.Attr<int>("random_rematch_ratio");
    double vec_sim_max = ctx.Attr<float>("vec_sim_max");
    double vec_sim_base = ctx.Attr<float>("vec_sim_base");
    double fea_match_base = ctx.Attr<float>("fea_match_base");
    std::string weight_formula = ctx.Attr<std::string>("weight_formula");
    bool need_initialize = ctx.Attr<bool>("need_initialize");

    auto* self_input_data = self_input_tensor->data<T>();
    auto* other_input_data = other_input_tensor->data<T>();
    auto* feature_input_data = feature_input_tensor->data<T>();
    
    // debug
    auto place = ctx.GetPlace();
    auto dev_id = place.GetDeviceId(); 
    
    auto gpu_stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();

    if (need_initialize) {

      random_rematch_tensor->Resize({self_input_rows * random_rematch_ratio, 1});
      int64_t* random_rematch_data = random_rematch_tensor->mutable_data<int64_t>(ctx.GetPlace());
     
      random_label_target->Resize({self_input_rows * random_rematch_ratio, 1});
      int64_t* random_label_target_data = random_label_target->mutable_data<int64_t>(ctx.GetPlace());

      auto* label_data = label_input_tensor->data<int64_t>();

      if (!use_global_random_rematch) {
        int grid_size = (self_input_rows * random_rematch_ratio - 1) / 1024 + 1;
        int block_size = 1024;
        InitializeKernel<<<grid_size, block_size, 0, gpu_stream>>>(random_rematch_data, random_label_target_data, label_data, self_input_rows, random_rematch_ratio);

      } // else {
      //   InitializeKernel<<<>>>(random_rematch_data, random_label_target_data, label_data, self_input_rows, 1);
      //  for (int i = 1; i < random_rematch_ratio; i++) {
      //  RandomShuffle(tmp_v);
      //  InitializeKernel2<<<>>>(random_rematch_data + i * self_input_rows, random_label_target_data + i * self_input_rows, tmp_v, self_input_rows);
      //  }
      // }

/*
      // fill random_rematch_mat
      std::vector<int> tmp_v(self_input_rows);
      // 前面ins_num行，每一行的的值是j
      // 后面 3 * ins_num行，每一行的值是0
      for (int j = 0; j < self_input_rows; ++j) {
        random_val(j, 0) = j;
      }

      for (int i = 1; i < random_rematch_ratio; ++i) {
        // false
        if (use_global_random_rematch) {
            custom_random_shuffle(tmp_v);
        }
        for (int j = 0; j < self_input_rows; ++j) {
            // false
            if (use_global_random_rematch) {
               random_val(i * self_input_rows + j, 0) = tmp_v[j];
            }
            random_label_target_val(i * self_input_rows + j, 0) = 0;
        }
      }

      auto* label_input_data = label_input_tensor->data<int64_t>();
      for (int i = 0; i < self_input_rows; i++) {
        random_label_target_val(i, 0) = label_input_data[i]; 
      }
*/

    } 

    int in_num_row = self_input_rows;
    int out_num_row = in_num_row * random_rematch_ratio;

    output_tensor->Resize({out_num_row, self_input_cols});
    output_tensor->mutable_data<T>(ctx.GetPlace());

    auto* output_data = output_tensor->mutable_data<T>(ctx.GetPlace());
    auto* random_rematch_data = random_rematch_tensor->mutable_data<int64_t>(ctx.GetPlace());
    auto* random_label_target_data = random_label_target->mutable_data<int64_t>(ctx.GetPlace());

    if (do_random) {
        int fea_num = feature_input_tensor->dims()[1];
        WeightedRematch(ctx, fea_num, in_num_row, self_input_cols, out_num_row, vec_sim_max, vec_sim_base, fea_match_base, self_input_data, other_input_data, feature_input_data, random_label_target_data, random_rematch_data);
        int copy2_grid_size = (out_num_row * self_input_cols - 1) / 1024 + 1;
        int copy2_block_size = 1024;
        CopyKernel2<<<copy2_grid_size, copy2_block_size, 0, gpu_stream>>>(output_data, self_input_data, random_rematch_data, out_num_row, self_input_cols);
    } else {
        int copy_grid_size = (out_num_row * self_input_cols - 1) / 1024 + 1;
        int copy_block_size = 1024;
        CopyKernel<<<copy_grid_size, copy_block_size, 0, gpu_stream>>>(output_data, self_input_data, out_num_row, in_num_row, self_input_cols);
    }

  }
};

template <typename T>
__global__ void SampleGradKernel2(const T *output_grad_values, T *input_grad_values, const int out_grad_rows, const int out_grad_cols, const int in_grad_rows) {
    CUDA_KERNEL_LOOP(i, out_grad_rows * out_grad_cols) {
      int out_row_id = i / out_grad_cols;
      int col_id = i % out_grad_cols;
      int in_row_id = out_row_id % in_grad_rows;
      input_grad_values[in_row_id * out_grad_cols + col_id] = output_grad_values[i];
    }
}

template <typename T>
__global__ void SampleGradKernel(const T *output_grad_values, T *input_grad_values, const int64_t* random_values, const int out_grad_rows, const int out_grad_cols) {
    CUDA_KERNEL_LOOP(i, out_grad_rows * out_grad_cols) {
      int out_row_id = i / out_grad_cols;
      int col_id = i % out_grad_cols;
      int in_row_id = random_values[out_row_id];
      input_grad_values[in_row_id * out_grad_cols + col_id] = output_grad_values[i];
    }
}

template <typename T>
class WeightedRandomSampleGradOpGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    
        auto gpu_stream =
            ctx.template device_context<platform::CUDADeviceContext>().stream();

        auto* output_grad = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
        auto* self_input_grad = ctx.Output<LoDTensor>(framework::GradVarName("SelfInput"));
        auto* self_input = ctx.Input<LoDTensor>("SelfInput");
        bool do_random = ctx.Attr<bool>("do_random");
        auto* random_rematch = ctx.Input<LoDTensor>("RandomRematch");

        auto* output_grad_data = output_grad->data<T>();
        auto* random_rematch_data = random_rematch->data<int64_t>();

        self_input_grad->Resize(self_input->dims());
        auto self_input_grad_data = self_input_grad->mutable_data<T>(ctx.GetPlace());

        int out_grad_rows = output_grad->dims()[0];
        int out_grad_cols = output_grad->dims()[1];
        int in_grad_rows = self_input_grad->dims()[0];
        int sample_grid_size = (out_grad_rows * out_grad_cols - 1) / 1024 + 1;
        int sample_block_size = 1024;
        if (do_random) {
            SampleGradKernel<<<sample_grid_size, sample_block_size, 0, gpu_stream>>>(output_grad_data, self_input_grad_data, random_rematch_data,
                                                                       out_grad_rows, out_grad_cols);
        } else {
            SampleGradKernel2<<<sample_grid_size, sample_block_size, 0, gpu_stream>>>(output_grad_data, self_input_grad_data, out_grad_rows, out_grad_cols, in_grad_rows);
        }

  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(weighted_random_sample,
                        ops::WeightedRandomSampleOpGPUKernel<float>);

REGISTER_OP_CUDA_KERNEL(weighted_random_sample_grad,
                        ops::WeightedRandomSampleGradOpGPUKernel<float>);
