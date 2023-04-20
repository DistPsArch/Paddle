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
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/timer.h"


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
        const double vec_sim_max,
        const double vec_sim_base,
        const double fea_match_base, 
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



// self_in_val: news_u2_output [ins_num, 64]
// other_in_val: news_u1_output [ins_num, 64]
// random_val: [ins_num * 4, 1], 前面前面ins_num行，每行的值为ins的id, 后面每行都是0
// _weighted_rematch(self_in_val, other_in_val, random_val);
void _weighted_rematch(
        const double vec_sim_max,
        const double vec_sim_base,
        const double fea_match_base, 
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

    // 这块逻辑后面看
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




namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;
using LoD = framework::LoD;
using LoDTensor = framework::LoDTensor;

template <typename T>
class WeightedRandomSampleOpCPUKernel : public framework::OpKernel<T> {
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


    int ins_num = self_input_tensor->dims()[0];
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
    
    platform::Timer timeline;
    timeline.Start();

    // auto* output_data = output_tensor->data<T>();
    // auto* random_rematch_data = random_rematch_tensor->data<T>();
 
    // initialize random_val
    // 从self_input_tensor，other_input_tensor, feature_input_tensor拷贝到Eigen::MatrixXf
    Eigen::MatrixXf self_in_val;
    Eigen::MatrixXf other_in_val;
    Eigen::MatrixXf feature_in_val;
    Eigen::MatrixXf out_val;

    self_in_val.setZero(ins_num, self_input_tensor->dims()[1]);
    other_in_val.setZero(other_input_tensor->dims()[0], other_input_tensor->dims()[1]);
    feature_in_val.setZero(feature_input_tensor->dims()[0], feature_input_tensor->dims()[1]);

    timeline.Pause();
    if (dev_id == 0) {
      // std::cout << "debug EigenMatrixXf init self_in_val, other_in_val, feature_in_val " << timeline.ElapsedSec() << std::endl;
    }

    timeline.Start();
    
    for (int i = 0; i < ins_num; i++) {
      for (int j = 0; j < self_input_tensor->dims()[1]; j++) {
        self_in_val(i, j) = self_input_data[i * self_input_tensor->dims()[1] + j]; 
      }
    }
    for (int i = 0; i < other_input_tensor->dims()[0]; i++) {
      for (int j = 0; j < other_input_tensor->dims()[1]; j++) {
        other_in_val(i, j) = other_input_data[i * other_input_tensor->dims()[1] + j];   
      }
    }
    for (int i = 0; i < feature_input_tensor->dims()[0]; i++) {
      for (int j = 0; j < feature_input_tensor->dims()[1]; j++) {
        feature_in_val(i, j) = feature_input_data[i * feature_input_tensor->dims()[1] + j];   
      }
    }
 
    timeline.Pause();
    if (dev_id == 0) {
      // std::cout << "debug EigenMatrixXf init2 self_in_val, other_in_val, feature_in_val " << timeline.ElapsedSec() << std::endl;
    }

    timeline.Start();
    //shape = [ins_num * 4, 1]
    // initialize random_val
    Eigen::MatrixXf random_val;
    Eigen::MatrixXf random_label_target_val;
     
    random_val.setZero(ins_num * random_rematch_ratio, 1);
    random_label_target_val.setZero(ins_num * random_rematch_ratio, 1);

    if (need_initialize) {

      random_rematch_tensor->Resize({ins_num * random_rematch_ratio, 1});
      random_rematch_tensor->mutable_data<int64_t>(ctx.GetPlace());
     
      random_label_target->Resize({ins_num * random_rematch_ratio, 1});
      random_label_target->mutable_data<int64_t>(ctx.GetPlace());

      // fill random_rematch_mat
      std::vector<int> tmp_v(ins_num);
      // 前面ins_num行，每一行的的值是j
      // 后面 3 * ins_num行，每一行的值是0
      for (int j = 0; j < ins_num; ++j) {
        random_val(j, 0) = j;
      }

      for (int i = 1; i < random_rematch_ratio; ++i) {
        // false
        if (use_global_random_rematch) {
            custom_random_shuffle(tmp_v);
        }
        for (int j = 0; j < ins_num; ++j) {
            // false
            if (use_global_random_rematch) {
               random_val(i * ins_num + j, 0) = tmp_v[j];
            }
            random_label_target_val(i * ins_num +j, 0) = 0;
        }
      }

      auto* label_input_data = label_input_tensor->data<int64_t>();
      for (int i = 0; i < ins_num; i++) {
        random_label_target_val(i, 0) = label_input_data[i]; 
      }
    } else {
      // copy from random_label_target/random_rematch_tensor
      auto* random_rematch_data = random_rematch_tensor->data<int64_t>();
      auto* random_label_target_data = random_label_target->data<int64_t>();
      // random_val.setZero(ins_num * random_rematch_ratio, 1);
      // random_label_target_val.setZero(ins_num * random_rematch_ratio, 1);
      for (int i = 0 ; i < ins_num * random_rematch_ratio;i++) {
          random_val(i, 0) = random_rematch_data[i];
          random_label_target_val(i, 0) = random_label_target_data[i];
      }
    }

    timeline.Pause();
    
    if (dev_id == 0) {
      // std::cout << "debug init random_val, random_target_val " << timeline.ElapsedSec() << std::endl;
    }

    timeline.Start();

    // ins_num
    int in_num_row = self_in_val.rows();
    // ins_num * 4
    int out_num_row = random_val.rows();
    // (ins_num * 4, 64)
    out_val.setZero(out_num_row, self_in_val.cols());
    output_tensor->Resize({out_num_row, self_in_val.cols()});
    output_tensor->mutable_data<T>(ctx.GetPlace());


    if (do_random) {
        // self_in_val: news_u2_output [ins_num, 64]
        // other_in_val: news_u1_output [ins_num, 64]
        // random_val: [ins_num * 4, 1], 前面前面ins_num行，每行的值为ins的id, 后面每行都是0
        _weighted_rematch(vec_sim_max, vec_sim_base, fea_match_base, self_in_val, other_in_val, feature_in_val, random_label_target_val, random_val);
        for (int i = 0; i < out_num_row; ++i) {
            out_val.row(i) = self_in_val.row(random_val(i, 0));
        }
    } else {
        // 这逻辑就是从self_input里，通过 i % in_num_row来取对应的行。。
        // 这样以后，lable能对上吗
        // 这是row扩大了4倍呀,把输入赋值了4份 
        for (int i = 0; i < out_num_row; ++i) {
            out_val.row(i) = self_in_val.row(i % in_num_row);
        }
    }

    timeline.Pause();
    if (dev_id == 0) {
      // std::cout << "debug weighted random " << timeline.ElapsedSec() << std::endl;
    }


    timeline.Start();
    auto* output_data = output_tensor->mutable_data<T>(ctx.GetPlace());
    auto* random_rematch_data = random_rematch_tensor->mutable_data<int64_t>(ctx.GetPlace());
    auto* random_label_target_data = random_label_target->mutable_data<int64_t>(ctx.GetPlace());

    // 从out_val拷贝回output_tensor
    // 从random_val拷贝回random_rematch_tensor
    for (int i = 0; i < out_num_row; i++) {
      for (int j = 0; j < self_in_val.cols(); j++) {
        output_data[i * self_in_val.cols() + j] = out_val(i, j);   
      }
    }
    for (int i = 0; i < out_num_row; i++) {
      for (int j = 0; j < random_val.cols(); j++) {
        random_rematch_data[i * random_val.cols() + j] = (int64_t)random_val(i, j);
        random_label_target_data[i * random_val.cols() + j] = (int64_t)random_label_target_val(i, j);
      }
    }

    timeline.Pause();
    if (dev_id == 0) {
      // std::cout << "debug copy output " << timeline.ElapsedSec() << std::endl;
    }
  }
};

template <typename T>
class WeightedRandomSampleGradOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
        auto* output_grad = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
        auto* self_input_grad = ctx.Output<LoDTensor>(framework::GradVarName("SelfInput"));
        auto* self_input = ctx.Input<LoDTensor>("SelfInput");
        bool do_random = ctx.Attr<bool>("do_random");
        auto* random_rematch = ctx.Input<LoDTensor>("RandomRematch");

        Eigen::MatrixXf out_grad;
        Eigen::MatrixXf in_grad;
        Eigen::MatrixXf random_val;
       
        out_grad.setZero(output_grad->dims()[0], output_grad->dims()[1]);
        random_val.setZero(random_rematch->dims()[0], random_rematch->dims()[1]); 
        int in_num_row = self_input->dims()[0];
        in_grad.setZero(in_num_row, out_grad.cols());

        auto* output_grad_data = output_grad->data<T>();
        auto* random_rematch_data = random_rematch->data<int64_t>();

        // copy output_grad to out_grad
        // copy random_rematch_tensor to random_val;
        for (int i = 0; i < output_grad->dims()[0]; i++) {
          for (int j = 0; j < output_grad->dims()[1]; j++) {
            out_grad(i, j) = output_grad_data[i * output_grad->dims()[1] + j];
          }
        }
        for (int i = 0; i < random_rematch->dims()[0]; i++) {
          for (int j = 0; j < random_rematch->dims()[1]; j++) {
            random_val(i, j) = random_rematch_data[i * random_rematch->dims()[1] + j];
          }
        }
        if (do_random) {
            for (int i = 0; i < out_grad.rows(); ++i) {
                in_grad.row(random_val(i, 0)) += out_grad.row(i);
            }
        } else {
            for (int i = 0; i < out_grad.rows(); ++i) {
                in_grad.row(i % in_num_row) += out_grad.row(i);
            }
        }
        // copy in_grad to self_input_grad
        self_input_grad->Resize(self_input->dims());
        // self_input_grad->mutable_data<T>(ctx.GetPlace());

        auto* self_input_grad_data = self_input_grad->mutable_data<T>(ctx.GetPlace());
        for (int i = 0; i < self_input->dims()[0]; i++) {
          for (int j = 0; j < self_input->dims()[1]; j++) {
            self_input_grad_data[i * self_input->dims()[1] + j] = in_grad(i, j);
          }
        } 

  }
};

}  // namespace operators
}  // namespace paddle
