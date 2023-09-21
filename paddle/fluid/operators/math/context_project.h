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

#pragma once

#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {

namespace math {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

/*
 * \brief Context projection concatenates features in adjacent time-steps in
 * a sequence. The i-th row of the output is the concatenation of
 * context_length rows of the input. The context_length rows are the
 * consecutive rows from the i+shift_start row.
 * ContextProjectGradFunctor is the inverse process of ContextProjectFunctor.
 *
 * \param in            Input data.
 * \param Shape         The shape of Input data:
 *                        [mini-batch, input_hidden_size].
 *
 * \param padding_data  Padding data.
 * \param Shape         The shape of Padding data:
 *                        [up_pad + down_pad, input_hidden_size].
 *
 * \param col           Col data.
 * \param Shape         The shape of Col data:
 *                        [mini-batch, context_length * input_hidden_size].
 *
 * For a mini-batch of 2 variable lengths sentences, containing 3, and 1
 * time-steps:
 *
 * Assumed input (X) is a [4, M, N] float LoDTensor, and X->lod()[0] = [0, 3,
 * 4].
 * Besides, for the sake of simplicity, we assume M=1 and N=2.
 *
 * X = [[a1, a2;
 *       b1, b2;
 *       c1, c2]
 *      [d1, d2]]
 *
 * This is to say that input (X) has 4 words and the dimension of each word
 * representation is 2.
 *
 * - Case1:
 *   If context_start is -1 and padding_trainable is false, we use zero to pad
 *   instead of learned weight to pad,
 *   and the context_length is 3, the output (Out) is:
 *
 *   Out =[[0,  0,  a1, a2, b1, b2;
 *          a1, a2, b1, b2, c1, c2;
 *          b1, b2, c1, c2, 0,  0 ]
 *          [0,  0, d1, d2, 0,  0 ]]
 *
 * - Case2:
 *   If context_start is -1 and padding_trainable is true, we use learned weight
 *   to pad,
 *   and the context_length is 3, the output (Out) is:
 *
 *   Out = [[w1, w2, a1, a2, b1, b2;
 *           a1, a2, b1, b2, c1, c2;
 *           b1, b2, c1, c2, w3, w4]
 *          [w1, w2, d1, d2, w3, w4]]
 *
 */

template <typename DeviceContext, typename T>
class ContextProjectFunctor {
 public:
  void operator()(const DeviceContext& context, const LoDTensor& in,
                  const Tensor* padding_data, bool padding_trainable,
                  const int context_start, const int context_length,
                  const int context_stride, const int up_pad,
                  const int down_pad, Tensor* col) {
    auto lod_level_0 = in.lod()[0];

    // math::Im2ColFunctor<math::ColFormat::kOCF, DeviceContext, float> im2col_ocf;
    math::Im2ColFuseFunctor<math::ColFormat::kOCF, DeviceContext, float> im2col_ocf_fuse;

    std::vector<int> dilation({1, 1});
    std::vector<int> padding({up_pad, 0, down_pad, 0});
    std::vector<int> stride({context_stride, 1});

    // int input_row_begin, input_row_end;
    int sequence_width;
    sequence_width = in.dims()[1];

    // === optimize ===== 
/*
    for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
      if (lod_level_0[i] == lod_level_0[i + 1]) continue;

      input_row_begin = (context_start > 0)
                            ? static_cast<int>(lod_level_0[i]) + context_start
                            : static_cast<int>(lod_level_0[i]);
      input_row_end = static_cast<int>(lod_level_0[i + 1]);

      Tensor out_t = col->Slice(static_cast<int>(lod_level_0[i]),
                                static_cast<int>(lod_level_0[i + 1]));

      sequence_height = static_cast<int>(out_t.dims()[0]);

      if (input_row_begin < input_row_end) {
        Tensor in_t = in.Slice(input_row_begin, input_row_end);

        std::vector<int64_t> output_shape(
            {sequence_height, 1, 1, context_length,
             sequence_width});  // output_height, output_width,
        // input_channels, filter_height, filter_width
        out_t.Resize(phi::make_ddim(output_shape));

        std::vector<int64_t> input_shape(
            {1, input_row_end - input_row_begin,
             sequence_width});  // input_channels, input_height, input_width
        in_t.Resize(phi::make_ddim(input_shape));
        im2col_ocf(context, in_t, dilation, stride, padding, &out_t);
        out_t.Resize({sequence_height, context_length * sequence_width});
      }
    }
*/
    int concurrency_size = 5;
    int thread_size = static_cast<int>(lod_level_0.size()) - 1;
    std::vector<std::thread> threads_vec(concurrency_size);
    // std::vector<Tensor> input_tensor(thread_size); 
    // std::vector<Tensor> output_tensor(thread_size);


   // 优化掉，放在第一个循环里
    int im_channels = 1;
    std::vector<int> im_height(thread_size, 0);
    int im_width = sequence_width;
    int filter_height = context_length;
    int filter_width = sequence_width;
    int col_width = 1;
    std::vector<int> col_height(thread_size, 0);
    // framework::Vector<size_t> lod_level_0_cuda = lod_level_0;
    // im_data, col_data
    std::vector<float*> im_datas(thread_size);
    std::vector<float*> col_datas(thread_size);
    int max_col_height = -1;

/*
    for (int i= 0; i < size; i++) {
      if (lod_level_0[i] == lod_level_0[i+1]) {
        im_datas[i] = nullptr;
        col_datas[i] = nullptr;
        continue;
      }
      // == 其实也可以去掉 ==
      PADDLE_ENFORCE_EQ(im[i].dims().size(), 3,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'im' should be 3. But got "
                          "the dims of tensor 'im' is [%s].",
                          im[i].dims()));
      PADDLE_ENFORCE_EQ(col[i].dims().size(), 5,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'col' should be 5. But got "
                          "the dims of tensor 'col' is [%s].",
                          col[i].dims()));
      // == 其实也可以去掉 ==
      im_height[i] = im[i].dims()[1];
      if (im_channels == -1)im_channels = im[i].dims()[0];
      if (im_width == -1) im_width = im[i].dims()[2];
      if (filter_height == -1) filter_height = col[i].dims()[3];
      if (filter_width == -1) filter_width = col[i].dims()[4];
      if (col_width == -1) col_width = col[i].dims()[1];
      col_height[i] = col[i].dims()[0];
      if (col_height[i] > max_col_height) max_col_height = col_height[i];

      im_datas[i] = im[i].data<T>(); 
      col_datas[i] = col[i].data<T>();
    }
*/
    int avg_ele = thread_size / concurrency_size;
    int left_ele = thread_size % concurrency_size;
    
    for (int i = 0; i < concurrency_size; i++) {
     // int start_id = i * avg_ele;
     // int end_id = i < left_ele ? start_id + avg_ele + 1 : start_id + avg_ele;
        int start_id = -1, end_id = -1;
        if (i < left_ele) {
          start_id = i * (avg_ele + 1);
          end_id = start_id + avg_ele + 1;
        } else {
          start_id = (i - left_ele) * avg_ele + left_ele * (avg_ele + 1);
          end_id = start_id + avg_ele;
        }
 
     threads_vec[i] = std::thread([sequence_width, context_length, context_start, &lod_level_0, &in, &col, &im_datas, &col_datas, &col_height, &im_height](int start_id, int end_id) {
      for (int t = start_id; t <end_id; t++) {
        if (lod_level_0[t] == lod_level_0[t + 1]) {
          // input_tensor[i] = Tensor();
          // output_tensor[i] = Tensor();
          im_datas[t] = nullptr;
          col_datas[t] = nullptr;
          return;
        }
        int input_row_begin = (context_start > 0)
                            ? static_cast<int>(lod_level_0[t]) + context_start
                            : static_cast<int>(lod_level_0[t]);
        int input_row_end = static_cast<int>(lod_level_0[t + 1]);

        Tensor out_t = col->Slice(static_cast<int>(lod_level_0[t]),
                                      static_cast<int>(lod_level_0[t + 1]));
      
        col_datas[t] = out_t.data<float>();
        int sequence_height = static_cast<int>(out_t.dims()[0]);

        if (input_row_begin < input_row_end) {
          Tensor in_t = in.Slice(input_row_begin, input_row_end);

          // std::vector<int64_t> output_shape(
          //     {sequence_height, 1, 1, context_length,
          //      sequence_width});  // output_height, output_width,
          // input_channels, filter_height, filter_width
          // out_t.Resize(phi::make_ddim(output_shape));
          col_height[t] = sequence_height;
          // if (col_height[i] > max_col_height) max_col_height = col_height[i];

          // std::vector<int64_t> input_shape(
          //     {1, input_row_end - input_row_begin,
          //      sequence_width});  // input_channels, input_height, input_width
          // in_t.Resize(phi::make_ddim(input_shape));
          im_datas[t] = in_t.data<float>(); 
          im_height[t] = input_row_end - input_row_begin;
          // im2col_ocf(context, in_t, dilation, stride, padding, &out_t);
          // out_t.Resize({sequence_height, context_length * sequence_width});
        }
       }
      }, start_id, end_id);
    }
    for (int i = 0; i < concurrency_size; i++) {
      if (threads_vec[i].joinable()) threads_vec[i].join();
    }

    max_col_height = *std::max_element(col_height.begin(), col_height.end());
    // === kernel ===
    auto gpu_place = context.GetPlace();
    auto all_hbm = memory::Alloc(gpu_place, (4 * thread_size  + 1 + (4 * thread_size + 1) % 2) * sizeof(uint64_t));
    /*
    auto mixv_im_height = memory::Alloc(gpu_place, size * sizeof(int));
    auto mixv_col_height = memory::Alloc(gpu_place, size * sizeof(int));
    auto mixv_lod_level_0_cuda = memory::Alloc(gpu_place, (size + 1) * sizeof(size_t));
    auto mixv_im_data = memory::Alloc(gpu_place, size * sizeof(T*));
    auto mixv_col_data = memory::Alloc(gpu_place, size * sizeof(T*));

    int* im_height_data = reinterpret_cast<int*>(mixv_im_height->ptr());
    int* col_height_data = reinterpret_cast<int*>(mixv_col_height->ptr());
    size_t* lod_level_0_data = reinterpret_cast<size_t*>(mixv_lod_level_0_cuda->ptr());
    T** im_data = reinterpret_cast<T**>(mixv_im_data->ptr());
    T** col_data = reinterpret_cast<T**>(mixv_col_data->ptr());
    */
    
    int* im_height_data = reinterpret_cast<int*>(all_hbm->ptr());
    int* col_height_data = reinterpret_cast<int*>(im_height_data + thread_size);
    size_t* lod_level_0_data = reinterpret_cast<size_t*>(col_height_data + thread_size);
    float** im_data = reinterpret_cast<float**>(lod_level_0_data + thread_size + 1);
    float** col_data = reinterpret_cast<float**>(im_data + thread_size);

    // 其实im_height 就是col_height，这块可以继续优化 
    cudaMemcpy(im_height_data, im_height.data(), thread_size * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(col_height_data, col_height.data(), thread_size * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(lod_level_0_data, lod_level_0.data(), (thread_size + 1)  * sizeof(size_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(im_data, im_datas.data(), thread_size * sizeof(float*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(col_data, col_datas.data(), thread_size * sizeof(float*),
               cudaMemcpyHostToDevice);

/*
    int block_dim_x = 0;
    int block_dim_y = 0;
    if (filter_height <= 4 && filter_width <= 4) {
      block_dim_x = 4;
      block_dim_y = 4;
    } else if (filter_height <= 8 && filter_width <= 8) {
      block_dim_x = 8;
      block_dim_y = 8;
    } else if (filter_height <= 16 && filter_width <= 16) {
      block_dim_x = 16;
      block_dim_y = 16;
    } else {
      block_dim_x = 32;
      block_dim_y = 32;
    } 
    int block_dim_z = 1024 / block_dim_x / block_dim_y;
    dim3 threads(block_dim_x, block_dim_y, std::min(block_dim_z, im_channels));
    dim3 grid(col_width, max_col_height, thread_size);

    im2colOCF_Fuse_2<T><<<grid, threads, 0, context.stream()>>>(
     im_data, thread_size, lod_level_0_data, im_channels, im_height_data, im_width, filter_height,
     filter_width, stride[0], stride[1], padding[0], padding[1], col_height_data,
     col_width, col_data);
*/
    im2col_ocf_fuse(context, im_data, thread_size, filter_height, filter_width, im_width, col_width, max_col_height, im_channels,
                    col_height_data, im_height_data, lod_level_0_data, dilation, stride, padding, col_data);
    // === kernel ===
/*
      threads_vec.clear();
      threads_vec.resize(concurrency_size); 

    for (int i = 0; i < concurrency_size; i++) {
     // int start_id = i * avg_ele;
     // int end_id = i < left_ele ? start_id + avg_ele + 1 : start_id + avg_ele; 
        int start_id = -1, end_id = -1;
        if (i < left_ele) {
          start_id = i * (avg_ele + 1);
          end_id = start_id + avg_ele + 1;
        } else {
          start_id = (i - left_ele) * avg_ele + left_ele * (avg_ele + 1);
          end_id = start_id + avg_ele;
        }
     threads_vec[i] = std::thread([context_length, sequence_width, &lod_level_0, &output_tensor](int start_id, int end_id) {
       for (int t = start_id; t <end_id; t++) {
         if (lod_level_0[t] == lod_level_0[t + 1]) continue;
         int sequence_height = static_cast<int>(output_tensor[t].dims()[0]);
         output_tensor[t].Resize({sequence_height, context_length * sequence_width});
       }
      }, start_id, end_id);
    }
    for (int i = 0; i < concurrency_size; i++) {
      if (threads_vec[i].joinable()) threads_vec[i].join();
    }
    */

    // === optimize ===== 

    if (padding_trainable) {
      PADDLE_ENFORCE_NOT_NULL(
          padding_data,
          platform::errors::InvalidArgument(
              "The input tensor 'padding_data' should not be NULL."));
      for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
        if (lod_level_0[i] == lod_level_0[i + 1]) continue;

        Tensor out_t = col->Slice(static_cast<int>(lod_level_0[i]),
                                  static_cast<int>(lod_level_0[i + 1]));

        int sequence_height = static_cast<int>(out_t.dims()[0]);

        // add up trainable data
        out_t.Resize({static_cast<int64_t>(sequence_height) * context_length,
                      sequence_width});

        if (up_pad > 0) {  // add up pad
          int padding_rows = std::min(
              up_pad, static_cast<int>(lod_level_0[i + 1] - lod_level_0[i]));

          for (int k = 0; k < padding_rows; ++k) {
            int padding_size =
                k + context_length < up_pad ? context_length : up_pad - k;
            Tensor out_t_sub = out_t.Slice(k * context_length,
                                           k * context_length + padding_size);
            Tensor w_sub = padding_data->Slice(k, k + padding_size);
            framework::TensorCopy(w_sub, context.GetPlace(), context,
                                  &out_t_sub);
          }
        }
        if (down_pad > 0) {  // add down pad
          int down_pad_begin_row =
              std::max(0,
                       (sequence_height - context_start - context_length) + 1) +
              1;
          int padding_begin = std::max(0, context_start - sequence_height);
          int padding_size =
              sequence_height - context_start >= context_length
                  ? 1
                  : context_length - (sequence_height - context_start);
          if (context_start >= sequence_height) padding_size = context_length;
          int padding_idx = padding_begin;
          for (int t = 0; t + down_pad_begin_row <= sequence_height;
               ++t, ++padding_size) {
            if (context_start >= sequence_height) padding_size = context_length;
            if (padding_size > context_length) {
              padding_size = context_length;
              padding_idx++;
            }
            if (padding_begin > 0 || sequence_height == context_start)
              padding_idx = padding_begin + t;

            Tensor out_t_sub = out_t.Slice(
                (down_pad_begin_row + t) * context_length - padding_size,
                (down_pad_begin_row + t) * context_length);
            Tensor w_sub = padding_data->Slice(
                up_pad + padding_idx, up_pad + padding_idx + padding_size);
            framework::TensorCopy(w_sub, context.GetPlace(), context,
                                  &out_t_sub);
          }
        }
        out_t.Resize({sequence_height,
                      static_cast<int64_t>(context_length) * sequence_width});
      }
    }
  }
};

template <typename DeviceContext, typename T>
class ContextProjectGradFunctor {
 public:
  void operator()(const DeviceContext& context, const LoDTensor& in,
                  bool padding_trainable, const int context_start,
                  const int context_length, const int context_stride,
                  const int up_pad, const int down_pad, bool pad_grad,
                  bool input_grad, Tensor* padding_data, Tensor* col) {
    auto lod_level_0 = in.lod()[0];

    // math::Col2ImFunctor<math::ColFormat::kOCF, DeviceContext, float> col2im_ocf;
    math::Col2ImFuseFunctor<math::ColFormat::kOCF, DeviceContext, float> col2im_ocf_fuse;

    std::vector<int> dilation({1, 1});
    std::vector<int> padding({up_pad, 0, down_pad, 0});
    std::vector<int> stride({context_stride, 1});

    // int input_row_begin, input_row_end;
    int sequence_width;
    sequence_width = in.dims()[1];
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);

    if (input_grad) {

/*
      for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
        if (lod_level_0[i] == lod_level_0[i + 1]) continue;

        input_row_begin = (context_start > 0)
                              ? static_cast<int>(lod_level_0[i]) + context_start
                              : static_cast<int>(lod_level_0[i]);
        input_row_end = static_cast<int>(lod_level_0[i + 1]);

        Tensor out_t = col->Slice(static_cast<int>(lod_level_0[i]),
                                  static_cast<int>(lod_level_0[i + 1]));

        sequence_height = static_cast<int>(out_t.dims()[0]);

        if (input_row_begin < input_row_end) {
          Tensor in_t = in.Slice(input_row_begin, input_row_end);

          std::vector<int64_t> output_shape(
              {sequence_height, 1, 1, context_length,
               sequence_width});  // output_height, output_width,
          // input_channels, filter_height, filter_width
          out_t.Resize(phi::make_ddim(output_shape));

          std::vector<int64_t> input_shape(
              {1, input_row_end - input_row_begin,
               sequence_width});  // input_channels, input_height, input_width
          in_t.Resize(phi::make_ddim(input_shape));

          col2im_ocf(context, out_t, dilation, stride, padding, &in_t);
          out_t.Resize({sequence_height, context_length * sequence_width});
        }
      }

*/
      int concurrency_size = 5;
      int thread_size = static_cast<int>(lod_level_0.size()) - 1;
      std::vector<std::thread> threads_vec(concurrency_size);
      // std::vector<Tensor> output_tensor(thread_size); 

      int im_channels = 1;
      std::vector<int> im_height(thread_size, 0);
      int im_width = sequence_width;
      int filter_height = context_length;
      int filter_width = sequence_width;
      int col_width = 1;
      std::vector<int> col_height(thread_size, 0);
    
      // framework::Vector<size_t> lod_level_0_cuda = lod_level_0;
      // im_data, col_data
      std::vector<float*> im_datas(thread_size);
      std::vector<float*> col_datas(thread_size);
      int max_col_height = -1;


      int avg_ele = thread_size / concurrency_size;
      int left_ele = thread_size % concurrency_size;
      for (int i = 0; i < concurrency_size; i++) {
       //  int start_id = i * avg_ele;
       //  int end_id = i < left_ele ? start_id + avg_ele + 1 : start_id + avg_ele;
 
        int start_id = -1, end_id = -1;
        if (i < left_ele) {
          start_id = i * (avg_ele + 1);
          end_id = start_id + avg_ele + 1;
        } else {
          start_id = (i - left_ele) * avg_ele + left_ele * (avg_ele + 1);
          end_id = start_id + avg_ele;
        }

        threads_vec[i] = std::thread([sequence_width, context_length, context_start, &lod_level_0, &in, &col, &im_datas, &col_datas, &col_height, &im_height](int start_id, int end_id) {
         for (int t = start_id; t < end_id; t++) {

          if (lod_level_0[t] == lod_level_0[t + 1]) {
            im_datas[t] = nullptr;     
            col_datas[t] = nullptr;     
            // input_tensor[i] = Tensor();
            // output_tensor[i] = Tensor();
            return;
          }
          int input_row_begin = (context_start > 0)
                              ? static_cast<int>(lod_level_0[t]) + context_start
                              : static_cast<int>(lod_level_0[t]);
          int input_row_end = static_cast<int>(lod_level_0[t + 1]);

          Tensor out_t = col->Slice(static_cast<int>(lod_level_0[t]),
                                      static_cast<int>(lod_level_0[t + 1]));
      
          col_datas[t] = out_t.data<float>();
          int sequence_height = static_cast<int>(out_t.dims()[0]);

          if (input_row_begin < input_row_end) {
            Tensor in_t = in.Slice(input_row_begin, input_row_end);

            col_height[t] = sequence_height;
            // std::vector<int64_t> output_shape(
            //     {sequence_height, 1, 1, context_length,
            //      sequence_width});  // output_height, output_width,
            // input_channels, filter_height, filter_width
            // output_tensor[i].Resize(phi::make_ddim(output_shape));

            // std::vector<int64_t> input_shape(
            //    {1, input_row_end - input_row_begin,
            //     sequence_width});  // input_channels, input_height, input_width
            im_datas[t] = in_t.data<float>(); 
            im_height[t] = input_row_end - input_row_begin;
            // input_tensor[i].Resize(phi::make_ddim(input_shape));
            // im2col_ocf(context, in_t, dilation, stride, padding, &out_t);
            // out_t.Resize({sequence_height, context_length * sequence_width});
           }
          }
        }, start_id, end_id);
      }

      for (int i = 0; i < concurrency_size; i++) {
        if (threads_vec[i].joinable()) threads_vec[i].join();
      }
     
      threads_vec.clear();
      threads_vec.resize(concurrency_size); 
      max_col_height = *std::max_element(col_height.begin(), col_height.end());

      auto gpu_place = context.GetPlace();
      auto all_hbm = memory::Alloc(gpu_place, (4 * thread_size  + 1 + (4 * thread_size + 1) % 2) * sizeof(uint64_t));
    /*
    auto mixv_im_height = memory::Alloc(gpu_place, size * sizeof(int));
    auto mixv_col_height = memory::Alloc(gpu_place, size * sizeof(int));
    auto mixv_lod_level_0_cuda = memory::Alloc(gpu_place, (size + 1) * sizeof(size_t));
    auto mixv_im_data = memory::Alloc(gpu_place, size * sizeof(T*));
    auto mixv_col_data = memory::Alloc(gpu_place, size * sizeof(T*));

    int* im_height_data = reinterpret_cast<int*>(mixv_im_height->ptr());
    int* col_height_data = reinterpret_cast<int*>(mixv_col_height->ptr());
    size_t* lod_level_0_data = reinterpret_cast<size_t*>(mixv_lod_level_0_cuda->ptr());
    T** im_data = reinterpret_cast<T**>(mixv_im_data->ptr());
    T** col_data = reinterpret_cast<T**>(mixv_col_data->ptr());
    */
    
      int* im_height_data = reinterpret_cast<int*>(all_hbm->ptr());
      int* col_height_data = reinterpret_cast<int*>(im_height_data + thread_size);
      size_t* lod_level_0_data = reinterpret_cast<size_t*>(col_height_data + thread_size);
      float** im_data = reinterpret_cast<float**>(lod_level_0_data + thread_size + 1);
      float** col_data = reinterpret_cast<float**>(im_data + thread_size);

      // 其实im_height 就是col_height，这块可以继续优化 
      cudaMemcpy(im_height_data, im_height.data(), thread_size * sizeof(int),
                cudaMemcpyHostToDevice);
      cudaMemcpy(col_height_data, col_height.data(), thread_size * sizeof(int),
               cudaMemcpyHostToDevice);
      cudaMemcpy(lod_level_0_data, lod_level_0.data(), (thread_size + 1)  * sizeof(size_t),
                cudaMemcpyHostToDevice);
      cudaMemcpy(im_data, im_datas.data(), thread_size * sizeof(float*),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(col_data, col_datas.data(), thread_size * sizeof(float*),
                 cudaMemcpyHostToDevice);


      col2im_ocf_fuse(context, col_data, thread_size, filter_height, filter_width, im_width, col_width, max_col_height, im_channels,
                    col_height_data, im_height_data, lod_level_0_data, dilation, stride, padding, im_data);


      // col2im_ocf_fuse(context, output_tensor, thread_size, lod_level_0, dilation, stride, padding, input_tensor.data());

      // col2im_ocf_fuse(context, input_tensor, thread_size, lod_level_0, dilation, stride, padding, output_tensor.data());
      // for (int i = 0; i < concurrency_size; i++) {
      //   if (lod_level_0[i] == lod_level_0[i + 1]) continue;
      //   int sequence_height = static_cast<int>(output_tensor[i].dims()[0]);
      //   output_tensor[i].Resize({sequence_height, context_length * sequence_width});
      // }
/*
      for (int i = 0; i < concurrency_size; i++) {
        int start_id = -1, end_id = -1;
        if (i < left_ele) {
          start_id = i * (avg_ele + 1);
          end_id = start_id + avg_ele + 1;
        } else {
          start_id = (i - left_ele) * avg_ele + left_ele * (avg_ele + 1);
          end_id = start_id + avg_ele;
        }
        threads_vec[i] = std::thread([context_length, sequence_width, &lod_level_0, &output_tensor](int start_id, int end_id) {
         for (int t = start_id; t < end_id; t++) {
           if (lod_level_0[t] == lod_level_0[t + 1]) continue;
           int sequence_height = static_cast<int>(output_tensor[t].dims()[0]);
           output_tensor[t].Resize({sequence_height, context_length * sequence_width});
         }
        }, start_id, end_id);
      }
    
      for (int i = 0; i < concurrency_size; i++) {
        if (threads_vec[i].joinable()) threads_vec[i].join();
      }
*/
    // === optimize ===== 

    }
    if (pad_grad) {
      if (padding_trainable) {
        for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
          if (lod_level_0[i] == lod_level_0[i + 1]) continue;

          Tensor out_t = col->Slice(static_cast<int>(lod_level_0[i]),
                                    static_cast<int>(lod_level_0[i + 1]));

          int sequence_height = static_cast<int>(out_t.dims()[0]);
          out_t.Resize({static_cast<int64_t>(sequence_height) * context_length,
                        sequence_width});

          if (up_pad > 0) {
            int padding_rows = std::min(
                up_pad, static_cast<int>(lod_level_0[i + 1] - lod_level_0[i]));

            for (int k = 0; k < padding_rows; ++k) {
              int padding_size =
                  k + context_length < up_pad ? context_length : up_pad - k;
              Tensor out_t_sub = out_t.Slice(k * context_length,
                                             k * context_length + padding_size);
              Tensor w_sub = padding_data->Slice(k, k + padding_size);
              blas.AXPY(w_sub.numel(), static_cast<T>(1), out_t_sub.data<T>(),
                        w_sub.data<T>());
            }
          }
          if (down_pad > 0) {
            int down_pad_begin_row =
                std::max(
                    0, (sequence_height - context_start - context_length) + 1) +
                1;
            int padding_begin = std::max(0, context_start - sequence_height);
            int padding_size =
                sequence_height - context_start >= context_length
                    ? 1
                    : context_length - (sequence_height - context_start);
            if (context_start >= sequence_height) padding_size = context_length;
            int padding_idx = padding_begin;
            for (int t = 0; t + down_pad_begin_row <= sequence_height;
                 ++t, ++padding_size) {
              if (context_start >= sequence_height)
                padding_size = context_length;
              if (padding_size > context_length) {
                padding_size = context_length;
                padding_idx++;
              }
              if (padding_begin > 0 || sequence_height == context_start)
                padding_idx = padding_begin + t;

              Tensor out_t_sub = out_t.Slice(
                  (down_pad_begin_row + t) * context_length - padding_size,
                  (down_pad_begin_row + t) * context_length);
              Tensor w_sub = padding_data->Slice(
                  up_pad + padding_idx, up_pad + padding_idx + padding_size);
              blas.AXPY(w_sub.numel(), static_cast<T>(1), out_t_sub.data<T>(),
                        w_sub.data<T>());
            }
          }
          out_t.Resize({sequence_height,
                        static_cast<int64_t>(context_length) * sequence_width});
        }
      }
    }
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
