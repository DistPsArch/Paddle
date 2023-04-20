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

#include "paddle/fluid/operators/weighted_random_sample_op.h"
#include <string>

namespace paddle {
namespace operators {

class WeightedRandomSampleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {

    PADDLE_ENFORCE_EQ(
        ctx->Inputs("SelfInput").size(), 1UL,
        platform::errors::InvalidArgument(
            "Input(SelfInput) of WeightedRandomSampleOp should not be empty."));
    PADDLE_ENFORCE_EQ(
        ctx->Inputs("OtherInput").size(), 1UL,
        platform::errors::InvalidArgument(
            "Input(OtherInput) of WeightedRandomSampleOp should not be empty."));
    PADDLE_ENFORCE_EQ(
        ctx->Inputs("LabelInput").size(), 1UL,
        platform::errors::InvalidArgument(
            "Input(LabelInput) of WeightedRandomSampleOp should not be empty."));
    PADDLE_ENFORCE_EQ(
        ctx->Inputs("FeatureInput").size(), 1UL,
        platform::errors::InvalidArgument(
            "Input(FeatureInput) of WeightedRandomSampleOp should not be empty."));
    //PADDLE_ENFORCE_EQ(
    //    ctx->Inputs("RandomRematch").size(), 1UL,
    //    platform::errors::InvalidArgument(
    //        "Input(RandomRematch) of WeightedRandomSampleOp should not be empty."));
    //PADDLE_ENFORCE_EQ(
    //    ctx->Inputs("RandomTargetLabel").size(), 1UL,
    //    platform::errors::InvalidArgument(
    //        "Input(RandomRematch) of WeightedRandomSampleOp should not be empty."));
    PADDLE_ENFORCE_EQ(
        ctx->Outputs("Out").size(), 1UL,
        platform::errors::InvalidArgument(
            "Output(Out) of WeightedRandomSampleOp should not be empty."));
     
    auto self_input_dims = ctx->GetInputDim("SelfInput");
    auto other_input_dims = ctx->GetInputDim("OtherInput");
    auto feature_input_dims = ctx->GetInputDim("FeatureInput");
    auto label_input_dims = ctx->GetInputDim("LabelInput");
    // auto random_rematch_dims = ctx->GetInputDim("RandomRematch");
    // auto random_target_label_dims = ctx->GetInputDim("RandomTargetLabel");


    PADDLE_ENFORCE_EQ(
         self_input_dims.size(), 2UL,
         platform::errors::InvalidArgument("Input(SelfInput)'s rank should be 2."));
    PADDLE_ENFORCE_EQ(
         other_input_dims.size(), 2UL,
         platform::errors::InvalidArgument("Input(OtherInput)'s rank should be 2."));
    PADDLE_ENFORCE_EQ(
         label_input_dims.size(), 2UL,
         platform::errors::InvalidArgument("Input(labelInput)'s rank should be 2."));
    PADDLE_ENFORCE_EQ(
         feature_input_dims.size(), 2UL,
         platform::errors::InvalidArgument("Input(FeatureInput)'s rank should be 2."));
    // PADDLE_ENFORCE_EQ(
    //     random_rematch_dims.size(), 2UL,
    //     platform::errors::InvalidArgument("Input(RandomRematch)'s rank should be 2."));
    // PADDLE_ENFORCE_EQ(
    //     random_target_label_dims.size(), 2UL,
    //     platform::errors::InvalidArgument("Input(RandomTargetLabel)'s rank should be 2."));

    PADDLE_ENFORCE_EQ(self_input_dims[1], other_input_dims[1], platform::errors::InvalidArgument(
                                            "The 2nd dimension of "
                                            "Input(SelfInput) and Input(OtherInput) should be same."));
        
    PADDLE_ENFORCE_EQ(self_input_dims[0], label_input_dims[0], platform::errors::InvalidArgument(
                                            "The 1st dimension of "
                                            "Input(SelfInput) and Input(LabelInput) should be same."));




    // auto cvm_dims = ctx->GetInputDim("CVM");
    // PADDLE_ENFORCE_EQ(
    //    cvm_dims.size(), 2UL,
    //    platform::errors::InvalidArgument("Input(CVM)'s rank should be 2."));

    // PADDLE_ENFORCE_EQ(cvm_dims[1], 2UL, platform::errors::InvalidArgument(
    //                                        "The 2nd dimension of "
    //                                        "Input(CVM) should be 2."));

    // auto ins_dims = ctx->GetInputsDim("X");
    // const int cvm_offset = ctx->Attrs().Get<int>("cvm_offset");

    const size_t num_outputs = 3;
    // const size_t num_inputs = 3;
    std::vector<framework::DDim> outs_dims;
    outs_dims.resize(num_outputs);
    int random_rematch_ratio = ctx->Attrs().Get<int>("random_rematch_ratio");

    // PADDLE_ENFORCE_GT(num_inputs, 0UL,
    //                  platform::errors::InvalidArgument(
    //                      "Input tensors count should be greater than 0, "
    //                      "but received value is %d.",
    //                      num_inputs));

    // The output height should be confirmed in Compute,
    // since input lod is not accessible here.
    // PADDLE_ENFORCE_EQ(ins_dims[0].size(), 2,
    //                  platform::errors::InvalidArgument(
    //                      "The dims size of first input should be equal to 2, "
    //                      "but received value is %d.",
    //                      ins_dims[0].size()));

    // async infer shape 
    if (ctx->IsRuntime()) {
      int batch_size = -1;
      auto self_input_tensor = ctx->GetInputVarPtrs("SelfInput");
      auto other_input_tensor = ctx->GetInputVarPtrs("OtherInput");
      std::vector<::paddle::framework::InferShapeVarPtr> inputs_tensor{self_input_tensor[0],
                                                                       other_input_tensor[0]};
      for (size_t i = 0; i < inputs_tensor.size(); ++i) {
        int cur_batch_size = 0;
        framework::Variable* x_var =
          BOOST_GET(framework::Variable*, inputs_tensor[i]);
        // get lod info
        const auto& x_tensor = x_var->Get<LoDTensor>();
        const auto& x_lod = x_tensor.lod();
        if (x_lod.size() > 0) {
          cur_batch_size = x_lod[0].size() - 1;
        } else {
          cur_batch_size = x_tensor.dims()[0];
        }
        if (batch_size == -1) {
          batch_size = cur_batch_size;
        } else {
          PADDLE_ENFORCE_EQ(batch_size, cur_batch_size,
                            platform::errors::PreconditionNotMet(
                                "The batch size of all input should be same, "
                                "please check, last batch_size is %d, current "
                                "batch_size is %d",
                                batch_size, cur_batch_size));
        }
        // if (use_cvm) {
        //  out_dim = {batch_size * , dims[rank - 1]};
        // } else {
        //  out_dim = {batch_size, dims[rank - 1] - cvm_offset};
        //}
        // outs_dims[i] = phi::make_ddim(out_dim);
      }

      int rank = self_input_dims.size();
      std::vector<int64_t> out_dim_vec_0{batch_size * random_rematch_ratio, self_input_dims[rank - 1]};
      std::vector<int64_t> out_dim_vec_1{batch_size * random_rematch_ratio, 1};

      outs_dims[0] = phi::make_ddim(out_dim_vec_0);
      outs_dims[1] = phi::make_ddim(out_dim_vec_1);
      outs_dims[2] = phi::make_ddim(out_dim_vec_1);

    } else {
      // for (size_t i = 0; i < num_inputs; ++i) {
        // const auto dims = ins_dims[i];
        // int rank = dims.size();
        // if (use_cvm) {
        //  PADDLE_ENFORCE_GT(
        //      dims[rank - 1], 2,
        //      platform::errors::InvalidArgument(
        //          "Shape error in %lu id, the last dimension(embedding) of the "
        //          "'X' tensor must be larger than 2.",
        //          i));
        //}
        // std::vector<int64_t> out_dim;
        // if (use_cvm) {
        //  out_dim = {-1, dims[rank - 1]};
        // } else {
        //  out_dim = {-1, dims[rank - 1] - cvm_offset};
        // }
        // outs_dims[i] = phi::make_ddim(out_dim);
      // }
      int rank = self_input_dims.size();
      std::vector<int64_t> out_dim_vec_0{-1, self_input_dims[rank - 1]};
      std::vector<int64_t> out_dim_vec_1{-1, 1};
      outs_dims[0] = phi::make_ddim(out_dim_vec_0);
      outs_dims[1] = phi::make_ddim(out_dim_vec_1);
      outs_dims[2] = phi::make_ddim(out_dim_vec_1);
    }
    ctx->SetOutputDim("Out", outs_dims[0]);
    ctx->SetOutputDim("RandomRematch",outs_dims[1]);
    ctx->SetOutputDim("RandomLabelTarget", outs_dims[2]);
    // "check"
    // ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto* self_input = ctx.Input<framework::Tensor>("SelfInput");
    auto* other_input = ctx.Input<framework::Tensor>("OtherInput");

    std::vector<const framework::Tensor*> inputs{self_input, other_input};

    auto input_data_type = framework::proto::VarType::Type(0);
    bool flag = 0;
    for (auto* input : inputs) {
      if (input->IsInitialized() && input->numel() > 0) {
        input_data_type = framework::TransToProtoVarType(input->dtype());
        flag = 1;
        break;
      }
    }
    PADDLE_ENFORCE_EQ(flag, 1,
                      platform::errors::InvalidArgument(
                          "All Inputs of weighted_random_sample OP are Empty!"));
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
    // return framework::OpKernelType(framework::proto::VarType::FP32,
    //                                ctx.device_context());
    // return framework::OpKernelType(
    //   OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class WeightedRandomSampleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("SelfInput",
             "(vector<Tensor>) The self_input tensor of"
             " operator.")
        .AsDuplicable();
    AddInput("OtherInput",
             "(vector<Tensor>) The other_input tensor of"
             " operator.")
        .AsDuplicable();
    AddInput("FeatureInput",
             "(vector<Tensor>) The feature_input tensor of"
             " operator.")
        .AsDuplicable();
    AddInput("LabelInput",
             "(vector<LoDTensor>) The label_input tensor of"
             " operator.")
        .AsDuplicable();
    AddOutput("Out",
              "(vector<Tensor>) The output of Op does not contain LoD "
              "information.")
        .AsDuplicable();
    AddOutput("RandomLabelTarget",
              "(vector<Tensor>) The output of Op does not contain LoD "
              "information.")
        .AsDuplicable();
    AddOutput("RandomRematch",
              "(vector<Tensor>) The output of Op does not contain LoD "
              "information.")
        .AsDuplicable();

    AddAttr<float>("vec_sim_max",
                   "(float, default 0.7) the max of similarity score.")
        .SetDefault(0.7);
    AddAttr<float>("vec_sim_base",
                   "(float, default 1.0) The base similarity score.")
        .SetDefault(1.0);

    AddAttr<float>("fea_match_base",
                   "(float, default 1.0) The base score for matching feasign.")
        .SetDefault(1.0);

    AddAttr<std::string>("weight_formula",
                   "(std::string, default 'default')")
        .SetDefault("default")
        .InEnum({"default"});

    AddAttr<bool>("do_random", "bool, do random or not").SetDefault(false);
    AddAttr<bool>("need_initialize", "bool, initialize random_rematch or not").SetDefault(false);
    AddAttr<bool>("use_global_random_rematch", "bool, use global random rematch or not").SetDefault(false);
    AddAttr<int>("random_rematch_ratio", "(int, default 1)").SetDefault(1);

    AddComment(R"DOC(
weighted random sample negative ins pair op.
)DOC");
  }
};

class WeightedRandomSampleGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {

    auto og_dim = ctx->GetInputDim(framework::GradVarName("Out"));
    auto self_input_dim = ctx->GetInputDim("SelfInput");

    const int random_rematch_ratio = ctx->Attrs().Get<int>("random_rematch_ratio");
    // bool do_random = ctx->Attrs().Get<bool>("do_random");

    PADDLE_ENFORCE_EQ(
        self_input_dim.size(), 2,
        platform::errors::InvalidArgument("Input(self_input)'s rank should be 2."));

    // check out@grad's row size 
    PADDLE_ENFORCE_EQ(
        og_dim.size(), 2,
        platform::errors::InvalidArgument("Output(Out)'s rank should be 2."));
    // check
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
        og_dim[0], self_input_dim[0] * random_rematch_ratio,
        platform::errors::InvalidArgument(
            "The dimension mismatch between Input(OUT@GRAD) and "
            "Input(X). Received Input(OUT@GRAD): input rank %u, "
            "input shape [%s]; received Input(X): input rank %u, "
            "input shape [%s].",
            og_dim.size(), og_dim, self_input_dim.size(), self_input_dim));
    }
    ctx->ShareDim("SelfInput", framework::GradVarName("SelfInput")); 
    // for (size_t i = 0; i < x_dims.size(); ++i) {
    //  ctx->ShareLoD("X", framework::GradVarName("X"), i, i);
    //  ctx->ShareDim("X", framework::GradVarName("X"), i, i);
    //}

  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class WeightedRandomSampleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op_desc_ptr) const override {
    op_desc_ptr->SetType("weighted_random_sample_grad");

    op_desc_ptr->SetInput("SelfInput", this->Input("SelfInput"));
    // op_desc_ptr->SetInput("CVM", this->Input("CVM"));
    op_desc_ptr->SetInput("RandomRematch", this->Output("RandomRematch"));

    op_desc_ptr->SetInput(framework::GradVarName("Out"),
                          this->OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("SelfInput"),
                           this->InputGrad("SelfInput", false)); // drop_empty_grad=false
    // op_desc_ptr->SetOutput(framework::GradVarName("CVM"),
    //                       this->InputGrad("CVM"));
    op_desc_ptr->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(weighted_random_sample, ops::WeightedRandomSampleOp,
                  ops::WeightedRandomSampleOpMaker,
                  ops::WeightedRandomSampleGradOpMaker<paddle::framework::OpDesc>,
                  ops::WeightedRandomSampleGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(weighted_random_sample_grad, ops::WeightedRandomSampleGradOp)

REGISTER_OP_CPU_KERNEL(weighted_random_sample,
                       ops::WeightedRandomSampleOpCPUKernel<float>)
REGISTER_OP_CPU_KERNEL(weighted_random_sample_grad,
                       ops::WeightedRandomSampleGradOpCPUKernel<float>)
