#pragma once
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime.h" // must include this at first place
#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <type_traits>

#include <torch/extension.h>

using namespace Ort::Custom;


namespace Cuda {

struct AttnParam {
  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t head_size_;
  int64_t num_queries_per_kv_;
  float scale_;
  std::string mask_type_;
  torch::Tensor head_mapping_;
};

struct KernelPagedAttentionOp {
  const OrtApi &_api;
  const OrtKernelInfo *_info;
  AttnParam attn_param_;

  KernelPagedAttentionOp(const OrtApi &api, const OrtKernelInfo *info);

  void Compute(OrtKernelContext *context);
};
// legacy custom op registration
struct PagedAttentionOp
    : Ort::CustomOpBase<PagedAttentionOp, KernelPagedAttentionOp> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
    return std::make_unique<KernelPagedAttentionOp>(api, info).release();
  };
  const char *GetName() const { return "PagedAttention"; };
  const char *GetExecutionProviderType() const {
  #ifndef USE_ROCM
    return "CUDAExecutionProvider";
  #else
    return "ROCMExecutionProvider";
  #endif
  };
  size_t GetInputTypeCount() const { return 6; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 5 || index == 6) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    // if constexpr (std::is_same<T, float>::value) {
    //   return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    // } else if constexpr (std::is_same<T, Ort::Float16_t>::value) {
    //   return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    // } else if constexpr (std::is_same<T, int8_t>::value) {
    //   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    // } else {
    //   throw std::runtime_error("Unsupported type");
    // }
  };
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    // if constexpr (std::is_same<T, float>::value) {
    //   return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    // } else if constexpr (std::is_same<T, Ort::Float16_t>::value) {
    //   return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    // } else if constexpr (std::is_same<T, int8_t>::value) {
    //   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    // } else {
    //   throw std::runtime_error("Unsupported type");
    // }
  };
};
const PagedAttentionOp pageattn;

} // namespace Cuda
