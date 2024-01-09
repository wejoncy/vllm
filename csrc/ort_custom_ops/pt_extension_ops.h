
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnxruntime.h" // must include this at first place
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <unordered_map>

#include <torch/extension.h>
using namespace Ort::Custom;

namespace Cuda {
struct PTExtensionParam {
  int64_t num_inputs;
  int64_t num_outputs;
  std::string func_name;
  std::unordered_map<std::string, std::string> attr_dict_;
};

struct TorchExtensionKernel {
  const OrtApi &_api;
  const OrtKernelInfo *_info;
  PTExtensionParam ext_param_;

  TorchExtensionKernel(const OrtApi &api, const OrtKernelInfo *info);

  void Compute(OrtKernelContext *context);
};

// legacy custom op registration
struct TorchExtensionOp
    : Ort::CustomOpBase<TorchExtensionOp, TorchExtensionKernel> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
    return std::make_unique<TorchExtensionKernel>(api, info).release();
  };
  const char *GetName() const { return "TorchExtension"; };
  const char *GetExecutionProviderType() const {
    return "CUDAExecutionProvider";
  };
  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  constexpr bool GetVariadicInputHomogeneity() const noexcept {
    return false; // heterogenous
  }

  constexpr bool GetVariadicOutputHomogeneity() const noexcept {
    return false; // heterogeneous
  }
  constexpr OrtCustomOpInputOutputCharacteristic
  GetInputCharacteristic(size_t /* index */) const noexcept {
    return INPUT_OUTPUT_VARIADIC;
  }

  constexpr OrtCustomOpInputOutputCharacteristic
  GetOutputCharacteristic(size_t /* index */) const noexcept {
    return INPUT_OUTPUT_VARIADIC;
  }
};

} // namespace cuda
