// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <torch/extension.h>

#define CHECK_CUDA_ERROR()                                                     \
  {                                                                            \
    auto err = cudaGetLastError();                                             \
    if (err != cudaSuccess) {                                                  \
      std::ostringstream oss;                                                  \
      oss << " line:" << __LINE__ << " with err (" << cudaGetErrorName(err)    \
          << ":" << cudaGetErrorString(err);                                   \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  }

#define CHECK_CUDA_ERROR_FUNC(name)                                            \
  {                                                                            \
    auto err = cudaGetLastError();                                             \
    if (err != cudaSuccess) {                                                  \
      std::ostringstream oss(name);                                            \
      oss << " line:" << __LINE__ << " with err (" << cudaGetErrorName(err)    \
          << ":" << cudaGetErrorString(err);                                   \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  }

#define CUSTOM_ENFORCE(cond, msg)                                              \
  if (!(cond)) {                                                               \
    throw std::runtime_error(msg);                                             \
  }

template <typename T> struct PTTypeInfo {};
template <> struct PTTypeInfo<float> {
  static constexpr auto type = torch::kFloat32;
};
template <> struct PTTypeInfo<int8_t> {
  static constexpr auto type = torch::kInt8;
};
template <> struct PTTypeInfo<Ort::Float16_t> {
  static constexpr auto type = torch::kFloat16;
};

namespace Cuda {

void RegisterOps(Ort::CustomOpDomain &domain);

}