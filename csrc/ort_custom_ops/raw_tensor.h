// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnxruntime.h"

#include <cstdint>
#include <type_traits>
#include <vector>


using TensorShape = std::vector<int64_t>;
#define ORT_DISALLOW_COPY(TypeName) TypeName(const TypeName &) = delete

#define ORT_DISALLOW_ASSIGNMENT(TypeName)                                      \
  TypeName &operator=(const TypeName &) = delete

#define ORT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName)                             \
  ORT_DISALLOW_COPY(TypeName);                                                 \
  ORT_DISALLOW_ASSIGNMENT(TypeName)
class RawTensor final {
public:
  RawTensor() = default; // to allow creating vector<RawTensor> to support seq(tensor)

  template <typename T>
  RawTensor(const TensorShape &shape, const T *p_data) {
    shape_ = shape;
    p_data_ = const_cast<T *>(p_data);
    set_dtype<T>();
  }
  template <typename T> RawTensor(const TensorShape &shape, T *p_data) {
    shape_ = shape;
    p_data_ = static_cast<void *>(p_data);
    set_dtype<T>();
  }
  ~RawTensor()=default;

  // Move is allowed
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(RawTensor);

  RawTensor(RawTensor &&other) noexcept;
  RawTensor &operator=(RawTensor &&other) noexcept;

  template<typename T>
  void set_dtype() {
    if (std::is_same<T, float>::value){
      dtype_ = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }else if (std::is_same<T, int8_t>::value){
      dtype_ = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    } else if (std::is_same<T, double>::value) {
      dtype_ = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    } else if (std::is_same<T, Ort::Float16_t>::value) {
      dtype_ = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    } else if (std::is_same<T, Ort::BFloat16_t>::value) {
      dtype_ = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    }else{
        //ORT_THROW("Unsupported data type: ", typeid(T).name());
        throw std::runtime_error("Unsupported data type: ");
    }
  }
  /**
     Returns the data type enum constant
     @remarks Use utils::ToTensorProtoElementType<T> for comparison.
  */
  int32_t GetElementType() const { return dtype_; }

  /**
     Returns the shape of the tensor.
  */
  const TensorShape &Shape() const noexcept { return shape_; }


  template <typename T> const T *Data() const {
    return reinterpret_cast<const T *>(static_cast<char *>(p_data_) +
                                       byte_offset_);
  }

  void *MutableDataRaw() noexcept {
    return static_cast<char *>(p_data_) + byte_offset_;
  }

  const void *DataRaw() const noexcept {
    return static_cast<char *>(p_data_) + byte_offset_;
  }


  // More API methods.
private:
#ifdef ENABLE_STRIDED_TENSORS
  bool CheckIsContiguous() const;
#endif

  void *p_data_;
  int64_t byte_offset_=0;
  TensorShape shape_;
  ONNXTensorElementDataType dtype_;
};
