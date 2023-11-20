// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "pt_extension_ops.h"
#include "cuda_context.h"
#include "external_torch_func_declare.h"
#include "raw_tensor.h"
#include "cuda_ops.h"
#include "pt_to_ort_resource.h"
#include "utils.h"


namespace Cuda {

std::vector<std::string> splitString(const std::string input,
                                     const char delimiter) {
  std::vector<std::string> elements;
  std::stringstream stream(input);
  std::string element;

  while (getline(stream, element, delimiter)) {
    elements.push_back(element);
  }

  return elements;
}

void DebugIntoPython(const torch::Tensor &input){
  GilGuard guard;
  auto py_module = py::module::import("vllm.model_executor.ort_backend.onnx_debug");
  py_module.attr("Debug")(input);
}

TorchExtensionKernel::TorchExtensionKernel(
    const OrtApi &api, const OrtKernelInfo *info)
    : _api(api), _info(info) {
  int64_t num_inputs = 0;
  int64_t num_outputs = 0;
  CUSTOM_ENFORCE(api.KernelInfoGetAttribute_int64(info, "num_inputs",
                                                  &num_inputs) == nullptr &&
                     num_inputs > 0,
                 "num_inputs attribute is not set");
  ext_param_.num_inputs = static_cast<int32_t>(num_inputs);

  CUSTOM_ENFORCE(api.KernelInfoGetAttribute_int64(info, "num_outputs",
                                                  &num_outputs) == nullptr &&
                     num_outputs > 0,
                 "num_outputs attribute is not set");
  ext_param_.num_outputs = static_cast<int32_t>(num_outputs);


  std::string extra_attributes(1024, '0');
  size_t extra_attributes_size = 0;
  api.KernelInfoGetAttribute_string(info, "extra_attributes", extra_attributes.data(),
                     &extra_attributes_size);
  api.KernelInfoGetAttribute_string(info, "extra_attributes", extra_attributes.data(),
                     &extra_attributes_size);
  if (extra_attributes_size>0) {
    extra_attributes = extra_attributes.substr(0, extra_attributes_size);
    auto extra_attributes_vec =
        splitString(extra_attributes.substr(0, extra_attributes_size));
    for (auto attr : extra_attributes_vec) {
      auto attr_vec = splitString(attr, '=');
      CUSTOM_ENFORCE(attr_vec.size() == 2,
                    "extra_attributes should be in format of key=value, but got "+extra_attributes);
      ext_param_.attr_dict_[std::string(attr_vec[0])] = std::string(attr_vec[1]);
    }
  }

  ext_param_.func_name = std::string(128, '0');
  size_t func_name_size = 0;
  api.KernelInfoGetAttribute_string(
      info, "func_name", ext_param_.func_name.data(), &func_name_size);
  CUSTOM_ENFORCE(api.KernelInfoGetAttribute_string(info, "func_name",
                                                   ext_param_.func_name.data(),
                                                   &func_name_size) == nullptr,
                 "func_name attribute is not set");
  ext_param_.func_name = ext_param_.func_name.substr(0, func_name_size - 1);
};

void TorchExtensionKernel::Compute(OrtKernelContext *context) {
  Ort::Custom::CudaContext cuda_ctx;
  cuda_ctx.Init(*context);
  Ort::KernelContext ctx(context);

  using T = Ort::Float16_t;
  const char* LOCAL_RANK = getenv("LOCAL_RANK");
  int device_id = LOCAL_RANK ? std::stoi(LOCAL_RANK) : 0;
  at::cuda::CUDAStream myStream =
      at::cuda::getStreamFromExternal(cuda_ctx.cuda_stream, device_id);
  at::cuda::setCurrentCUDAStream(myStream);

  // get input tensors
  std::vector<Ort::ConstValue> input_ort_tensors(ext_param_.num_inputs);
  std::vector<torch::Tensor> input_torch_tensors(ext_param_.num_inputs);
  for (int i = 0; i < ext_param_.num_inputs; ++i) {
    input_ort_tensors[i] = ctx.GetInput(i);
    auto itype =
        input_ort_tensors[i].GetTensorTypeAndShapeInfo().GetElementType();
    auto pt_options = torch::TensorOptions()
              .device(torch::kCUDA, device_id);
    if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      pt_options = pt_options.dtype(torch::kFloat32);
    }else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      pt_options = pt_options.dtype(torch::kFloat16);
    }else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
      pt_options = pt_options.dtype(torch::kFloat64);
    }else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      pt_options = pt_options.dtype(torch::kInt32);
    }else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      pt_options = pt_options.dtype(torch::kInt64);
    }else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      pt_options = pt_options.dtype(torch::kUInt8);
    }else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
      pt_options = pt_options.dtype(torch::kInt8);
    }else if (itype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
      pt_options = pt_options.dtype(torch::kBool);
    } else {
      CUSTOM_ENFORCE(false, "Unsupported data type: " + std::to_string(itype));
    }
    input_torch_tensors[i] = torch::from_blob(
        const_cast<void *>(input_ort_tensors[i].GetTensorRawData()),
        input_ort_tensors[i].GetTensorTypeAndShapeInfo().GetShape(),
        pt_options);
  }

  if (ext_param_.func_name == "silu_and_mul") {
    auto output_shape = input_torch_tensors[0].sizes().vec();
    output_shape[output_shape.size() - 1] /= 2;
    auto output_tensor = ctx.GetOutput(0, output_shape);
    auto output_torch = torch::from_blob((output_tensor.GetTensorMutableData<T>()),
                     output_tensor.GetTensorTypeAndShapeInfo().GetShape(),
                     torch::TensorOptions()
                         .device(torch::kCUDA, device_id)
                         .dtype(PTTypeInfo<T>::type));
    silu_and_mul(output_torch, input_torch_tensors[0]);
  } else if (ext_param_.func_name == "rms_norm") {
    const float epsilon = std::stof(ext_param_.attr_dict_[("variance_epsilon")]);
    auto output_shape = input_torch_tensors[0].sizes().vec();
    auto output_tensor = ctx.GetOutput(0, output_shape);
    auto output_torch =
        torch::from_blob((output_tensor.GetTensorMutableData<T>()),
                         output_tensor.GetTensorTypeAndShapeInfo().GetShape(),
                         torch::TensorOptions()
                             .device(torch::kCUDA, device_id)
                             .dtype(PTTypeInfo<T>::type));
    rms_norm(output_torch, input_torch_tensors[0], input_torch_tensors[1],
             epsilon);
  } else if (ext_param_.func_name == "fused_add_rms_norm") {
    const float epsilon = std::stof(ext_param_.attr_dict_[("variance_epsilon")]);
    auto output_shape = input_torch_tensors[0].sizes().vec();
    auto output_hiddenstate = ctx.GetOutput(0, output_shape);
    auto output_residual = ctx.GetOutput(1, output_shape);
    auto output_torch_hiddenstate = torch::from_blob(
        (output_hiddenstate.GetTensorMutableData<T>()),
        output_hiddenstate.GetTensorTypeAndShapeInfo().GetShape(),
        torch::TensorOptions()
            .device(torch::kCUDA, device_id)
            .dtype(PTTypeInfo<T>::type));
    auto output_torch_residual =
        torch::from_blob((output_residual.GetTensorMutableData<T>()),
                         output_residual.GetTensorTypeAndShapeInfo().GetShape(),
                         torch::TensorOptions()
                             .device(torch::kCUDA, device_id)
                             .dtype(PTTypeInfo<T>::type));
    auto a = c10::make_optional(output_torch_hiddenstate);
    auto b = c10::make_optional(output_torch_residual);
    fused_add_rms_norm(a, b, input_torch_tensors[0], input_torch_tensors[1],
                       input_torch_tensors[2], epsilon);
  } else if (ext_param_.func_name == "rotary_embedding") {
    const int32_t head_size = std::stoi(ext_param_.attr_dict_[("head_size")]);
    const int32_t is_neox = std::stoi(ext_param_.attr_dict_[("is_neox_style")]);
    auto output_query = ctx.GetOutput(0, input_torch_tensors[1].sizes().vec());
    auto output_key = ctx.GetOutput(1, input_torch_tensors[2].sizes().vec());
    auto output_query_torch = torch::from_blob(
        (output_query.GetTensorMutableData<T>()),
        output_query.GetTensorTypeAndShapeInfo().GetShape(),
        torch::TensorOptions()
            .device(torch::kCUDA, device_id)
            .dtype(PTTypeInfo<T>::type));
    auto output_key_torch =
        torch::from_blob((output_key.GetTensorMutableData<T>()),
                         output_key.GetTensorTypeAndShapeInfo().GetShape(),
                         torch::TensorOptions()
                             .device(torch::kCUDA, device_id)
                             .dtype(PTTypeInfo<T>::type));
    auto a = c10::make_optional(output_query_torch);
    auto b = c10::make_optional(output_key_torch);
    rotary_embedding(input_torch_tensors[0], input_torch_tensors[1],
                     input_torch_tensors[2], head_size, input_torch_tensors[3],
                     is_neox, a, b);
  } else if (ext_param_.func_name == "debug_step") {
    auto output_shape = input_torch_tensors[0].sizes().vec();
    auto output_tensor = ctx.GetOutput(0, output_shape);
    auto output_torch =
        torch::from_blob((output_tensor.GetTensorMutableData<T>()),
                         output_tensor.GetTensorTypeAndShapeInfo().GetShape(),
                         torch::TensorOptions()
                             .device(torch::kCUDA, device_id)
                             .dtype(PTTypeInfo<T>::type));
    DebugIntoPython(input_torch_tensors[0]);
    cudaMemcpyAsync(output_torch.data_ptr(), input_torch_tensors[0].data_ptr(),
                    output_torch.numel() * sizeof(T), cudaMemcpyDeviceToDevice,
                    cuda_ctx.cuda_stream);
  } else {
    CUSTOM_ENFORCE(false, "Unsupported function: " + ext_param_.func_name);
  }
  CHECK_CUDA_ERROR_FUNC(ext_param_.func_name);
  return;
}

} // namespace Cuda

namespace torch_ext{

}