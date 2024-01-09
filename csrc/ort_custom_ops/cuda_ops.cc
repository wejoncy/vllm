// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime.h" // must include this at first place
#include <torch/extension.h>

#include "pt_extension_ops.h"
#include "paged_attn_ops.h"


namespace Cuda {
void RegisterOps(Ort::CustomOpDomain &domain) {
  static const PagedAttentionOp pageattn;
  static const TorchExtensionOp ptext;
  domain.Add(&pageattn);
  domain.Add(&ptext);
}

} // namespace Cuda

namespace torch_ext {
torch::Tensor paged_attention_forwad(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key,
    torch::Tensor &value, c10::optional<torch::Tensor> &key_cache_op,
    c10::optional<torch::Tensor> &value_cache_op, int64_t t_input_metadata,
    float scale, c10::optional<torch::Tensor> alibi_slopes, int num_head, int num_kv_head,
    int head_size);
void reset_ort_input_metadata();
}


//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  m.def("PagedAttention", &torch_ext::paged_attention_forwad,
//        "torch::Tensor paged_attention_forwad("
//        "torch::Tensor out,"
//        "const torch::Tensor &query, const torch::Tensor &key,"
//        "const torch::Tensor &value, const torch::Tensor &key_cache,"
//        "const torch::Tensor &value_cache, const torch::Tensor &input_metadata,"
//        "const torch::Tensor &positions, const torch::Tensor &cos_sin_cache,"
//        "std::string mask_type, float scale,"
//        "const c10::optional<torch::Tensor> &alibi_slopes, int num_head,"
//        "int num_kv_head, int head_size) ");
//  m.def("reset_ort_input_metadata", &torch_ext::reset_ort_input_metadata, "void reset_ort_input_metadata()");
//}
