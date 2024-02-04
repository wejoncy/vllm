#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include <torch/extension.h>
namespace torch_ext {
torch::Tensor paged_attention_forwad(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key,
    torch::Tensor &value, c10::optional<torch::Tensor> &key_cache_op,
    c10::optional<torch::Tensor> &value_cache_op, int64_t t_input_metadata,
    float scale, c10::optional<torch::Tensor> alibi_slopes, int num_head,
    int num_kv_head, int head_size);
void reset_ort_input_metadata();
} // namespace torch_ext


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // vLLM custom ops
  pybind11::module ops = m.def_submodule("ops", "vLLM custom operators");

  // Attention ops
  ops.def(
    "paged_attention_v1",
    &paged_attention_v1,
    "Compute the attention between an input query and the cached keys/values using PagedAttention.");
  ops.def(
    "paged_attention_v2",
    &paged_attention_v2,
    "PagedAttention V2.");

  // Activation ops
  ops.def(
    "silu_and_mul",
    &silu_and_mul,
    "Activation function used in SwiGLU.");
  ops.def(
    "gelu_new",
    &gelu_new,
    "GELU implementation used in GPT-2.");
  ops.def(
    "gelu_fast",
    &gelu_fast,
    "Approximate GELU implementation.");

  // Layernorm
  ops.def(
    "rms_norm",
    &rms_norm,
    "Apply Root Mean Square (RMS) Normalization to the input tensor.");

  ops.def(
    "fused_add_rms_norm",
    &fused_add_rms_norm,
    "In-place fused Add and RMS Normalization");

  // Rotary embedding
  ops.def(
    "rotary_embedding",
    &rotary_embedding,
    "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");

#ifndef USE_ROCM
  // Quantization ops
  ops.def("awq_gemm", &awq_gemm, "Quantized GEMM for AWQ");
  ops.def("awq_dequantize", &awq_dequantize, "Dequantization for AWQ");
#endif
  ops.def("gptq_gemm", &gptq_gemm, "Quantized GEMM for GPTQ");
  ops.def("gptq_shuffle", &gptq_shuffle, "Post processing for GPTQ");
  ops.def("squeezellm_gemm", &squeezellm_gemm, "Quantized GEMM for SqueezeLLM");
  ops.def(
    "moe_align_block_size",
    &moe_align_block_size,
    "Aligning the number of tokens to be processed by each expert such that it is divisible by the block size.");

  // Cache ops
  pybind11::module cache_ops = m.def_submodule("cache_ops", "vLLM cache ops");
  cache_ops.def(
    "swap_blocks",
    &swap_blocks,
    "Swap in (out) the cache blocks from src to dst");
  cache_ops.def(
    "copy_blocks",
    &copy_blocks,
    "Copy the cache blocks from src to dst");
  cache_ops.def(
    "reshape_and_cache",
    &reshape_and_cache,
    "Reshape the key and value tensors and cache them");
  cache_ops.def(
    "gather_cached_kv",
    &gather_cached_kv,
    "Gather key and value from the cache into contiguous QKV tensors");
  cache_ops.def(
    "convert_fp8_e5m2",
    &convert_fp8_e5m2,
    "Convert the key and value cache to fp8_e5m2 data type");

  // Cuda utils
  pybind11::module cuda_utils = m.def_submodule("cuda_utils", "vLLM cuda utils");
  cuda_utils.def(
    "get_device_attribute",
    &get_device_attribute,
    "Gets the specified device attribute.");

  cuda_utils.def(
    "get_max_shared_memory_per_block_device_attribute",
    &get_max_shared_memory_per_block_device_attribute,
    "Gets the maximum shared memory per block device attribute.");

#ifndef USE_ROCM
  // Custom all-reduce kernels
  pybind11::module custom_ar = m.def_submodule("custom_ar", "custom allreduce");
  custom_ar.def("init_custom_ar", &init_custom_ar, "init_custom_ar");
  custom_ar.def("should_custom_ar", &should_custom_ar, "should_custom_ar");
  custom_ar.def("all_reduce_reg", &all_reduce_reg, "all_reduce_reg");
  custom_ar.def("all_reduce_unreg", &all_reduce_unreg, "all_reduce_unreg");
  custom_ar.def("dispose", &dispose, "dispose");
  custom_ar.def("meta_size", &meta_size, "meta_size");
  custom_ar.def("register_buffer", &register_buffer, "register_buffer");
  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta,
                "get_graph_buffer_ipc_meta");
  custom_ar.def("register_graph_buffers", &register_graph_buffers,
                "register_graph_buffers");
#endif

    
  pybind11::module ort_ops = m.def_submodule("ort_ops", "vLLM paged_attn ops");
  ort_ops.def("PagedAttention", &torch_ext::paged_attention_forwad,
        "torch::Tensor paged_attention_forwad("
        "torch::Tensor out,"
        "const torch::Tensor &query, const torch::Tensor &key,"
        "const torch::Tensor &value, const torch::Tensor &key_cache,"
        "const torch::Tensor &value_cache, const torch::Tensor &input_metadata,"
        "const torch::Tensor &positions, const torch::Tensor &cos_sin_cache,"
        "std::string mask_type, float scale,"
        "const c10::optional<torch::Tensor> &alibi_slopes, int num_head,"
        "int num_kv_head, int head_size) ");
  ort_ops.def("reset_ort_input_metadata", &torch_ext::reset_ort_input_metadata,
              "void reset_ort_input_metadata()");
}
