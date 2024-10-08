// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime.h"

#include "custom_op_library.h"

#include <vector>
#include <cmath>
#include <mutex>
#include <system_error>

#include "paged_attn_ops.h"
#include "pt_extension_ops.h"

static const char* c_OpDomain = "vllm.ort.ext";

#define ORT_TRY try
#define ORT_CATCH(x) catch (x)
#define ORT_RETHROW throw;
#define ORT_HANDLE_EXCEPTION(func) func()

namespace Cuda {
void RegisterOps(Ort::CustomOpDomain &domain) {
  static const PagedAttentionOp pageattn;
  static const TorchExtensionOp ptext;
  domain.Add(&pageattn);
  domain.Add(&ptext);
}

} // namespace Cuda

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);
  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};
    Cuda::RegisterOps(domain);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);

    AddOrtCustomOpDomainToContainer(std::move(domain));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      Ort::Status status{e};
      result = status.release();
    });
  }
  return result;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}
