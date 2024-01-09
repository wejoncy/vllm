// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <algorithm>

#include <torch/extension.h>

namespace Cuda {
std::vector<std::string> splitString(const std::string input, const char delimiter='\n');
}

class GilGuard {
  public:
    GilGuard() : state_(PyGILState_Ensure()){};
    ~GilGuard() { PyGILState_Release(state_); };

  private:
    PyGILState_STATE state_;
};