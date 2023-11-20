// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma  once

#include "input_metadata.h"

namespace Cuda {
class Resource {
  public:
  InputMetadata* input_metadata = nullptr;
  float norm_variance_epsilon = 1e-5;
public:
    ~Resource(){}
    Resource(const Resource&)=delete;
    Resource& operator=(const Resource&)=delete;
    static Resource& get_instance(){
        static Resource instance;
        return instance;

    }
private:
    Resource(){}
};
}
