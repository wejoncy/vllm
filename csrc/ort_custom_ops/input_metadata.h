// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
//#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>


// The three struct is used to represent InputMedata in the python side.
struct AttnBias {
  typedef struct {
    int64_t seqstart;
    int64_t max_seqlen;
    int64_t seqstart_py;
  } block_tables;
  block_tables k_seqinfo;
  block_tables q_seqinfo;
  int64_t batchsize;
  const char *attn_name;
};

struct THEvent {
  cudaEvent_t events[128]; // assume we have at most 128 layers.
};

struct InputMetadata {
  int64_t schedule_type; // 0: vllm. 1:sarathi, 2:custom, 3:self-build
  int64_t block_tables;
  int64_t max_num_blocks_per_seq;
  int64_t context_lens;
  int64_t max_context_len;
  int64_t is_prompt;
  int64_t block_tables_size_1;
  int64_t slot_mapping;
  int64_t context_lens_size_1;
  AttnBias attn_bias;
  THEvent cache_events;
  cudaStream_t cache_stream;
};
