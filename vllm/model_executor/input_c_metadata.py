from typing import Dict, List, Optional, Tuple
import torch
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                                LowerTriangularMaskWithTensorBias)
import ctypes
from ctypes import (Structure, c_longlong, c_void_p, c_char_p, byref)

from .layers.attention import _make_alibi_bias
from . import InputMetadata
from vllm import paged_attn

class C_AttnBiasBase(Structure):
    _fields_ = [
        ("seqstart", c_longlong),
        ("max_seqlen", c_longlong),
        ("seqstart_py", c_longlong),
    ]
class C_AttnBias(Structure):
    _fields_ = [
        ("k_seqinfo", C_AttnBiasBase),
        ("q_seqinfo", C_AttnBiasBase),
        ("batchsize", c_longlong),
        ("attn_name", c_char_p),
    ]
class C_THEvent(Structure):
    _fields_ = [
        ("cuda_event", c_void_p*128),
    ]
class C_InputMetadata(Structure):
    _fields_ = [
        ("schedule_type", c_longlong),
        ("block_tables", c_longlong),
        ("max_num_blocks_per_seq", c_longlong),
        ("context_lens", c_longlong),
        ("max_context_len", c_longlong),
        ("is_prompt", c_longlong),
        ("block_tables_size_1", c_longlong),
        ("slot_mapping", c_longlong),
        ("context_lens_size_1", c_longlong),
        ("attn_bias", C_AttnBias),
        ("cache_events", C_THEvent),
        ("cache_stream", c_void_p),
    ]

def GetAddrForCStruct(c_struct):
    return (ctypes.cast(byref(c_struct), ctypes.c_void_p).value)

current_input_metadata = None
input_metadata_c = C_InputMetadata()
k_seqinfo = C_AttnBiasBase()
q_seqinfo = C_AttnBiasBase()
attn_bias = C_AttnBias()
event_list_c = C_THEvent()


def set_current_input_metadata(input_metadata:InputMetadata):
    paged_attn.reset_ort_input_metadata()
    global current_input_metadata
    current_input_metadata = input_metadata

# call during ort custom op 
def ConvertInputMetadataToC(input_metadata:InputMetadata = None, 
                            event_list:List[torch.cuda.Event] = None, 
                            alibi_slopes:Optional[torch.Tensor] = None, 
                            sliding_window:Optional[int] = None):
    ###
    input_metadata = current_input_metadata if current_input_metadata is not None else input_metadata
    batch_size, seq_len = input_metadata.num_prompts, input_metadata.max_prompt_len

    if input_metadata.is_prompt and input_metadata.attn_bias is None:
        if alibi_slopes is None:
            pt_attn_bias = BlockDiagonalCausalMask.from_seqlens([seq_len] * batch_size)
            pt_attn_bias.k_seqinfo.to(input_metadata.slot_mapping.device)
            pt_attn_bias.q_seqinfo.to(input_metadata.slot_mapping.device)
            if sliding_window is not None:
                pt_attn_bias = pt_attn_bias.make_local_attention(sliding_window)
            input_metadata.attn_bias = pt_attn_bias
        else:
            input_metadata.attn_bias = _make_alibi_bias(alibi_slopes, batch_size, seq_len, query.dtype)


        attn_bias.attn_name = input_metadata.attn_bias.__class__.__name__.encode('utf-8')

        pt_attn_bias = input_metadata.attn_bias
        if isinstance(pt_attn_bias, LowerTriangularMaskWithTensorBias):
            k_seqinfo.seqstart = None
            k_seqinfo.max_seqlen = -1
            #k_seqinfo.seqstart_py = pt_attn_bias.k_seqinfo.seqstart_py
            q_seqinfo.seqstart = None
            q_seqinfo.max_seqlen = -1
            #q_seqinfo.seqstart_py = pt_attn_bias.q_seqinfo.seqstart_py.data_ptr()
            attn_bias.batchsize = len(input_metadata.prompt_lens)
        elif isinstance(pt_attn_bias, BlockDiagonalCausalMask):
            k_seqinfo.seqstart = pt_attn_bias.k_seqinfo.seqstart.data_ptr()
            k_seqinfo.max_seqlen = pt_attn_bias.k_seqinfo.max_seqlen
            #k_seqinfo.seqstart_py = pt_attn_bias.k_seqinfo.seqstart_py
            q_seqinfo.seqstart = pt_attn_bias.q_seqinfo.seqstart.data_ptr()
            q_seqinfo.max_seqlen = pt_attn_bias.q_seqinfo.max_seqlen
            #q_seqinfo.seqstart_py = pt_attn_bias.q_seqinfo.seqstart_py.data_ptr()
            attn_bias.batchsize = pt_attn_bias.q_seqinfo.seqstart.size(0)-1

        attn_bias.k_seqinfo = k_seqinfo
        attn_bias.q_seqinfo = q_seqinfo
        input_metadata_c.attn_bias = attn_bias


    input_metadata_c.schedule_type = 0 # 0: vllm. 1:sarathi, 2:custom, 3:self-build
    if input_metadata.block_tables is not None:
        input_metadata_c.block_tables = input_metadata.block_tables.data_ptr()
        input_metadata_c.context_lens = input_metadata.context_lens.data_ptr()
        input_metadata_c.block_tables_size_1 = input_metadata.block_tables.shape[-1]
        input_metadata_c.context_lens_size_1 = input_metadata.context_lens.shape[-1]
        input_metadata_c.max_context_len = input_metadata.max_context_len
        
    input_metadata_c.is_prompt = input_metadata.is_prompt
    input_metadata_c.slot_mapping = input_metadata.slot_mapping.data_ptr()

    # process event
    if event_list and len(event_list) > 0:
        for i in range(len(event_list)):
            event_list_c.cuda_event[i] = event_list[i]._as_parameter_

    input_metadata_c.cache_events = event_list_c
    #input_metadata_c.cache_stream = input_metadata.cache_stream._as_parameter_ if input_metadata.cache_stream else c_void_p(0)

    
    return GetAddrForCStruct(input_metadata_c)
