import torch
from vllm.model_executor.parallel_utils.parallel_state import (get_tensor_model_parallel_group,
                                                               get_tensor_model_parallel_rank, 
                                                               get_tensor_model_parallel_world_size)
def Debug(inputs:torch.Tensor, ):
    return inputs

class DebugStep(torch.autograd.Function):
    @staticmethod
    def forward(self, inputs:torch.Tensor, ):
        return inputs
    @staticmethod
    def symbolic(graph, inputs: torch.Tensor, ):
        return graph.op("vllm.ort.ext::TorchExtension", inputs, outputs=1,
                        num_inputs_i=1, num_outputs_i=1, func_name_s="debug_step")
