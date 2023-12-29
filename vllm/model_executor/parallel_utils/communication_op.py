import torch

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)


class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            return input_
        # All-reduce.
        torch.distributed.all_reduce(input_,
                                    group=get_tensor_model_parallel_group())
        return input_

    @staticmethod
    def symbolic(g: torch.Graph, x) -> torch.Value:
        return g.op('com.microsoft::AllReduce', x)


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim) -> torch.Tensor:
        input_size = input_.size()
        world_size = get_tensor_model_parallel_world_size()

        # Allocate output tensor.
        output_tensor = torch.empty((world_size, ) + input_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        if torch.onnx.is_in_onnx_export():
            return output_tensor
        # All-gather.
        torch.distributed.all_gather_into_tensor(
            output_tensor, input_, group=get_tensor_model_parallel_group())
        return output_tensor

    @staticmethod
    def symbolic(g: torch.Graph, x, dim) -> torch.Value:
        world_size = get_tensor_model_parallel_world_size()
        axes = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
        x = g.op('Unsqueeze', x, axes)
        return g.op('com.microsoft::AllGather', x, group_size_i=world_size, axis_i=0)

def tensor_model_parallel_all_reduce(input_):
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation is applied in-place on the input tensor.
    """
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    return AllReduce.apply(input_)


def tensor_model_parallel_all_gather(input_, dim=-1):
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    output_tensor = AllGather.apply(input_, dim)

    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim], ) +
                                          input_size[dim + 1:])
    return output_tensor
