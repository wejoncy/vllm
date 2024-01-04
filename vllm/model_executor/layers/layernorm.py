"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from vllm._C import ops

ONNX_EXPORT_LEVEL = 2
class ONNXExtentions(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, x, weight, variance_epsilon, residual):
        extra_attributes = f"variance_epsilon={variance_epsilon}"
        if residual is None:
            if ONNX_EXPORT_LEVEL == 2:
                return graph.op("SimplifiedLayerNormalization",  x, weight)
            elif ONNX_EXPORT_LEVEL == 3:
                return graph.op("vllm.ort.ext::TorchExtension",  x, weight, outputs=1,
                            num_inputs_i=2, num_outputs_i=1, func_name_s="rms_norm",extra_attributes_s=extra_attributes)
        else:
            return graph.op("vllm.ort.ext::TorchExtension", x, residual, weight, outputs=2,
                            num_inputs_i=3, num_outputs_i=2, func_name_s="fused_add_rms_norm",
                            extra_attributes_s=extra_attributes)

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight, variance_epsilon, residual: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        if residual is not None:
            return x, residual
        return x

def rms_for_onnx_export(x: torch.Tensor, weight, variance_epsilon, residual: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
    if ONNX_EXPORT_LEVEL == 2:
        if residual is not None:
            x =x+ residual
            xresidual = x
        rms = ONNXExtentions.apply(x, weight.data, variance_epsilon, None)
    elif ONNX_EXPORT_LEVEL == 3:
        rms_out = ONNXExtentions.apply(x, weight.data, variance_epsilon, residual)
        if residual is not None:
            rms, xresidual = rms_out
        else:
            rms = rms_out
    else:
        hidden_states = x
        if residual is not None:
            hidden_states += residual
            xresidual = hidden_states
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        rms = (weight * hidden_states).to(input_dtype)
    if residual is not None:
        return rms, xresidual
    return rms


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def _forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if torch.onnx.is_in_onnx_export():
            return rms_for_onnx_export(x, self.weight.data, self.variance_epsilon, residual)
        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out
