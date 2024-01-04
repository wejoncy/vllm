from onnxruntime.quantization import matmul_4bits_quantizer
from typing import List,Tuple
import onnx
import numpy as np
from onnx.onnx_pb import GraphProto, ModelProto, NodeProto, TensorProto
import torch
torch.ops.load_library("weightOnlyQuantOp.cpython-311-x86_64-linux-gnu.so")



def __get_initializer(name, graph_path: List[GraphProto]) -> Tuple[TensorProto, GraphProto]:
        for gid in range(len(graph_path) - 1, -1, -1):
            graph = graph_path[gid]
            for tensor in graph.initializer:
                if tensor.name == name:
                    return tensor, graph
        return None, None

def quant_moe_weight(weights, quant_mode:bool=True):
    weights = torch.from_numpy(weights.copy())
    # use the test version `_symmetric_...` to get the non-interleaved weights
    _, processed_q_weight, torch_weight_scales = torch.ops.fastertransformer._symmetric_quantize_last_axis_of_batched_matrix(
        weights.cpu().contiguous(), torch.quint4x2)

    return processed_q_weight.numpy().astype(np.uint8),torch_weight_scales.numpy()



def quant_MOE_node(node, graph_stack):
    node_inputs = node.input
    fc_weights = node_inputs[2], node_inputs[5], node_inputs[8]  # noqa: N806
    for idx, fc_weight in zip((2,5,8), fc_weights):
        B, Bs_graph = __get_initializer(fc_weight, graph_stack)  # noqa: N806
        B_array = onnx.numpy_helper.to_array(B)  # noqa: N806
        packed, scales = quant_moe_weight(B_array)
        B_quant = onnx.numpy_helper.from_array(packed)  # noqa: N806
        B_quant.name = B.name + "_Q4"
        node_inputs[idx] = B_quant.name
        for input in Bs_graph.input:
            if input.name == fc_weight:
                Bs_graph.input.remove(input)
                break
        scales_tensor = onnx.numpy_helper.from_array(scales)
        scales_tensor.name = B.name + "_scales"
        Bs_graph.initializer.extend([B_quant, scales_tensor])
        node_inputs[idx+1] = scales_tensor.name

    moe_q4_node = onnx.helper.make_node(
            "MoE",
            node_inputs,
            node.output,
            node.name,
            domain="com.microsoft",
        )
    moe_q4_node.attribute.extend(node.attribute)
    return moe_q4_node

def quant_MOE(onnx_model):
    q_moe_nodes = []
    nodes_removed = []
    for node in onnx_model.graph.node:
        if node.op_type == "MoE":
            nodes_removed.append(node)
            q_moe_nodes.append(quant_MOE_node(node, [onnx_model.graph]))
    for node in nodes_removed:
        onnx_model.graph.node.remove(node)
    onnx_model.graph.node.extend(q_moe_nodes)
    
    print('')


def quant_onnx_by_4bits(input_model_path, output_model_path):
    model = onnx.load(input_model_path)
    block_size = 32
    symmetric = False
    nodes_to_exclude = []
    for node in model.graph.node:
        if node.op_type == "MatMul" and 'gate' in node.name:
            nodes_to_exclude.append(node.name)
    quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(model, block_size, symmetric, nodes_to_exclude=nodes_to_exclude)
    quant_MOE(quant.model.model)
    quant.process()
    quant.model.save_model_to_file(output_model_path, True)