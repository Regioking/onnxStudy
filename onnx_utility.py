import onnx
from onnx import shape_inference
import re


def add_output(src_file: str, node_names, target_file=None):
    if target_file is None:
        target_file = "{}_add_{}.onnx".format(src_file.split('.')[0], node_names)
    onnx_model = onnx.load(src_file)
    onnx.checker.check_model(onnx_model)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    graph = onnx_model.graph
    for i, tensor in enumerate(graph.value_info):
        if tensor.name in node_names:
            graph.output.insert(i + 1, tensor)
    onnx.save(onnx_model, target_file)
    print("{} saved".format(target_file))


if __name__ == "__main__":
    add_output('dbnet-op13.onnx', '378')
