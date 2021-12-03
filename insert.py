import onnx
from onnx import shape_inference



# def insert_output(model_path, nodename):
#     model = onnx.load(model_path)
#     model = shape_inference.infer_shapes(model)
#     model.graph.output.insert(0, nodename)
#     onnx.save(model, 'onnx_model_new.onnx')
#
#
# insert_output('dbnet-op13.onnx', "out")


def add_output(model_file, node_names):
    onnx_model = onnx.load(model_file)
    onnx.checker.check_model(onnx_model)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    graph = onnx_model.graph
    graph.output.insert(0, graph.input[0])
    for i, tensor in enumerate(graph.value_info):
        if tensor.name in node_names:
            graph.output.insert(i + 1, tensor)
    model_file = "temp.onnx"
    onnx.save(onnx_model, model_file)
    print("new model saved")

add_output('dbnet-op13.onnx', '377')