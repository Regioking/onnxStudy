import onnx


model = onnx.load('pointnet_sem.onnx')
# 此处可以理解为获得了一个维度 “引用”，通过该 “引用“可以修改其对应的维度
dim_proto0 = model.graph.input[0].type.tensor_type.shape.dim[1]
# 将该维度赋值为字符串，其维度不再为和dummy_input绑定的值
dim_proto0.dim_param = 'input.0_1'
dim_proto_o1 = model.graph.output[0].type.tensor_type.shape.dim[1]
onnx.save(model, 'dynamic_input_encoder.onnx')