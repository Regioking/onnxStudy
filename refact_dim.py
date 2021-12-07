import onnx

model = onnx.load('dbnet-op13.onnx')
# 此处可以理解为获得了一个维度 “引用”，通过该 “引用“可以修改其对应的维度
dim_proto2 = model.graph.input[0].type.tensor_type.shape.dim[2]
dim_proto3 = model.graph.input[0].type.tensor_type.shape.dim[3]
# 将该维度赋值为字符串，其维度不再为和dummy_input绑定的值
dim_proto2.dim_param = '320'
dim_proto3.dim_param = '320'
dim_proto_o1 = model.graph.output[0].type.tensor_type.shape.dim[1]
onnx.save(model, 'dynamic_input_encoder.onnx')
