import onnx
import onnxruntime
from onnx import helper
import numpy as np
print(onnx.__version__)
print(onnxruntime.__version__)
model1 = 'dbnet-op13.onnx'
model2 = 'pointnet_sem.onnx'
dbnet = onnx.load(model1)
output = dbnet.graph.output
print(output)
# 检查模型格式是否完整及正确
# print(onnx.checker.check_model(dbnet))
# 输出层信息

# 获取中间节点输出
# # 创建中间节点：层名称、数据类型、维度信息
#prob_info = onnx.helper.make_tensor_value_info('layer1', onnx.TensorProto.FLOAT, [1, 3, 320, 320])
# dbnet.graph.output.insert(3, prob_info)
#onnx.save(dbnet, 'onnx_model_new.onnx')
#onnx_session = onnxruntime.InferenceSession('onnx_model_new.onnx')
onnx_session = onnxruntime.InferenceSession(model1)
input_name = onnx_session.get_inputs()[0].name
output_names = [node.name for node in onnx_session.get_outputs() ]
print(output_names)
np.random.seed(0)
# input = np.random.randn(1, 3, 640, 640).astype(np.float32)
input = np.ones((1, 3, 320, 320)).astype(np.float32)

'''# (arg0: List[str], arg1: Dict[str, object])'''
outputs = onnx_session.run(['331'], {input_name: input})

print("outputs value is :")
print(outputs)
