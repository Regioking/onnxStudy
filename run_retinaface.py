import os
import cv2
import numpy as np
import TopsInference
import onnx
import onnxruntime

def onnxruntime_infer(model_name, data):
    session = onnxruntime.InferenceSession(model_name)
    outputs = session.run(['face_rpn_cls_prob_stride32_data', 'face_rpn_cls_prob_stride32'],
                          {'data': data})
    return outputs

def compile_executale(model_name, executable_name):
    onnx_parser = TopsInference.create_parser(TopsInference.ONNX_MODEL)
    network = onnx_parser.read(model_name)
    optimizer = TopsInference.create_optimizer()
    optimizer.set_build_flag(TopsInference.KDEFAULT)
    executable = optimizer.build(network)
    executable.save_executable(executable_name)
    print("saved executable file: ", executable_name)
    return executable

def tops_infer(model_name, data):   
    executable = None
    name, ext = os.path.splitext(model_name)
    executable_name = name + '.exec'
    with TopsInference.device(0, 0):
        if os.path.exists(executable_name):
            executable = TopsInference.load(executable_name)
        else:
            executable = compile_executale(model_name, executable_name)
        outputs = []
        print(type(data))
        executable.run([data], outputs,
                    TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST)
    return outputs

def run_retinaface(model_name, img_name):
    img = cv2.imread(img_name)
    img = cv2.resize(img, (640, 640))
    # input shape: 1, 3, 640, 640
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    data = np.array(img.astype(np.float32), order="C")
    outputs = onnxruntime_infer(model_name, data)
    #outputs = tops_infer(model_name, data)
    print(outputs[0])
    print(outputs[1])
    
def add_softmax_as_output(model_name):
    model = onnx.load(model_name)
    softmax_input = onnx.helper.make_tensor_value_info('face_rpn_cls_prob_stride32_data', onnx.TensorProto.FLOAT, [1, 2, 40, 20])
    softmax_output = onnx.helper.make_tensor_value_info('face_rpn_cls_prob_stride32', onnx.TensorProto.FLOAT, [1, 2, 40, 20])
    model.graph.output.insert(0, softmax_input)
    model.graph.output.insert(0, softmax_output)
    onnx.save(model, 'new_model.onnx')
    
def create_softmax_model(dims):
    input = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, dims)
    output = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, dims)
    node = onnx.helper.make_node('Softmax', ['input'], ['output'], axis=1)
    graph = onnx.helper.make_graph([node], 'softmax_model', [input], [output])
    model = onnx.helper.make_model(graph, producer_name='softmax')
    onnx.checker.check_model(model)
    onnx.save(model, 'softmax.onnx')
    
def run_softmax_model():
    session = onnxruntime.InferenceSession('softmax.onnx')
    input = np.array([0.5619, 0.1096, 0.0298, -0.7844])
    input = input.reshape(2, 2)
    outputs = session.run(['output'], {'data': input})
    print(outputs[0])
    
if __name__ == '__main__':
    #add_softmax_as_output('R50_h640_w640_op13.onnx')
    #run_retinaface('new_model.onnx', 'test.jpg')
    #create_softmax_model([2, 2])
    run_softmax_model()