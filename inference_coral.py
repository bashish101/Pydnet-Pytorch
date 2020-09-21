from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

# Load input image
img1 = Image.open('test/1.png')
img1 = img1.resize((640, 192), Image.ANTIALIAS)
example1_tf = img1.astype(np.float32)

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="mobile_pydnet_pruned35.tflite",
                  experimental_delegates=[
                      tflite.load_delegate(EDGETPU_SHARED_LIB)
                  ])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32) # example1.to('cpu').detach().numpy().astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], examplee1_tf)

print(input_shape)
%timeit interpreter.invoke()
print("done")
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.

output1 = interpreter.get_tensor(output_details[0]['index']).squeeze()