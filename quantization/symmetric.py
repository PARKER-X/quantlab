import numpy as np


# Step1: Calculate the scale factor for symmetric quantization
def cal_scale(tensor):
    abs_max = np.max(np.abs(tensor))
    scale = abs_max / 127.0
    return scale

#Step2: Quantize the tensor using the scale factor
def quantize_tensor(tensor, scale):
    quantized_tensor = np.round(tensor/scale)
    quantized_tensor = np.clip(quantized_tensor, -127, 127)
    return quantized_tensor.astype(np.int8)

#Step3: Dequantize the tensor back to its original form
def dequantize_tensor(quantized_tensor, scale):
    return quantized_tensor.astype(np.float32) * scale
