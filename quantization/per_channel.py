import numpy as np

Qmin = -128
Qmax = 127

def cal_channel_scale(tensor):
    tensor = np.array(tensor)
    if tensor.ndim==0:
        raise ValueError("Input tensor must have at least one dimension.")
    
    max_val_per_channel = np.max(np.abs(tensor),axis = tuple(range(1,tensor.ndim)))
    scales = np.where(max_val_per_channel == 0, 1.0, max_val_per_channel / Qmax)
    return scales.astype(np.float32)

def quantize_per_channel(tensor, scales):
    tensor = np.asarray(tensor)
    scales = np.asarray(scales)

    if tensor.ndim==0:
        raise ValueError("Input tensor must have at least one dimension.")
    if scales.ndim != 1:
        raise ValueError("Scales must be a 1D array.")
    if tensor.shape[0] != scales.shape[0]:
        raise ValueError("The first dimension of the tensor must match the length of the scales array.")
    if np.any(scales <= 0):
        raise ValueError("Scales must not contain zero or negative values.")
    re_shaped_scales = scales.reshape((scales.shape[0],) + (1,) * (tensor.ndim - 1))
    quantized_tensor = np.round(tensor / re_shaped_scales)
    quantized_tensor = np.clip(quantized_tensor, Qmin, Qmax)    
    return quantized_tensor.astype(np.int8)

def dequantize_per_channel(quantized_tensor, scales):
    quantized_tensor = np.asarray(quantized_tensor)
    scales = np.asarray(scales)

    if quantized_tensor.ndim == 0:
        raise ValueError(
            "Quantized tensor must have at least one dimension."
        )

    if scales.ndim != 1:
        raise ValueError("Scales must be a 1-dimensional array.")

    if len(scales) != quantized_tensor.shape[0]:
        raise ValueError(
            "Number of scales must match the number of channels."
        )

    if np.any(scales <= 0):
        raise ValueError("All scales must be greater than zero.")

    reshape_shape = (
        quantized_tensor.shape[0],
    ) + (1,) * (quantized_tensor.ndim - 1)

    scales_reshaped = scales.reshape(reshape_shape)

    # Convert INT8 to FLOAT32 before multiplication
    dequantized = (
        quantized_tensor.astype(np.float32)
        * scales_reshaped
    )

    return dequantized.astype(np.float32)