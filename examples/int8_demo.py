import numpy as np
from quantization.symmetric import cal_scale, quantize_tensor, dequantize_tensor

weights = np.array(
     [1.2, -3.5, 0.8, 2.1, -1.9, 3.5],
     dtype=np.float32
)

scale = cal_scale(weights)
quantized_weights = quantize_tensor(weights, scale)
dequantized_weights = dequantize_tensor(quantized_weights, scale)

print("Original Weights: ", weights)
print("Original Weights Dtype: ", weights.dtype)
print("Scale Factor: ", scale)
print("Quantized Weights: ", quantized_weights)
print("Quantized Weights Dtype: ", quantized_weights.dtype)
print("Dequantized Weights: ", dequantized_weights)
print("Dequantized Weights Dtype: ", dequantized_weights.dtype)

#Errors

error = weights - dequantized_weights
print("Quantization Error: ", error)
absolute_error = np.abs(error)
print("Absolute Quantization Error: ", absolute_error)
mae = np.mean(absolute_error)
print("Mean Absolute Error (MAE): ", mae)
mse = np.mean(error ** 2)
print("Mean Squared Error (MSE): ", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE): ", rmse)

