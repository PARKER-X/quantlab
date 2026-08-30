import numpy as np

from quantization.per_channel import (
    cal_channel_scale,
    quantize_per_channel,
    dequantize_per_channel,
)


# ============================================================
# WEIGHTS
# ============================================================

weights = np.array(
    [
        [0.10, 0.20, 0.15, 0.30],
        [5.00, -4.00, 3.50, -6.00],
        [0.01, 0.02, 0.03, 0.02],
    ],
    dtype=np.float32
)


# ============================================================
# ORIGINAL
# ============================================================

print("=" * 60)
print("PER-CHANNEL INT8 QUANTIZATION")
print("=" * 60)

print("\nOriginal Weights:")
print(weights)

print("\nOriginal Shape:")
print(weights.shape)

print("\nOriginal Dtype:")
print(weights.dtype)


# ============================================================
# CALCULATE SCALES
# ============================================================

scales = cal_channel_scale(weights)

print("\nPer-Channel Scales:")
print(scales)

print("\nScales Shape:")
print(scales.shape)

print("\nScales Dtype:")
print(scales.dtype)


# ============================================================
# QUANTIZE
# ============================================================

quantized_weights = quantize_per_channel(
    weights,
    scales
)

print("\nQuantized Weights:")
print(quantized_weights)

print("\nQuantized Shape:")
print(quantized_weights.shape)

print("\nQuantized Dtype:")
print(quantized_weights.dtype)

print("\nMinimum INT8 Value:")
print(np.min(quantized_weights))

print("\nMaximum INT8 Value:")
print(np.max(quantized_weights))


# ============================================================
# DEQUANTIZE
# ============================================================

dequantized_weights = dequantize_per_channel(
    quantized_weights,
    scales
)

print("\nDequantized Weights:")
print(dequantized_weights)

print("\nDequantized Dtype:")
print(dequantized_weights.dtype)


# ============================================================
# ERROR
# ============================================================

error = weights - dequantized_weights

absolute_error = np.abs(error)

mae = np.mean(absolute_error)

mse = np.mean(error ** 2)

rmse = np.sqrt(mse)


print("\nQuantization Error:")
print(error)

print("\nAbsolute Error:")
print(absolute_error)

print("\nMAE:")
print(mae)

print("\nMSE:")
print(mse)

print("\nRMSE:")
print(rmse)


# ============================================================
# PER-CHANNEL ERROR
# ============================================================

channel_mae = np.mean(
    absolute_error,
    axis=1
)

print("\nMAE Per Channel:")
print(channel_mae)


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)