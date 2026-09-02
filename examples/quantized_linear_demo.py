import torch
import torch.nn as nn

from model.quantized_linear import QuantizedLinear


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(42)


# ============================================================
# CREATE NORMAL FP32 LINEAR
# ============================================================

linear = nn.Linear(
    in_features=8,
    out_features=4
)

linear.eval()


# ============================================================
# CREATE QUANTIZED VERSION
# ============================================================

quantized_linear = QuantizedLinear.from_float(
    linear
)

quantized_linear.eval()


# ============================================================
# INPUT
# ============================================================

x = torch.randn(
    2,
    8
)


# ============================================================
# FP32 OUTPUT
# ============================================================

with torch.no_grad():

    fp32_output = linear(x)


# ============================================================
# INT8 OUTPUT
# ============================================================

with torch.no_grad():

    int8_output = quantized_linear(x)


# ============================================================
# OUTPUTS
# ============================================================

print("=" * 70)
print("FP32 vs INT8 LINEAR")
print("=" * 70)

print("\nInput:")
print(x)

print("\nFP32 Output:")
print(fp32_output)

print("\nINT8 Output:")
print(int8_output)


# ============================================================
# WEIGHT INFORMATION
# ============================================================

print("\nOriginal Weight Dtype:")
print(linear.weight.dtype)

print("\nQuantized Weight Dtype:")
print(quantized_linear.weight_int8.dtype)

print("\nQuantized Weight Shape:")
print(quantized_linear.weight_int8.shape)

print("\nScale Shape:")
print(quantized_linear.scales.shape)


# ============================================================
# ERROR
# ============================================================

error = fp32_output - int8_output

absolute_error = torch.abs(error)

mae = torch.mean(
    absolute_error
).item()

mse = torch.mean(
    error ** 2
).item()

rmse = torch.sqrt(
    torch.mean(error ** 2)
).item()


print("\nError:")
print(error)

print("\nMAE:")
print(mae)

print("\nMSE:")
print(mse)

print("\nRMSE:")
print(rmse)


# ============================================================
# WEIGHT ERROR
# ============================================================

original_weight = linear.weight.detach()

dequantized_weight = (
    quantized_linear.dequantize_weight()
)


weight_error = (
    original_weight - dequantized_weight
)

weight_mae = torch.mean(
    torch.abs(weight_error)
).item()


print("\nWeight MAE:")
print(weight_mae)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
