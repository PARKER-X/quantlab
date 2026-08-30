import numpy as np

from quantization.symmetric import (
    cal_scale,
    quantize_tensor,
    dequantize_tensor,
)


# ============================================================
# 1. SCALE TESTS
# ============================================================

def test_scale_calculation():
    weights = np.array(
        [1.2, -3.5, 0.8, 2.1, -1.9, 3.5],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    expected = 3.5 / 127.0

    assert np.isclose(scale, expected)


def test_scale_uses_absolute_maximum():
    weights = np.array(
        [-10.0, 2.0, 5.0],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    expected = 10.0 / 127.0

    assert np.isclose(scale, expected)


def test_scale_positive_values():
    weights = np.array(
        [1.0, 2.0, 5.0],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    assert np.isclose(scale, 5.0 / 127.0)


def test_scale_negative_values():
    weights = np.array(
        [-1.0, -2.0, -5.0],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    assert np.isclose(scale, 5.0 / 127.0)


# ============================================================
# 2. QUANTIZATION TESTS
# ============================================================

def test_quantized_dtype():
    weights = np.array(
        [1.2, -3.5, 0.8],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    quantized = quantize_tensor(weights, scale)

    assert quantized.dtype == np.int8


def test_quantized_range():
    weights = np.array(
        [-100.0, -20.0, 0.0, 20.0, 100.0],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    quantized = quantize_tensor(weights, scale)

    assert np.min(quantized) >= -128
    assert np.max(quantized) <= 127


def test_maximum_value_maps_to_127():
    weights = np.array(
        [-3.5, 3.5],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    quantized = quantize_tensor(weights, scale)

    assert quantized[1] == 127
    assert quantized[0] == -127


def test_zero_maps_to_zero():
    weights = np.array(
        [-3.5, 0.0, 3.5],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    quantized = quantize_tensor(weights, scale)

    assert quantized[1] == 0


def test_quantization_is_symmetric():
    weights = np.array(
        [-2.0, 2.0],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    quantized = quantize_tensor(weights, scale)

    assert quantized[0] == -quantized[1]


# ============================================================
# 3. DEQUANTIZATION TESTS
# ============================================================

def test_dequantization():
    quantized = np.array(
        [-127, 0, 127],
        dtype=np.int8
    )

    scale = 2.0 / 127.0

    dequantized = dequantize_tensor(
        quantized,
        scale
    )

    assert np.isclose(dequantized[0], -2.0)
    assert np.isclose(dequantized[1], 0.0)
    assert np.isclose(dequantized[2], 2.0)


def test_dequantized_dtype():
    quantized = np.array(
        [-10, 0, 10],
        dtype=np.int8
    )

    scale = 0.1

    dequantized = dequantize_tensor(
        quantized,
        scale
    )

    assert dequantized.dtype == np.float32


# ============================================================
# 4. ROUND-TRIP TEST
# ============================================================

def test_quantization_round_trip():
    weights = np.array(
        [1.2, -3.5, 0.8, 2.1, -1.9, 3.5],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    quantized = quantize_tensor(
        weights,
        scale
    )

    reconstructed = dequantize_tensor(
        quantized,
        scale
    )

    error = np.abs(
        weights - reconstructed
    )

    mae = np.mean(error)

    assert mae < 0.02


# ============================================================
# 5. ERROR / QUALITY TESTS
# ============================================================

def test_quantization_error_is_small():
    weights = np.array(
        [0.1, 0.5, 1.0, 2.0],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    quantized = quantize_tensor(
        weights,
        scale
    )

    reconstructed = dequantize_tensor(
        quantized,
        scale
    )

    mae = np.mean(
        np.abs(weights - reconstructed)
    )

    assert mae >= 0.0
    assert mae < 0.02


def test_mse_is_non_negative():
    weights = np.array(
        [1.2, -3.5, 0.8],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    quantized = quantize_tensor(
        weights,
        scale
    )

    reconstructed = dequantize_tensor(
        quantized,
        scale
    )

    mse = np.mean(
        (weights - reconstructed) ** 2
    )

    assert mse >= 0.0


def test_rmse_is_non_negative():
    weights = np.array(
        [1.2, -3.5, 0.8],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    quantized = quantize_tensor(
        weights,
        scale
    )

    reconstructed = dequantize_tensor(
        quantized,
        scale
    )

    mse = np.mean(
        (weights - reconstructed) ** 2
    )

    rmse = np.sqrt(mse)

    assert rmse >= 0.0


# ============================================================
# 6. SPECIAL CASES
# ============================================================

def test_zero_tensor():
    weights = np.zeros(
        10,
        dtype=np.float32
    )

    scale = cal_scale(weights)

    assert scale == 0.0


def test_zero_values_remain_zero_after_round_trip():
    weights = np.array(
        [0.0, 1.0, -1.0],
        dtype=np.float32
    )

    scale = cal_scale(weights)

    quantized = quantize_tensor(
        weights,
        scale
    )

    reconstructed = dequantize_tensor(
        quantized,
        scale
    )

    assert reconstructed[0] == 0.0