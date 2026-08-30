import numpy as np
import pytest

from quantization.per_channel import (
    cal_channel_scale,
    quantize_per_channel,
    dequantize_per_channel,
)


# ============================================================
# TEST 1: SCALE CALCULATION
# ============================================================

def test_channel_scales():

    weights = np.array(
        [
            [0.10, 0.20, 0.15, 0.30],
            [5.00, -4.00, 3.50, -6.00],
            [0.01, 0.02, 0.03, 0.02],
        ],
        dtype=np.float32
    )

    scales = cal_channel_scale(weights)

    expected = np.array(
        [
            0.30 / 127.0,
            6.00 / 127.0,
            0.03 / 127.0,
        ],
        dtype=np.float32
    )

    np.testing.assert_allclose(
        scales,
        expected,
        rtol=1e-6,
        atol=1e-7
    )


# ============================================================
# TEST 2: SCALE DTYPE
# ============================================================

def test_channel_scale_dtype():

    weights = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32
    )

    scales = cal_channel_scale(weights)

    assert scales.dtype == np.float32


# ============================================================
# TEST 3: NUMBER OF SCALES
# ============================================================

def test_number_of_channel_scales():

    weights = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32
    )

    scales = cal_channel_scale(weights)

    assert scales.shape == (3,)


# ============================================================
# TEST 4: QUANTIZED DTYPE
# ============================================================

def test_quantized_dtype():

    weights = np.array(
        [
            [1.2, -2.3],
            [3.4, -4.5],
        ],
        dtype=np.float32
    )

    scales = cal_channel_scale(weights)

    quantized = quantize_per_channel(
        weights,
        scales
    )

    assert quantized.dtype == np.int8


# ============================================================
# TEST 5: QUANTIZED VALUES STAY IN INT8 RANGE
# ============================================================

def test_quantized_range():

    weights = np.array(
        [
            [1.2, -2.3, 0.5],
            [10.0, -20.0, 15.0],
        ],
        dtype=np.float32
    )

    scales = cal_channel_scale(weights)

    quantized = quantize_per_channel(
        weights,
        scales
    )

    assert np.all(quantized >= -128)
    assert np.all(quantized <= 127)


# ============================================================
# TEST 6: MAXIMUM VALUE BECOMES 127
# ============================================================

def test_channel_max_maps_to_127():

    weights = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 5.0, 2.0],
        ],
        dtype=np.float32
    )

    scales = cal_channel_scale(weights)

    quantized = quantize_per_channel(
        weights,
        scales
    )

    assert quantized[0, 2] == 127
    assert quantized[1, 0] == 127


# ============================================================
# TEST 7: NEGATIVE MAXIMUM MAPS CLOSE TO -127
# ============================================================

def test_negative_maximum():

    weights = np.array(
        [
            [-3.0, 1.0, 2.0],
            [4.0, -8.0, 2.0],
        ],
        dtype=np.float32
    )

    scales = cal_channel_scale(weights)

    quantized = quantize_per_channel(
        weights,
        scales
    )

    assert quantized[0, 0] == -127
    assert quantized[1, 1] == -127


# ============================================================
# TEST 8: DEQUANTIZATION DTYPE
# ============================================================

def test_dequantized_dtype():

    weights = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32
    )

    scales = cal_channel_scale(weights)

    quantized = quantize_per_channel(
        weights,
        scales
    )

    dequantized = dequantize_per_channel(
        quantized,
        scales
    )

    assert dequantized.dtype == np.float32


# ============================================================
# TEST 9: DEQUANTIZED SHAPE
# ============================================================

def test_dequantized_shape():

    weights = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32
    )

    scales = cal_channel_scale(weights)

    quantized = quantize_per_channel(
        weights,
        scales
    )

    dequantized = dequantize_per_channel(
        quantized,
        scales
    )

    assert dequantized.shape == weights.shape


# ============================================================
# TEST 10: ROUND TRIP
# ============================================================

def test_quantization_round_trip():

    weights = np.array(
        [
            [0.10, 0.20, 0.30],
            [5.00, -4.00, -6.00],
            [0.01, 0.02, 0.03],
        ],
        dtype=np.float32
    )

    scales = cal_channel_scale(weights)

    quantized = quantize_per_channel(
        weights,
        scales
    )

    dequantized = dequantize_per_channel(
        quantized,
        scales
    )

    np.testing.assert_allclose(
        dequantized,
        weights,
        rtol=0.02,
        atol=0.001
    )


# ============================================================
# TEST 11: ZERO CHANNEL
# ============================================================

def test_zero_channel():

    weights = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float32
    )

    scales = cal_channel_scale(weights)

    assert scales[0] == 1.0

    quantized = quantize_per_channel(
        weights,
        scales
    )

    dequantized = dequantize_per_channel(
        quantized,
        scales
    )

    assert np.all(quantized[0] == 0)
    assert np.all(dequantized[0] == 0.0)


# ============================================================
# TEST 12: INVALID SCALE SHAPE
# ============================================================

def test_invalid_scale_shape():

    weights = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32
    )

    scales = np.array(
        [1.0],
        dtype=np.float32
    )

    with pytest.raises(ValueError):

        quantize_per_channel(
            weights,
            scales
        )


# ============================================================
# TEST 13: ZERO SCALE IS INVALID
# ============================================================

def test_zero_scale():

    weights = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32
    )

    scales = np.array(
        [0.0, 1.0],
        dtype=np.float32
    )

    with pytest.raises(ValueError):

        quantize_per_channel(
            weights,
            scales
        )


# ============================================================
# TEST 14: NEGATIVE SCALE IS INVALID
# ============================================================

def test_negative_scale():

    weights = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32
    )

    scales = np.array(
        [-1.0, 1.0],
        dtype=np.float32
    )

    with pytest.raises(ValueError):

        quantize_per_channel(
            weights,
            scales
        )


# ============================================================
# TEST 15: HIGHER DIMENSION TENSOR
# ============================================================

def test_4d_tensor():

    weights = np.array(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
            ],
            [
                [[5.0, 6.0], [7.0, 8.0]],
            ],
        ],
        dtype=np.float32
    )

    scales = cal_channel_scale(weights)

    assert scales.shape == (2,)

    quantized = quantize_per_channel(
        weights,
        scales
    )

    assert quantized.shape == weights.shape
    assert quantized.dtype == np.int8

    dequantized = dequantize_per_channel(
        quantized,
        scales
    )

    assert dequantized.shape == weights.shape