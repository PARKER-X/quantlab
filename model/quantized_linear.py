import torch
import torch.nn as nn


# ============================================================
# INT8 RANGE
# ============================================================

QMIN = -128
QMAX = 127


# ============================================================
# QUANTIZED LINEAR
# ============================================================

class QuantizedLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # ----------------------------------------------------
        # INT8 WEIGHTS
        # ----------------------------------------------------

        self.register_buffer(
            "weight_int8",
            torch.zeros(
                out_features,
                in_features,
                dtype=torch.int8
            )
        )

        # ----------------------------------------------------
        # ONE SCALE PER OUTPUT CHANNEL
        # ----------------------------------------------------

        self.register_buffer(
            "scales",
            torch.ones(
                out_features,
                dtype=torch.float32
            )
        )

        # ----------------------------------------------------
        # BIAS
        # ----------------------------------------------------

        if bias:

            self.register_buffer(
                "bias",
                torch.zeros(
                    out_features,
                    dtype=torch.float32
                )
            )

        else:

            self.bias = None


    # ========================================================
    # CREATE QUANTIZED LAYER FROM NORMAL LINEAR
    # ========================================================

    @classmethod
    def from_float(cls, linear):

        if not isinstance(linear, nn.Linear):

            raise TypeError(
                "Expected torch.nn.Linear"
            )

        # ----------------------------------------------------
        # CREATE QUANTIZED LAYER
        # ----------------------------------------------------

        quantized_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None
        )

        # ----------------------------------------------------
        # GET FLOAT WEIGHTS
        # ----------------------------------------------------

        weight = linear.weight.detach().float()

        # ----------------------------------------------------
        # FIND MAX ABS VALUE PER OUTPUT CHANNEL
        #
        # Linear weight shape:
        #
        # [out_features, in_features]
        #
        # Each row = one output channel
        # ----------------------------------------------------

        max_abs = torch.amax(
            torch.abs(weight),
            dim=1
        )

        # ----------------------------------------------------
        # CALCULATE SCALE
        # ----------------------------------------------------

        scales = torch.where(
            max_abs == 0,
            torch.ones_like(max_abs),
            max_abs / QMAX
        )

        # ----------------------------------------------------
        # QUANTIZE
        # ----------------------------------------------------

        weight_int8 = torch.round(
            weight / scales.unsqueeze(1)
        )

        weight_int8 = torch.clamp(
            weight_int8,
            QMIN,
            QMAX
        )

        weight_int8 = weight_int8.to(
            torch.int8
        )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        quantized_layer.weight_int8.copy_(
            weight_int8
        )

        quantized_layer.scales.copy_(
            scales
        )

        # ----------------------------------------------------
        # COPY BIAS
        # ----------------------------------------------------

        if linear.bias is not None:

            quantized_layer.bias.copy_(
                linear.bias.detach().float()
            )

        return quantized_layer


    # ========================================================
    # DEQUANTIZE WEIGHTS
    # ========================================================

    def dequantize_weight(self):

        return (
            self.weight_int8.float()
            * self.scales.unsqueeze(1)
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # DEQUANTIZE WEIGHTS
        #
        # This is our first/simple implementation.
        #
        # Later we can implement actual INT8 GEMM kernels.
        # ----------------------------------------------------

        weight = self.dequantize_weight()

        return torch.nn.functional.linear(
            x,
            weight,
            self.bias
        )