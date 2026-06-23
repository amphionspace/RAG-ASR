"""Self-contained Zipformer encoder for HuggingFace inference (no k2/utils dependency)."""
import contextlib
import copy
import logging
import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# torch_autocast replacement
# ---------------------------------------------------------------------------
def torch_autocast(enabled=True):
    if torch.cuda.is_available():
        return torch.amp.autocast("cuda", enabled=enabled)
    return contextlib.nullcontext()


# ============================================================================
# scaling.py components
# ============================================================================


class PiecewiseLinear(object):
    """
    Piecewise linear function, from float to float, specified as nonempty list
    of (x,y) pairs with the x values in order.
    """

    def __init__(self, *args):
        assert len(args) >= 1, len(args)
        if len(args) == 1 and isinstance(args[0], PiecewiseLinear):
            self.pairs = list(args[0].pairs)
        else:
            self.pairs = [(float(x), float(y)) for x, y in args]
        for x, y in self.pairs:
            assert isinstance(x, (float, int)), type(x)
            assert isinstance(y, (float, int)), type(y)
        for i in range(len(self.pairs) - 1):
            assert self.pairs[i + 1][0] > self.pairs[i][0], (
                i,
                self.pairs[i],
                self.pairs[i + 1],
            )

    def __str__(self):
        return f"PiecewiseLinear({str(self.pairs)[1:-1]})"

    def __call__(self, x):
        if x <= self.pairs[0][0]:
            return self.pairs[0][1]
        elif x >= self.pairs[-1][0]:
            return self.pairs[-1][1]
        else:
            cur_x, cur_y = self.pairs[0]
            for i in range(1, len(self.pairs)):
                next_x, next_y = self.pairs[i]
                if x >= cur_x and x <= next_x:
                    return cur_y + (next_y - cur_y) * (x - cur_x) / (next_x - cur_x)
                cur_x, cur_y = next_x, next_y
            assert False

    def __mul__(self, alpha):
        return PiecewiseLinear(*[(x, y * alpha) for x, y in self.pairs])

    def __add__(self, x):
        if isinstance(x, (float, int)):
            return PiecewiseLinear(*[(p[0], p[1] + x) for p in self.pairs])
        s, x = self.get_common_basis(x)
        return PiecewiseLinear(
            *[(sp[0], sp[1] + xp[1]) for sp, xp in zip(s.pairs, x.pairs)]
        )

    def max(self, x):
        if isinstance(x, (float, int)):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(
            *[(sp[0], max(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)]
        )

    def min(self, x):
        if isinstance(x, float) or isinstance(x, int):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(
            *[(sp[0], min(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)]
        )

    def __eq__(self, other):
        return self.pairs == other.pairs

    def get_common_basis(self, p: "PiecewiseLinear", include_crossings: bool = False):
        assert isinstance(p, PiecewiseLinear), type(p)
        x_vals = sorted(set([x for x, _ in self.pairs] + [x for x, _ in p.pairs]))
        y_vals1 = [self(x) for x in x_vals]
        y_vals2 = [p(x) for x in x_vals]
        if include_crossings:
            extra_x_vals = []
            for i in range(len(x_vals) - 1):
                if (y_vals1[i] > y_vals2[i]) != (y_vals1[i + 1] > y_vals2[i + 1]):
                    diff_cur = abs(y_vals1[i] - y_vals2[i])
                    diff_next = abs(y_vals1[i + 1] - y_vals2[i + 1])
                    pos = diff_cur / (diff_cur + diff_next)
                    extra_x_val = x_vals[i] + pos * (x_vals[i + 1] - x_vals[i])
                    extra_x_vals.append(extra_x_val)
            if len(extra_x_vals) > 0:
                x_vals = sorted(set(x_vals + extra_x_vals))
        y_vals1 = [self(x) for x in x_vals]
        y_vals2 = [p(x) for x in x_vals]
        return (
            PiecewiseLinear(*zip(x_vals, y_vals1)),
            PiecewiseLinear(*zip(x_vals, y_vals2)),
        )


class ScheduledFloat(torch.nn.Module):
    """
    A floating point value whose value changes depending on the batch count
    of the training loop.
    """

    def __init__(self, *args, default: float = 0.0):
        super().__init__()
        self.batch_count = None
        self.name = None
        self.default = default
        self.schedule = PiecewiseLinear(*args)

    def extra_repr(self) -> str:
        return (
            f"batch_count={self.batch_count}, schedule={str(self.schedule.pairs[1:-1])}"
        )

    def __float__(self):
        batch_count = self.batch_count
        if (
            batch_count is None
            or not self.training
            or torch.jit.is_scripting()
            or torch.jit.is_tracing()
        ):
            return float(self.default)
        else:
            ans = self.schedule(self.batch_count)
            if random.random() < 0.0002:
                logging.info(
                    f"ScheduledFloat: name={self.name}, batch_count={self.batch_count}, ans={ans}"
                )
            return ans

    def __add__(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule + x, default=self.default)
        else:
            return ScheduledFloat(
                self.schedule + x.schedule, default=self.default + x.default
            )

    def max(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule.max(x), default=self.default)
        else:
            return ScheduledFloat(
                self.schedule.max(x.schedule), default=max(self.default, x.default)
            )


FloatLike = Union[float, ScheduledFloat]


class CutoffEstimator:
    """
    Estimates cutoffs of an arbitrary numerical quantity such that a specified
    proportion of items will be above the cutoff on average.
    """

    def __init__(self, p: float):
        self.p = p
        self.count = 0
        self.count_above = 0
        self.cutoff = 0

    def __call__(self, x: float) -> bool:
        ans = x > self.cutoff
        self.count += 1
        if ans:
            self.count_above += 1
        cur_p = self.count_above / self.count
        delta_p = cur_p - self.p
        if (delta_p > 0) == ans:
            q = abs(delta_p)
            self.cutoff = x * q + self.cutoff * (1 - q)
        return ans


def random_cast_to_half(x: Tensor, min_abs: float = 5.0e-06) -> Tensor:
    if x.dtype == torch.float16:
        return x
    x_abs = x.abs()
    is_too_small = x_abs < min_abs
    random_val = min_abs * x.sign() * (torch.rand_like(x) * min_abs < x_abs)
    return torch.where(is_too_small, random_val, x).to(torch.float16)


def logaddexp_onnx(x: Tensor, y: Tensor) -> Tensor:
    max_value = torch.max(x, y)
    diff = torch.abs(x - y)
    return max_value + torch.log1p(torch.exp(-diff))


def logaddexp(x: Tensor, y: Tensor) -> Tensor:
    if torch.jit.is_scripting():
        return torch.logaddexp(x, y)
    elif torch.onnx.is_in_onnx_export():
        return logaddexp_onnx(x, y)
    else:
        return torch.logaddexp(x, y)


# ---------------------------------------------------------------------------
# Core utility functions
# ---------------------------------------------------------------------------


def _no_op(x: Tensor) -> Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return x
    else:
        return x.chunk(1, dim=-1)[0]


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, dim: int):
        ans = x.softmax(dim=dim)
        if torch.is_autocast_enabled():
            ans = ans.to(torch.float16)
        ctx.save_for_backward(ans)
        ctx.x_dtype = x.dtype
        ctx.dim = dim
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        (ans,) = ctx.saved_tensors
        with torch_autocast(enabled=False):
            ans_grad = ans_grad.to(torch.float32)
            ans = ans.to(torch.float32)
            x_grad = ans_grad * ans
            x_grad = x_grad - ans * x_grad.sum(dim=ctx.dim, keepdim=True)
            return x_grad, None


def softmax(x: Tensor, dim: int):
    if not x.requires_grad or torch.jit.is_scripting() or torch.jit.is_tracing():
        return x.softmax(dim=dim)
    return SoftmaxFunction.apply(x, dim)


class LimitParamValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x)
        assert max >= min
        ctx.min = min
        ctx.max = max
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x,) = ctx.saved_tensors
        x_grad = x_grad * torch.where(
            torch.logical_and(x_grad > 0, x < ctx.min), -1.0, 1.0
        )
        x_grad *= torch.where(torch.logical_and(x_grad < 0, x > ctx.max), -1.0, 1.0)
        return x_grad, None, None


def limit_param_value(
    x: Tensor, min: float, max: float, prob: float = 0.6, training: bool = True
):
    if training and random.random() < prob:
        return LimitParamValue.apply(x, min, max)
    else:
        return x


class WithLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor, name: str):
        ctx.y_shape = y.shape
        if random.random() < 0.002 and name is not None:
            loss_sum = y.sum().item()
            logging.info(f"WithLoss: name={name}, loss-sum={loss_sum:.3e}")
        return x

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        return (
            ans_grad,
            torch.ones(ctx.y_shape, dtype=ans_grad.dtype, device=ans_grad.device),
            None,
        )


def with_loss(x, y, name):
    return WithLoss.apply(x, y, name)


def penalize_abs_values_gt(
    x: Tensor, limit: float, penalty: float, name: str = None
) -> Tensor:
    x_sign = x.sign()
    over_limit = (x.abs() - limit) > 0
    aux_loss = penalty * ((x_sign * over_limit).to(torch.int8) * x)
    x = with_loss(x, aux_loss, name)
    return x


def convert_num_channels(x: Tensor, num_channels: int) -> Tensor:
    if num_channels <= x.shape[-1]:
        return x[..., :num_channels]
    else:
        shape = list(x.shape)
        shape[-1] = num_channels - shape[-1]
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return torch.cat((x, zeros), dim=-1)


# ---------------------------------------------------------------------------
# BiasNorm
# ---------------------------------------------------------------------------


class BiasNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        bias: Tensor,
        log_scale: Tensor,
        channel_dim: int,
        store_output_for_backprop: bool,
    ) -> Tensor:
        assert bias.ndim == 1
        if channel_dim < 0:
            channel_dim = channel_dim + x.ndim
        ctx.store_output_for_backprop = store_output_for_backprop
        ctx.channel_dim = channel_dim
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        scales = (
            torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
        ) * log_scale.exp()
        ans = x * scales
        ctx.save_for_backward(
            ans.detach() if store_output_for_backprop else x,
            scales.detach(),
            bias.detach(),
            log_scale.detach(),
        )
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        ans_or_x, scales, bias, log_scale = ctx.saved_tensors
        if ctx.store_output_for_backprop:
            x = ans_or_x / scales
        else:
            x = ans_or_x
        x = x.detach()
        x.requires_grad = True
        bias.requires_grad = True
        log_scale.requires_grad = True
        with torch.enable_grad():
            scales = (
                torch.mean((x - bias) ** 2, dim=ctx.channel_dim, keepdim=True) ** -0.5
            ) * log_scale.exp()
            ans = x * scales
            ans.backward(gradient=ans_grad)
        return x.grad, bias.grad.flatten(), log_scale.grad, None, None


class BiasNorm(torch.nn.Module):
    """
    Simpler replacement for LayerNorm with trainable bias and scalar scale.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,
        log_scale: float = 1.0,
        log_scale_min: float = -1.5,
        log_scale_max: float = 1.5,
        store_output_for_backprop: bool = False,
    ) -> None:
        super(BiasNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale = nn.Parameter(torch.tensor(log_scale))
        self.bias = nn.Parameter(torch.empty(num_channels).normal_(mean=0, std=1e-4))

        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

        self.store_output_for_backprop = store_output_for_backprop

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            channel_dim = self.channel_dim
            if channel_dim < 0:
                channel_dim += x.ndim
            bias = self.bias
            for _ in range(channel_dim + 1, x.ndim):
                bias = bias.unsqueeze(-1)
            scales = (
                torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
            ) * self.log_scale.exp()
            return x * scales

        log_scale = limit_param_value(
            self.log_scale,
            min=float(self.log_scale_min),
            max=float(self.log_scale_max),
            training=self.training,
        )

        return BiasNormFunction.apply(
            x, self.bias, log_scale, self.channel_dim, self.store_output_for_backprop
        )


# ---------------------------------------------------------------------------
# Scaled modules
# ---------------------------------------------------------------------------


def ScaledLinear(*args, initial_scale: float = 1.0, **kwargs) -> nn.Linear:
    ans = nn.Linear(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


def ScaledConv1d(*args, initial_scale: float = 1.0, **kwargs) -> nn.Conv1d:
    ans = nn.Conv1d(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


def ScaledConv2d(*args, initial_scale: float = 1.0, **kwargs) -> nn.Conv2d:
    ans = nn.Conv2d(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


# ---------------------------------------------------------------------------
# ChunkCausalDepthwiseConv1d
# ---------------------------------------------------------------------------


class ChunkCausalDepthwiseConv1d(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        initial_scale: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1

        half_kernel_size = (kernel_size + 1) // 2
        self.causal_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=half_kernel_size,
            padding=0,
            bias=True,
        )

        self.chunkwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

        self.chunkwise_conv_scale = nn.Parameter(torch.zeros(2, channels, kernel_size))
        self.kernel_size = kernel_size

        with torch.no_grad():
            self.causal_conv.weight[:] *= initial_scale
            self.chunkwise_conv.weight[:] *= initial_scale
            if bias:
                torch.nn.init.uniform_(
                    self.causal_conv.bias, -0.1 * initial_scale, 0.1 * initial_scale
                )

    def forward(self, x: Tensor, chunk_size: int = -1) -> Tensor:
        (batch_size, num_channels, seq_len) = x.shape

        left_pad = self.kernel_size // 2

        if chunk_size < 0 or chunk_size > seq_len:
            chunk_size = seq_len
        right_pad = -seq_len % chunk_size

        x = torch.nn.functional.pad(x, (left_pad, right_pad))

        x_causal = self.causal_conv(x[..., : left_pad + seq_len])
        assert x_causal.shape == (batch_size, num_channels, seq_len)

        x_chunk = x[..., left_pad:]
        num_chunks = x_chunk.shape[2] // chunk_size
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks, chunk_size)
        x_chunk = x_chunk.permute(0, 2, 1, 3).reshape(
            batch_size * num_chunks, num_channels, chunk_size
        )
        x_chunk = self.chunkwise_conv(x_chunk)

        chunk_scale = self._get_chunk_scale(chunk_size)

        x_chunk = x_chunk * chunk_scale
        x_chunk = x_chunk.reshape(
            batch_size, num_chunks, num_channels, chunk_size
        ).permute(0, 2, 1, 3)
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks * chunk_size)[
            ..., :seq_len
        ]

        return x_chunk + x_causal

    def _get_chunk_scale(self, chunk_size: int):
        left_edge = self.chunkwise_conv_scale[0]
        right_edge = self.chunkwise_conv_scale[1]
        if chunk_size < self.kernel_size:
            left_edge = left_edge[:, :chunk_size]
            right_edge = right_edge[:, -chunk_size:]
        else:
            t = chunk_size - self.kernel_size
            channels = left_edge.shape[0]
            pad = torch.zeros(
                channels, t, device=left_edge.device, dtype=left_edge.dtype
            )
            left_edge = torch.cat((left_edge, pad), dim=-1)
            right_edge = torch.cat((pad, right_edge), dim=-1)
        return 1.0 + (left_edge + right_edge)


# ---------------------------------------------------------------------------
# Activation functions (k2-free)
# ---------------------------------------------------------------------------


def SwooshLForward(x: Tensor):
    x_offset = x - 4.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.035


def SwooshRForward(x: Tensor):
    x_offset = x - 1.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.313261687


class SwooshL(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return torch.logaddexp(zero, x - 4.0) - 0.08 * x - 0.035


class SwooshR(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687


class DoubleSwish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x - 1.0)


# ---------------------------------------------------------------------------
# Balancer (simplified for inference)
# ---------------------------------------------------------------------------


class Balancer(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        channel_dim: int,
        min_positive: FloatLike = 0.05,
        max_positive: FloatLike = 0.95,
        min_abs: FloatLike = 0.2,
        max_abs: FloatLike = 100.0,
        grad_scale: FloatLike = 0.04,
        prob: Optional[FloatLike] = None,
    ):
        super().__init__()

        if prob is None:
            prob = ScheduledFloat((0.0, 0.5), (8000.0, 0.125), default=0.4)
        self.prob = prob
        self.mem_cutoff = CutoffEstimator(0.05)

        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.grad_scale = grad_scale

    def forward(self, x: Tensor) -> Tensor:
        return _no_op(x)


# ---------------------------------------------------------------------------
# Whiten (simplified for inference)
# ---------------------------------------------------------------------------


class Whiten(nn.Module):
    def __init__(
        self,
        num_groups: int,
        whitening_limit: FloatLike,
        prob: Union[float, Tuple[float, float]],
        grad_scale: FloatLike,
    ):
        super(Whiten, self).__init__()
        assert num_groups >= 1
        assert float(whitening_limit) >= 1
        assert float(grad_scale) >= 0
        self.num_groups = num_groups
        self.whitening_limit = whitening_limit
        self.grad_scale = grad_scale

        if isinstance(prob, float):
            prob = (prob, prob)
        (self.min_prob, self.max_prob) = prob
        assert 0 < self.min_prob <= self.max_prob <= 1
        self.prob = self.max_prob
        self.name = None

    def forward(self, x: Tensor) -> Tensor:
        return _no_op(x)


# ---------------------------------------------------------------------------
# ScaleGrad, Identity
# ---------------------------------------------------------------------------


class ScaleGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad: Tensor):
        return grad * ctx.alpha, None


def scale_grad(x: Tensor, alpha: float):
    return ScaleGradFunction.apply(x, alpha)


class ScaleGrad(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return x
        return scale_grad(x, self.alpha)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return _no_op(x)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------


class Dropout2(nn.Module):
    def __init__(self, p: FloatLike):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.dropout(x, p=float(self.p), training=self.training)


class Dropout3(nn.Module):
    def __init__(self, p: FloatLike, shared_dim: int):
        super().__init__()
        self.p = p
        self.shared_dim = shared_dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.dropout(x, p=float(self.p), training=self.training)


# ---------------------------------------------------------------------------
# ActivationDropoutAndLinear (simplified, k2-free)
# ---------------------------------------------------------------------------


class ActivationDropoutAndLinear(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = "SwooshL",
        dropout_p: FloatLike = 0.0,
        dropout_shared_dim: Optional[int] = -1,
        initial_scale: float = 1.0,
    ):
        super().__init__()
        l = ScaledLinear(
            in_channels, out_channels, bias=bias, initial_scale=initial_scale
        )

        self.weight = l.weight
        self.register_parameter("bias", l.bias)

        self.activation = activation
        self.dropout_p = dropout_p
        self.dropout_shared_dim = dropout_shared_dim

    def forward(self, x: Tensor):
        if self.activation == "SwooshL":
            x = SwooshLForward(x)
        elif self.activation == "SwooshR":
            x = SwooshRForward(x)
        if self.training:
            x = torch.nn.functional.dropout(x, p=float(self.dropout_p), training=True)
        return torch.nn.functional.linear(x, self.weight, self.bias)


# ============================================================================
# encoder_interface.py
# ============================================================================


class EncoderInterface(nn.Module):
    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Please implement it in a subclass")


# ============================================================================
# subsampling.py
# ============================================================================


class ConvNeXt(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_ratio: int = 3,
        kernel_size: Tuple[int, int] = (7, 7),
        layerdrop_rate: FloatLike = None,
    ):
        super().__init__()
        self.padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        hidden_channels = channels * hidden_ratio
        if layerdrop_rate is None:
            layerdrop_rate = ScheduledFloat((0.0, 0.2), (20000.0, 0.015))
        self.layerdrop_rate = layerdrop_rate

        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=kernel_size,
            padding=self.padding,
        )

        self.pointwise_conv1 = nn.Conv2d(
            in_channels=channels, out_channels=hidden_channels, kernel_size=1
        )

        self.hidden_balancer = Balancer(
            hidden_channels,
            channel_dim=1,
            min_positive=0.3,
            max_positive=1.0,
            min_abs=0.75,
            max_abs=5.0,
        )

        self.activation = SwooshL()
        self.pointwise_conv2 = ScaledConv2d(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1,
            initial_scale=0.01,
        )

        self.out_balancer = Balancer(
            channels,
            channel_dim=1,
            min_positive=0.4,
            max_positive=0.6,
            min_abs=1.0,
            max_abs=6.0,
        )
        self.out_whiten = Whiten(
            num_groups=1,
            whitening_limit=5.0,
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return self.forward_internal(x)
        layerdrop_rate = float(self.layerdrop_rate)

        if layerdrop_rate != 0.0:
            batch_size = x.shape[0]
            mask = (
                torch.rand((batch_size, 1, 1, 1), dtype=x.dtype, device=x.device)
                > layerdrop_rate
            )
        else:
            mask = None
        return self.forward_internal(x, mask)

    def forward_internal(
        self, x: Tensor, layer_skip_mask: Optional[Tensor] = None
    ) -> Tensor:
        bypass = x
        x = self.depthwise_conv(x)
        x = self.pointwise_conv1(x)
        x = self.hidden_balancer(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        if layer_skip_mask is not None:
            x = x * layer_skip_mask

        x = bypass + x
        x = self.out_balancer(x)

        if x.requires_grad:
            x = x.transpose(1, 3)
            x = self.out_whiten(x)
            x = x.transpose(1, 3)

        return x


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = (T-3)//2 - 2 == (T-7)//2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer1_channels: int = 8,
        layer2_channels: int = 32,
        layer3_channels: int = 128,
        dropout: FloatLike = 0.1,
    ) -> None:
        assert in_channels >= 7
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=(0, 1),
            ),
            ScaleGrad(0.2),
            Balancer(layer1_channels, channel_dim=1, max_abs=1.0),
            SwooshR(),
            nn.Conv2d(
                in_channels=layer1_channels,
                out_channels=layer2_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            Balancer(layer2_channels, channel_dim=1, max_abs=4.0),
            SwooshR(),
            nn.Conv2d(
                in_channels=layer2_channels,
                out_channels=layer3_channels,
                kernel_size=3,
                stride=(1, 2),
            ),
            Balancer(layer3_channels, channel_dim=1, max_abs=4.0),
            SwooshR(),
        )

        self.convnext = ConvNeXt(layer3_channels, kernel_size=(7, 7))

        self.out_width = (((in_channels - 1) // 2) - 1) // 2
        self.layer3_channels = layer3_channels

        self.out = nn.Linear(self.out_width * layer3_channels, out_channels)
        self.out_whiten = Whiten(
            num_groups=1,
            whitening_limit=ScheduledFloat((0.0, 4.0), (20000.0, 8.0), default=4.0),
            prob=(0.025, 0.25),
            grad_scale=0.02,
        )

        self.out_norm = BiasNorm(out_channels)
        self.dropout = Dropout3(dropout, shared_dim=1)

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.convnext(x)

        b, c, t, f = x.size()

        x = x.transpose(1, 2).reshape(b, t, c * f)

        x = self.out(x)
        x = self.out_whiten(x)
        x = self.out_norm(x)
        x = self.dropout(x)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            x_lens = (x_lens - 7) // 2
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_lens = (x_lens - 7) // 2
        assert x.size(1) == x_lens.max().item(), (x.size(1), x_lens.max())

        return x, x_lens


# ============================================================================
# zipformer.py
# ============================================================================


def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x), (20000.0, ratio * x), default=x)


def _balancer_schedule(min_prob: float):
    return ScheduledFloat((0.0, 0.4), (8000.0, min_prob))


class Zipformer2(EncoderInterface):
    def __init__(
        self,
        output_downsampling_factor: int = 2,
        downsampling_factor: Tuple[int] = (2, 4),
        encoder_dim: Union[int, Tuple[int]] = 384,
        num_encoder_layers: Union[int, Tuple[int]] = 4,
        encoder_unmasked_dim: Union[int, Tuple[int]] = 256,
        query_head_dim: Union[int, Tuple[int]] = 24,
        pos_head_dim: Union[int, Tuple[int]] = 4,
        value_head_dim: Union[int, Tuple[int]] = 12,
        num_heads: Union[int, Tuple[int]] = 8,
        feedforward_dim: Union[int, Tuple[int]] = 1536,
        cnn_module_kernel: Union[int, Tuple[int]] = 31,
        pos_dim: int = 192,
        dropout: FloatLike = None,
        warmup_batches: float = 4000.0,
        causal: bool = False,
        chunk_size: Tuple[int] = [-1],
        left_context_frames: Tuple[int] = [-1],
    ) -> None:
        super(Zipformer2, self).__init__()

        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))

        def _to_tuple(x):
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            else:
                assert len(x) == len(downsampling_factor) and isinstance(x[0], int)
            return x

        self.output_downsampling_factor = output_downsampling_factor
        self.downsampling_factor = downsampling_factor
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim)
        self.encoder_unmasked_dim = encoder_unmasked_dim = _to_tuple(
            encoder_unmasked_dim
        )
        num_encoder_layers = _to_tuple(num_encoder_layers)
        self.num_encoder_layers = num_encoder_layers
        self.query_head_dim = query_head_dim = _to_tuple(query_head_dim)
        self.value_head_dim = value_head_dim = _to_tuple(value_head_dim)
        pos_head_dim = _to_tuple(pos_head_dim)
        self.num_heads = num_heads = _to_tuple(num_heads)
        feedforward_dim = _to_tuple(feedforward_dim)
        self.cnn_module_kernel = cnn_module_kernel = _to_tuple(cnn_module_kernel)

        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames

        for u, d in zip(encoder_unmasked_dim, encoder_dim):
            assert u <= d

        encoders = []

        num_encoders = len(downsampling_factor)
        for i in range(num_encoders):
            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim[i],
                pos_dim=pos_dim,
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                pos_head_dim=pos_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_dim=feedforward_dim[i],
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel[i],
                causal=causal,
            )

            encoder = Zipformer2Encoder(
                encoder_layer,
                num_encoder_layers[i],
                pos_dim=pos_dim,
                dropout=dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
                final_layerdrop_rate=0.035 * (downsampling_factor[i] ** 0.5),
            )

            if downsampling_factor[i] != 1:
                encoder = DownsampledZipformer2Encoder(
                    encoder,
                    dim=encoder_dim[i],
                    downsample=downsampling_factor[i],
                    dropout=dropout,
                )

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        self.downsample_output = SimpleDownsample(
            max(encoder_dim), downsample=output_downsampling_factor, dropout=dropout
        )

    def get_feature_masks(self, x: Tensor) -> Union[List[float], List[Tensor]]:
        num_encoders = len(self.encoder_dim)
        if not self.training:
            return [1.0] * num_encoders

        (num_frames0, batch_size, _encoder_dims0) = x.shape

        assert self.encoder_dim[0] == _encoder_dims0, (
            self.encoder_dim[0],
            _encoder_dims0,
        )

        feature_mask_dropout_prob = 0.125

        mask1 = (
            torch.rand(1, batch_size, 1, device=x.device) > feature_mask_dropout_prob
        ).to(x.dtype)

        mask2 = torch.logical_and(
            mask1,
            (
                torch.rand(1, batch_size, 1, device=x.device)
                > feature_mask_dropout_prob
            ).to(x.dtype),
        )

        mask = torch.cat((mask1, mask2), dim=-1)

        feature_masks = []
        for i in range(num_encoders):
            channels = self.encoder_dim[i]
            feature_mask = torch.ones(
                1, batch_size, channels, dtype=x.dtype, device=x.device
            )
            u1 = self.encoder_unmasked_dim[i]
            u2 = u1 + (channels - u1) // 2

            feature_mask[:, :, u1:u2] *= mask[..., 0:1]
            feature_mask[:, :, u2:] *= mask[..., 1:2]

            feature_masks.append(feature_mask)

        return feature_masks

    def get_chunk_info(self) -> Tuple[int, int]:
        if not self.causal:
            return -1, -1

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            assert len(self.chunk_size) == 1, self.chunk_size
            chunk_size = self.chunk_size[0]
        else:
            chunk_size = random.choice(self.chunk_size)

        if chunk_size == -1:
            left_context_chunks = -1
        else:
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                assert len(self.left_context_frames) == 1, self.left_context_frames
                left_context_frames = self.left_context_frames[0]
            else:
                left_context_frames = random.choice(self.left_context_frames)
            left_context_chunks = left_context_frames // chunk_size
            if left_context_chunks == 0:
                left_context_chunks = 1

        return chunk_size, left_context_chunks

    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        outputs = []
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            feature_masks = [1.0] * len(self.encoder_dim)
        else:
            feature_masks = self.get_feature_masks(x)

        chunk_size, left_context_chunks = self.get_chunk_info()

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            attn_mask = None
        else:
            attn_mask = self._get_attn_mask(x, chunk_size, left_context_chunks)

        for i, module in enumerate(self.encoders):
            ds = self.downsampling_factor[i]
            x = convert_num_channels(x, self.encoder_dim[i])

            x = module(
                x,
                chunk_size=chunk_size,
                feature_mask=feature_masks[i],
                src_key_padding_mask=(
                    None
                    if src_key_padding_mask is None
                    else src_key_padding_mask[..., ::ds]
                ),
                attn_mask=attn_mask,
            )
            outputs.append(x)

        x = self._get_full_dim_output(outputs)
        x = self.downsample_output(x)
        assert self.output_downsampling_factor == 2, self.output_downsampling_factor
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            lengths = (x_lens + 1) // 2
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lengths = (x_lens + 1) // 2

        return x, lengths

    def _get_attn_mask(
        self, x: Tensor, chunk_size: int, left_context_chunks: int
    ) -> Optional[Tensor]:
        if chunk_size <= 0:
            return None
        assert all(chunk_size % d == 0 for d in self.downsampling_factor)
        if left_context_chunks >= 0:
            num_encoders = len(self.encoder_dim)
            assert all(
                chunk_size * left_context_chunks
                >= (self.cnn_module_kernel[i] // 2) * self.downsampling_factor[i]
                for i in range(num_encoders)
            )
        else:
            left_context_chunks = 1000000

        seq_len = x.shape[0]

        t = torch.arange(seq_len, dtype=torch.int32, device=x.device)
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            c = t // chunk_size
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = t // chunk_size
        src_c = c
        tgt_c = c.unsqueeze(-1)

        attn_mask = torch.logical_or(
            src_c > tgt_c, src_c < tgt_c - left_context_chunks
        )
        return attn_mask

    def _get_full_dim_output(self, outputs: List[Tensor]):
        num_encoders = len(self.encoder_dim)
        assert len(outputs) == num_encoders
        output_dim = max(self.encoder_dim)
        output_pieces = [outputs[-1]]
        cur_dim = self.encoder_dim[-1]
        for i in range(num_encoders - 2, -1, -1):
            d = self.encoder_dim[i]
            if d > cur_dim:
                this_output = outputs[i]
                output_pieces.append(this_output[..., cur_dim:d])
                cur_dim = d
        assert cur_dim == output_dim
        return torch.cat(output_pieces, dim=-1)


class Zipformer2EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        dropout: FloatLike = 0.1,
        cnn_module_kernel: int = 31,
        causal: bool = False,
        attention_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0
        ),
        conv_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0
        ),
        const_attention_rate: FloatLike = ScheduledFloat(
            (0.0, 0.25), (4000.0, 0.025), default=0
        ),
        ff2_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)
        ),
        ff3_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)
        ),
        bypass_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.5), (4000.0, 0.02), default=0
        ),
    ) -> None:
        super(Zipformer2EncoderLayer, self).__init__()
        self.embed_dim = embed_dim

        self.bypass = BypassModule(
            embed_dim, skip_rate=bypass_skip_rate, straight_through_rate=0
        )
        self.bypass_mid = BypassModule(embed_dim, straight_through_rate=0)

        self.attention_skip_rate = copy.deepcopy(attention_skip_rate)
        self.conv_skip_rate = copy.deepcopy(conv_skip_rate)

        self.ff2_skip_rate = copy.deepcopy(ff2_skip_rate)
        self.ff3_skip_rate = copy.deepcopy(ff3_skip_rate)

        self.const_attention_rate = copy.deepcopy(const_attention_rate)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            dropout=0.0,
        )

        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.feed_forward1 = FeedforwardModule(
            embed_dim, (feedforward_dim * 3) // 4, dropout
        )

        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim, dropout)

        self.feed_forward3 = FeedforwardModule(
            embed_dim, (feedforward_dim * 5) // 4, dropout
        )

        self.nonlin_attention = NonlinAttention(
            embed_dim, hidden_channels=3 * embed_dim // 4
        )

        self.conv_module1 = ConvolutionModule(
            embed_dim, cnn_module_kernel, causal=causal
        )

        self.conv_module2 = ConvolutionModule(
            embed_dim, cnn_module_kernel, causal=causal
        )

        # TODO: remove it
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))

        self.norm = BiasNorm(embed_dim)

        self.balancer1 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            min_abs=0.2,
            max_abs=4.0,
        )

        self.balancer_na = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.004), (4000.0, 0.02)),
            prob=0.05,
        )

        self.balancer_ff2 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.1), default=0.0),
            max_abs=2.0,
            prob=0.05,
        )

        self.balancer_ff3 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.2), default=0.0),
            max_abs=4.0,
            prob=0.05,
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(4.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.balancer2 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            min_abs=0.1,
            max_abs=4.0,
        )

    def get_sequence_dropout_mask(
        self, x: Tensor, dropout_rate: float
    ) -> Optional[Tensor]:
        if (
            dropout_rate == 0.0
            or not self.training
            or torch.jit.is_scripting()
            or torch.jit.is_tracing()
        ):
            return None
        batch_size = x.shape[1]
        mask = (torch.rand(batch_size, 1, device=x.device) > dropout_rate).to(x.dtype)
        return mask

    def sequence_dropout(self, x: Tensor, dropout_rate: float) -> Tensor:
        dropout_mask = self.get_sequence_dropout_mask(x, dropout_rate)
        if dropout_mask is None:
            return x
        else:
            return x * dropout_mask

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src_orig = src

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            attention_skip_rate = 0.0
        else:
            attention_skip_rate = (
                float(self.attention_skip_rate) if self.training else 0.0
            )

        attn_weights = self.self_attn_weights(
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
        )

        src = src + self.feed_forward1(src)

        self_attn_dropout_mask = self.get_sequence_dropout_mask(
            src, attention_skip_rate
        )

        selected_attn_weights = attn_weights[0:1]
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif self.training and random.random() < float(self.const_attention_rate):
            selected_attn_weights = selected_attn_weights[0:1]
            selected_attn_weights = (selected_attn_weights > 0.0).to(
                selected_attn_weights.dtype
            )
            selected_attn_weights = selected_attn_weights * (
                1.0 / selected_attn_weights.sum(dim=-1, keepdim=True)
            )

        na = self.balancer_na(self.nonlin_attention(src, selected_attn_weights))

        src = src + (
            na if self_attn_dropout_mask is None else na * self_attn_dropout_mask
        )

        self_attn = self.self_attn1(src, attn_weights)

        src = src + (
            self_attn
            if self_attn_dropout_mask is None
            else self_attn * self_attn_dropout_mask
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            conv_skip_rate = 0.0
        else:
            conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.conv_module1(
                src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask
            ),
            conv_skip_rate,
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff2_skip_rate = 0.0
        else:
            ff2_skip_rate = float(self.ff2_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.balancer_ff2(self.feed_forward2(src)), ff2_skip_rate
        )

        src = self.bypass_mid(src_orig, src)

        self_attn = self.self_attn2(src, attn_weights)

        src = src + (
            self_attn
            if self_attn_dropout_mask is None
            else self_attn * self_attn_dropout_mask
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            conv_skip_rate = 0.0
        else:
            conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.conv_module2(
                src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask
            ),
            conv_skip_rate,
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff3_skip_rate = 0.0
        else:
            ff3_skip_rate = float(self.ff3_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.balancer_ff3(self.feed_forward3(src)), ff3_skip_rate
        )

        src = self.balancer1(src)
        src = self.norm(src)

        src = self.bypass(src_orig, src)

        src = self.balancer2(src)
        src = self.whiten(src)

        return src


class Zipformer2Encoder(nn.Module):
    r"""Zipformer2Encoder is a stack of N encoder layers."""

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        pos_dim: int,
        dropout: float,
        warmup_begin: float,
        warmup_end: float,
        initial_layerdrop_rate: float = 0.5,
        final_layerdrop_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(
            pos_dim, dropout_rate=0.15, length_factor=1.0
        )

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

        assert 0 <= warmup_begin <= warmup_end, (warmup_begin, warmup_end)

        delta = (1.0 / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin
        for i in range(num_layers):
            cur_end = cur_begin + delta
            self.layers[i].bypass.skip_rate = ScheduledFloat(
                (cur_begin, initial_layerdrop_rate),
                (cur_end, final_layerdrop_rate),
                default=0.0,
            )
            cur_begin = cur_end

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        pos_emb = self.encoder_pos(src)
        output = src

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            output = output * feature_mask

        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                pos_emb,
                chunk_size=chunk_size,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                output = output * feature_mask

        return output


class BypassModule(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        skip_rate: FloatLike = 0.0,
        straight_through_rate: FloatLike = 0.0,
        scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2), default=0),
        scale_max: FloatLike = 1.0,
    ):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
        self.skip_rate = copy.deepcopy(skip_rate)
        self.straight_through_rate = copy.deepcopy(straight_through_rate)
        self.scale_min = copy.deepcopy(scale_min)
        self.scale_max = copy.deepcopy(scale_max)

    def _get_bypass_scale(self, batch_size: int):
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(
                self.bypass_scale, min=float(self.scale_min), max=float(self.scale_max)
            )
            skip_rate = float(self.skip_rate)
            if skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > skip_rate
                ans = ans * mask
            straight_through_rate = float(self.straight_through_rate)
            if straight_through_rate != 0.0:
                mask = (
                    torch.rand((batch_size, 1), device=ans.device)
                    < straight_through_rate
                )
                ans = torch.maximum(ans, mask.to(ans.dtype))
            return ans

    def forward(self, src_orig: Tensor, src: Tensor):
        bypass_scale = self._get_bypass_scale(src.shape[1])
        return src_orig + (src - src_orig) * bypass_scale


class DownsampledZipformer2Encoder(nn.Module):
    r"""
    DownsampledZipformer2Encoder is a zipformer encoder evaluated at a reduced
    frame rate, after convolutional downsampling, and then upsampled again at
    the output.
    """

    def __init__(
        self, encoder: nn.Module, dim: int, downsample: int, dropout: FloatLike
    ):
        super(DownsampledZipformer2Encoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = SimpleDownsample(dim, downsample, dropout)
        self.num_layers = encoder.num_layers
        self.encoder = encoder
        self.upsample = SimpleUpsample(dim, downsample)
        self.out_combiner = BypassModule(dim, straight_through_rate=0)

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src_orig = src
        src = self.downsample(src)
        ds = self.downsample_factor
        if attn_mask is not None:
            attn_mask = attn_mask[::ds, ::ds]

        src = self.encoder(
            src,
            chunk_size=chunk_size // ds,
            feature_mask=feature_mask,
            attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        src = src[: src_orig.shape[0]]

        return self.out_combiner(src_orig, src)


class SimpleDownsample(torch.nn.Module):
    def __init__(self, channels: int, downsample: int, dropout: FloatLike):
        super(SimpleDownsample, self).__init__()

        self.bias = nn.Parameter(torch.zeros(downsample))

        self.name = None
        self.dropout = copy.deepcopy(dropout)

        self.downsample = downsample

    def forward(self, src: Tensor) -> Tensor:
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        pad = d_seq_len * ds - seq_len
        src_extra = src[src.shape[0] - 1 :].expand(pad, src.shape[1], src.shape[2])
        src = torch.cat((src, src_extra), dim=0)
        assert src.shape[0] == d_seq_len * ds

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)

        weights = self.bias.softmax(dim=0)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        ans = (src * weights).sum(dim=1)

        return ans


class SimpleUpsample(torch.nn.Module):
    def __init__(self, num_channels: int, upsample: int):
        super(SimpleUpsample, self).__init__()
        self.upsample = upsample

    def forward(self, src: Tensor) -> Tensor:
        upsample = self.upsample
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
        src = src.reshape(seq_len * upsample, batch_size, num_channels)
        return src


class CompactRelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (compact version)."""

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: FloatLike,
        max_len: int = 1000,
        length_factor: float = 1.0,
    ) -> None:
        super(CompactRelPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0, embed_dim
        self.dropout = Dropout2(dropout_rate)
        self.pe = None
        assert length_factor >= 1.0, length_factor
        self.length_factor = length_factor
        self.extend_pe(torch.tensor(0.0).expand(max_len))

    def extend_pe(self, x: Tensor, left_context_len: int = 0) -> None:
        T = x.size(0) + left_context_len

        if self.pe is not None:
            if self.pe.size(0) >= T * 2 - 1:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        x = torch.arange(-(T - 1), T, device=x.device).to(torch.float32).unsqueeze(1)

        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)

        compression_length = self.embed_dim**0.5
        x_compressed = (
            compression_length
            * x.sign()
            * ((x.abs() + compression_length).log() - math.log(compression_length))
        )

        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

        x_atan = (x_compressed / length_scale).atan()

        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()

        pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0

        self.pe = pe.to(dtype=x.dtype)

    def forward(self, x: Tensor, left_context_len: int = 0) -> Tensor:
        self.extend_pe(x, left_context_len)
        x_size_left = x.size(0) + left_context_len
        pos_emb = self.pe[
            self.pe.size(0) // 2
            - x_size_left
            + 1 : self.pe.size(0) // 2  # noqa E203
            + x.size(0),
            :,
        ]
        pos_emb = pos_emb.unsqueeze(0)
        return self.dropout(pos_emb)


class RelPositionMultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head attention weights with relative position encoding."""

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        dropout: float = 0.0,
        pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.0)),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)
        self.name = None

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads

        self.in_proj = ScaledLinear(
            embed_dim, in_proj_dim, bias=True, initial_scale=query_head_dim**-0.25
        )

        self.whiten_keys = Whiten(
            num_groups=num_heads,
            whitening_limit=_whitening_schedule(3.0),
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )

        self.balance_keys = Balancer(
            key_head_dim * num_heads,
            channel_dim=-1,
            min_positive=0.4,
            max_positive=0.6,
            min_abs=0.0,
            max_abs=100.0,
            prob=0.025,
        )

        self.linear_pos = ScaledLinear(
            pos_dim, num_heads * pos_head_dim, bias=False, initial_scale=0.05
        )

        self.copy_pos_query = Identity()
        self.copy_query = Identity()

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]
        p = x[..., 2 * query_dim :]
        assert p.shape[-1] == num_heads * pos_head_dim, (
            p.shape[-1],
            num_heads,
            pos_head_dim,
        )

        q = self.copy_query(q)
        k = self.whiten_keys(self.balance_keys(k))
        p = self.copy_pos_query(p)

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)

        q = q.permute(2, 1, 0, 3)
        p = p.permute(2, 1, 0, 3)
        k = k.permute(2, 1, 3, 0)

        attn_scores = torch.matmul(q, k)

        use_pos_scores = False
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            use_pos_scores = True
        elif not self.training or random.random() >= float(self.pos_emb_skip_rate):
            use_pos_scores = True

        if use_pos_scores:
            pos_emb = self.linear_pos(pos_emb)
            seq_len2 = 2 * seq_len - 1
            pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(
                2, 0, 3, 1
            )

            pos_scores = torch.matmul(p, pos_emb)
            if torch.jit.is_tracing():
                (num_heads, batch_size, time1, n) = pos_scores.shape
                rows = torch.arange(start=time1 - 1, end=-1, step=-1)
                cols = torch.arange(seq_len)
                rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
                indexes = rows + cols
                pos_scores = pos_scores.reshape(-1, n)
                pos_scores = torch.gather(pos_scores, dim=1, index=indexes)
                pos_scores = pos_scores.reshape(num_heads, batch_size, time1, seq_len)
            else:
                pos_scores = pos_scores.as_strided(
                    (num_heads, batch_size, seq_len, seq_len),
                    (
                        pos_scores.stride(0),
                        pos_scores.stride(1),
                        pos_scores.stride(2) - pos_scores.stride(3),
                        pos_scores.stride(3),
                    ),
                    storage_offset=pos_scores.stride(3) * (seq_len - 1),
                )

            attn_scores = attn_scores + pos_scores

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif self.training and random.random() < 0.1:
            attn_scores = penalize_abs_values_gt(
                attn_scores, limit=25.0, penalty=1.0e-04, name=self.name
            )

        assert attn_scores.shape == (num_heads, batch_size, seq_len, seq_len)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            attn_scores = attn_scores.masked_fill(attn_mask, -1000)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                batch_size,
                seq_len,
            ), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        attn_weights = softmax(attn_scores, dim=-1)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif random.random() < 0.001 and not self.training:
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        return attn_weights

    def _print_attn_entropy(self, attn_weights: Tensor):
        (num_heads, batch_size, seq_len, seq_len) = attn_weights.shape

        with torch.no_grad():
            with torch_autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
                attn_weights_entropy = (
                    -((attn_weights + 1.0e-20).log() * attn_weights)
                    .sum(dim=-1)
                    .mean(dim=(1, 2))
                )
                logging.info(
                    f"name={self.name}, attn_weights_entropy = {attn_weights_entropy}"
                )


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        value_head_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)

        self.out_proj = ScaledLinear(
            num_heads * value_head_dim, embed_dim, bias=True, initial_scale=0.05
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = self.in_proj(x)
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        value_head_dim = x.shape[-1]

        x = torch.matmul(attn_weights, x)

        x = (
            x.permute(2, 1, 0, 3)
            .contiguous()
            .view(seq_len, batch_size, num_heads * value_head_dim)
        )

        x = self.out_proj(x)
        x = self.whiten(x)

        return x


class FeedforwardModule(nn.Module):
    """Feedforward module in Zipformer2 model."""

    def __init__(self, embed_dim: int, feedforward_dim: int, dropout: FloatLike):
        super(FeedforwardModule, self).__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)

        self.hidden_balancer = Balancer(
            feedforward_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=1.0,
            min_abs=0.75,
            max_abs=5.0,
        )

        self.out_proj = ActivationDropoutAndLinear(
            feedforward_dim,
            embed_dim,
            activation="SwooshL",
            dropout_p=dropout,
            dropout_shared_dim=0,
            bias=True,
            initial_scale=0.1,
        )

        self.out_whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(self, x: Tensor):
        x = self.in_proj(x)
        x = self.hidden_balancer(x)
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class NonlinAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels

        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)

        self.balancer = Balancer(
            hidden_channels,
            channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.25), (20000.0, 0.05)),
            max_positive=ScheduledFloat((0.0, 0.75), (20000.0, 0.95)),
            min_abs=0.5,
            max_abs=5.0,
        )
        self.tanh = nn.Tanh()

        self.identity1 = Identity()
        self.identity2 = Identity()
        self.identity3 = Identity()

        self.out_proj = ScaledLinear(
            hidden_channels, channels, bias=True, initial_scale=0.05
        )

        self.whiten1 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.whiten2 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=2)

        s = self.balancer(s)
        s = self.tanh(s)

        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = self.whiten1(x)
        x = x * s
        x = self.identity1(x)

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        x = torch.matmul(attn_weights, x)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)

        y = self.identity2(y)
        x = x * y
        x = self.identity3(x)

        x = self.out_proj(x)
        x = self.whiten2(x)
        return x


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer2 model."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        causal: bool,
    ) -> None:
        super(ConvolutionModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0

        bottleneck_dim = channels
        self.causal = causal

        self.in_proj = nn.Linear(
            channels,
            2 * bottleneck_dim,
        )

        self.balancer1 = Balancer(
            bottleneck_dim,
            channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.05), (8000.0, 0.025)),
            max_positive=1.0,
            min_abs=1.5,
            max_abs=ScheduledFloat((0.0, 5.0), (8000.0, 10.0), default=1.0),
        )

        self.activation1 = Identity()

        self.sigmoid = nn.Sigmoid()

        self.activation2 = Identity()

        assert kernel_size % 2 == 1

        self.depthwise_conv = (
            ChunkCausalDepthwiseConv1d(channels=bottleneck_dim, kernel_size=kernel_size)
            if causal
            else nn.Conv1d(
                in_channels=bottleneck_dim,
                out_channels=bottleneck_dim,
                groups=bottleneck_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

        self.balancer2 = Balancer(
            bottleneck_dim,
            channel_dim=1,
            min_positive=ScheduledFloat((0.0, 0.1), (8000.0, 0.05)),
            max_positive=1.0,
            min_abs=ScheduledFloat((0.0, 0.2), (20000.0, 0.5)),
            max_abs=10.0,
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.out_proj = ActivationDropoutAndLinear(
            bottleneck_dim,
            channels,
            activation="SwooshR",
            dropout_p=0.0,
            initial_scale=0.05,
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        chunk_size: int = -1,
    ) -> Tensor:
        x = self.in_proj(x)

        x, s = x.chunk(2, dim=2)
        s = self.balancer1(s)
        s = self.sigmoid(s)
        x = self.activation1(x)
        x = x * s
        x = self.activation2(x)

        x = x.permute(1, 2, 0)

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and chunk_size >= 0
        ):
            assert (
                self.causal
            ), "Must initialize model with causal=True if you use chunk_size"
            x = self.depthwise_conv(x, chunk_size=chunk_size)
        else:
            x = self.depthwise_conv(x)

        x = self.balancer2(x)
        x = x.permute(2, 0, 1)

        x = self.whiten(x)
        x = self.out_proj(x)

        return x


class ScalarMultiply(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


# ============================================================================
# load.py components
# ============================================================================


class _AttrDict(dict):
    """Simple attribute-access dict, replacing utils.AttributeDict."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class ZipformerEncoderModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
    ):
        super().__init__()

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, x_lens = self.encoder_embed(x, x_lens)

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)
    return expaned_lengths >= lengths.unsqueeze(-1)


def _to_int_tuple(s):
    if isinstance(s, str):
        return tuple(map(int, s.split(",")))
    if isinstance(s, (list, tuple)):
        return tuple(int(x) for x in s)
    return (int(s),)


def get_model_params(type: str = "custom") -> _AttrDict:
    """Return a dict containing Zipformer model parameters."""
    custom_params = _AttrDict(
        {
            "feature_dim": 80,
            "num_encoder_layers": "2,2,4,5,4,2",
            "downsampling_factor": "1,2,4,8,4,2",
            "feedforward_dim": "768,1024,1536,2048,1536,1024",
            "num_heads": "8,8,8,12,8,8",
            "encoder_dim": "256,384,512,768,512,384",
            "query_head_dim": "32",
            "value_head_dim": "12",
            "pos_head_dim": "4",
            "pos_dim": 48,
            "encoder_unmasked_dim": "192,256,320,512,320,256",
            "cnn_module_kernel": "31,31,15,15,15,31",
            "causal": True,
            "chunk_size": "32",
            "left_context_frames": "256",
        }
    )

    custom_noncausal_params = _AttrDict(
        {
            "feature_dim": 80,
            "num_encoder_layers": "2,2,4,5,4,2",
            "downsampling_factor": "1,2,4,8,4,2",
            "feedforward_dim": "768,1024,1536,2048,1536,1024",
            "num_heads": "8,8,8,12,8,8",
            "encoder_dim": "256,384,512,768,512,384",
            "query_head_dim": "32",
            "value_head_dim": "12",
            "pos_head_dim": "4",
            "pos_dim": 48,
            "encoder_unmasked_dim": "192,256,320,512,320,256",
            "cnn_module_kernel": "31,31,15,15,15,31",
            "causal": False,
            "chunk_size": "-1",
            "left_context_frames": "-1",
        }
    )

    if type == "custom":
        return custom_params
    elif type == "custom_noncausal":
        return custom_noncausal_params
    else:
        raise ValueError(f"Unsupported model type: {type}")


def get_encoder_embed(params) -> nn.Module:
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


def get_encoder_model(params) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder


# ============================================================================
# Factory function
# ============================================================================


def build_zipformer_encoder(config_dict: dict):
    """Build a ZipformerEncoderModel from a HF config dict."""
    params = dict(config_dict)
    params.pop("model_type", None)
    params.pop("transformers_version", None)

    p = _AttrDict(params)
    if "feature_dim" not in p:
        p["feature_dim"] = 80

    encoder_embed = get_encoder_embed(p)
    encoder = get_encoder_model(p)
    return ZipformerEncoderModel(encoder_embed, encoder)
