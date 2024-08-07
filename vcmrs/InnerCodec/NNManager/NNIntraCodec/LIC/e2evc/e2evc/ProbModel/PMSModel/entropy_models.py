# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import numpy as np
import scipy.stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from entropy_model_ext import \
    pmf_to_quantized_cdf as _pmf_to_quantized_cdf  # pylint: disable=E0611,E0401
from .bound_ops import LowerBound
import ans

from e2evc.Quantizer import UniQuantizerFixedDelta
from e2evc.Dequantizer import UniDequantizerDelta

# channel dependencies
#from .chnl_dep import *

def default_entropy_coder():
    return 'rans'


def pmf_to_quantized_cdf(pmf, precision=16):
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    cdf = cdf.to(pmf.device)
    return cdf


class EntropyModel(nn.Module):
    r"""Entropy model base class.

    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
        quant_train: quantization training strategy 'noise' or 'ste'
    """
    def __init__(self,
                 likelihood_bound=1e-9,
                 entropy_coder=None,
                 entropy_coder_precision=16,
                 quant_train='noise'):
        super().__init__()

        #self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.quantizer = UniQuantizerFixedDelta(method=quant_train)
        self.dequantizer = UniDequantizerDelta()

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        # to be filled on update()
        self.register_buffer('_offset', torch.IntTensor())
        self.register_buffer('_quantized_cdf', torch.IntTensor())
        self.register_buffer('_cdf_length', torch.IntTensor())

    def forward(self, *args):
        raise NotImplementedError()

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[:pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, :_cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError('Uninitialized CDFs. Run update() first')

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f'Invalid CDF size {self._quantized_cdf.size()}')

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError('Uninitialized offsets. Run update() first')

        if len(self._offset.size()) != 1:
            raise ValueError(f'Invalid offsets size {self._offset.size()}')

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError('Uninitialized CDF lengths. Run update() first')

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f'Invalid offsets size {self._cdf_length.size()}')

    def compress(self, bitstream, inputs, indexes, means=None, oor_bypass=False):
        """
        Compress input tensors to char strings.

        Args:
            bitstream (EntropyCoder): entropy coder
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """

        if means is not None: inputs = inputs - means
        symbols = self.quantizer(inputs)

        assert len(inputs.size()) == 4, \
            'Invalid `inputs` size. Expected a 4-D tensor.'

        assert inputs.size(0) == 1, \
                'Compressing multiple images are not supported!'

        assert inputs.size() == indexes.size(), \
                '`inputs` and `indexes` should have the same size.'

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        for i in range(symbols.size(0)):
            bitstream.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
                oor_bypass=oor_bypass)

    def decompress(self, bitstream, indexes, means=None, oor_bypass=False):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """

        assert indexes.size(0) == 1, \
            'Multiple images are not supported'

        assert len(indexes.size()) == 4, \
            'Invalid `indexes` size. Expected a 4-D tensor.'

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        device = self._offset.device

        if means is not None:
            device = means.device
            assert means.size()[:-2] == indexes.size()[:-2], \
              'Invalid means or indexes parameters'

            assert (means.size() == indexes.size()) or \
                    (means.size(2) == 1) or (means.size(3) == 1), \
                   'Invalid means parameters'


        cdf = self._quantized_cdf

        values = bitstream.decode_stream(
                indexes[0].reshape(-1).int().tolist(), 
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
                oor_bypass=oor_bypass)

        outputs = torch.tensor(values).reshape(indexes.size()).to(device)

        outputs = self.dequantizer(outputs)
        if means is not None: outputs = outputs+means

        return outputs


class EntropyBottleneck(EntropyModel):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://tensorflow.github.io/compression/docs/entropy_bottleneck.html>`_
    for an introduction.
    """
    def __init__(self,
                 channels,
                 *args,
                 tail_mass=1e-9,
                 init_scale=10,
                 filters=(3, 3, 3, 3),
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        filters = (1, ) + self.filters + (1, )
        scale = self.init_scale**(1 / (len(self.filters) + 1))
        channels = self.channels

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))


            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer('target', torch.Tensor([-target, 0, target]))

    def _get_medians(self):
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force=False):
        
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:  # pylint: disable=E0203
            return

        #perform calculation on CPU to achieve numerical stability
        quantiles = self.quantiles.cpu()

        medians = quantiles[:, 0, 1]

        minima = medians - quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima.to(self.quantiles.device)

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max()
        samples = torch.arange(max_length).to(quantiles.device)

        samples = samples[None, :] + pmf_start[:, None, None]

        half = float(0.5)

        lower = self._logits_cumulative(samples - half, stop_gradient=True)
        upper = self._logits_cumulative(samples + half, stop_gradient=True)
        sign = -torch.sign(lower + upper)
        pmf = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) +\
            torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length,
                                         max_length)
        self._quantized_cdf = quantized_cdf.to(self.quantiles.device)
        self._cdf_length = pmf_length + 2


    def loss(self):
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs, stop_gradient):
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}").to(inputs.device)
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = getattr(self, f"_bias{i:d}").to(inputs.device)
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}").to(inputs.device)
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    @torch.jit.unused
    def _likelihood(self, inputs):
        half = float(0.5)
        v0 = inputs - half
        v1 = inputs + half
        lower = self._logits_cumulative(v0, stop_gradient=False)
        upper = self._logits_cumulative(v1, stop_gradient=False)
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        return likelihood

    def forward(self, x):
        # Convert to (channels, ... , batch) format
        x = x.permute(1, 2, 3, 0).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # quantization
        # original implementation means are not added in noise mode, since it will 
        # be substracted after dequantization. 
        means = self._get_medians()
        values = values - means 
        outputs = self.quantizer(values)
        outputs = self.dequantizer(outputs)
        outputs = outputs + means

        if not torch.jit.is_scripting():
            likelihood = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            # TorchScript not yet supported
            likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(3, 0, 1, 2).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(3, 0, 1, 2).contiguous()

        # calcualte cross entropy
        bits = -likelihood.log2()

        #return outputs, entropy
        return outputs, bits

    @staticmethod
    def _build_indexes(size):
        N, C, H, W = size
        indexes = torch.arange(C).view(1, -1, 1, 1)
        indexes = indexes.int()
        return indexes.repeat(N, 1, H, W)

    def compress(self, bitstream, x):
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach().view(1, -1, 1, 1)
        return super().compress(bitstream, x, indexes, medians, oor_bypass=True)

    def decompress(self, bitstream, size):
        output_size = (1, self._quantized_cdf.size(0), size[0],
                       size[1])
        indexes = self._build_indexes(output_size)
        medians = self._get_medians().detach().view(1, -1, 1, 1)

        return super().decompress(bitstream, indexes, medians, oor_bypass=True)


class GaussianConditional(EntropyModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`_.
    """
    def __init__(self,
                 scale_table,
                 *args,
                 scale_bound=0.11,
                 tail_mass=1e-9,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(
                f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(
                f'Invalid scale_table length "{len(scale_table)}"')

        if scale_table and \
                (scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        self.register_buffer(
            'scale_table',
            self._prepare_scale_table(scale_table)
            if scale_table else torch.Tensor())

        self.register_buffer(
            'scale_bound',
            torch.Tensor([float(scale_bound)])
            if scale_bound is not None else None)

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            self.lower_bound_scale = LowerBound(self.scale_table[0])
        elif scale_bound > 0:
            self.lower_bound_scale = LowerBound(scale_bound)
        else:
            raise ValueError('Invalid parameters')


    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs):
        # type: (Tensor) -> Tensor
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        self.scale_table = self._prepare_scale_table(scale_table)
        if self._offset.numel() > 0 and not force:
            return

        self.update()

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        samples = torch.abs(
            torch.arange(max_length).int() - pmf_center[:, None])
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length,
                                         max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(self, inputs, scales, means=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower

        return likelihood

    def forward(self, inputs, scales, means=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]

        # quantization
        if means is not None: inputs = inputs - means
        outputs = self.quantizer(inputs)
        outputs = self.dequantizer(outputs)
        if means is not None: outputs = outputs + means

        # apply channel dependency module
        # channel dependency should be applied after quantizaiton
        #means = self.chnl_deps(outputs, means)
        # Fixme: scales? 

        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)

        #entropy = -likelihood.log2().sum() 
        bits = -likelihood.log2()

        #return outputs, entropy
        return outputs, bits

    def build_indexes(self, scales):
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(),
                                  len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes
