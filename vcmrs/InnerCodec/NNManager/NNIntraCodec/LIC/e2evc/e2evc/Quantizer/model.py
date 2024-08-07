# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import torch
from torch import nn
from torch.autograd import Function

# Uniform quantization function
class UniQuantizeFun(Function):
  '''Uniform quantization function with straight through estimator
  '''
  @staticmethod
  def forward(ctx, x, min_val, max_val, levels):
     #ctx.save_for_backward(x)
     delta = (max_val - min_val)/levels + 1E-10 # prevent zero
     # returns a integer from 0 to levels
     x = torch.clamp(torch.floor((x - min_val)/delta), min=0, max=levels-1)
     return x
     #return x.int()

  @staticmethod
  def backward(ctx, grad_output):
    # this function must return grad to all inputs appeared in forward
    grad_min_val = grad_max_val = grad_levels = None
    grad_x = grad_output.clone()
    return grad_x, grad_min_val, grad_max_val, grad_levels

# Uniform quantization function
uni_quantizer_fun = UniQuantizeFun.apply


# Uniform quantization function
class UniQuantizeDeltaFun(Function):
  '''Uniform quantization function with a delta parameter with straight through estimator
  '''
  @staticmethod
  def forward(ctx, x, delta):
     #ctx.save_for_backward(x)
     # round operation
     x = torch.round(x / delta)
     return x

  @staticmethod
  def backward(ctx, grad_output):
    # this function must return grad to all inputs appeared in forward
    grad_min_val = grad_max_val = grad_levels = None
    grad_x = grad_output.clone()
    return grad_x, grad_min_val, grad_max_val, grad_levels

# Uniform quantization function
uni_quantizer_delta_fun = UniQuantizeDeltaFun.apply


# Uniform quantization module
class UniQuantizer(nn.Module):
  '''Uniform quantizer for all elements in the tensor
  '''

  def __init__(self, levels=256, retain_constant=0.99):
    '''
    Parameters
    -----------------
    levels : number of quantization levels, default 256
    retain_constant : the weight for moving average method to retain the historic data, default 0.99
    '''
    super(UniQuantizer, self).__init__()

    assert levels>1, f"quantization levels must be greater than 1 : leves={levels}"
    assert retain_constant>0, f"retain constant must be greater than 0 : retain_constant={retain_constant}"

    self.register_buffer('min_val', torch.tensor(float('nan')))
    self.register_buffer('max_val', torch.tensor(float('nan')))

    self.levels = levels
    self.retain_constant = retain_constant

  def extra_repr(self):
    return f'levels: {self.levels}, min: {self.min_val}, max: {self.max_val}'

  def forward(self, x):
    '''uniform quantization
    '''
    # update min and max in forward loop
    if self.training:
      with torch.no_grad():
        # moving average update
        if not torch.isnan(self.min_val):
          self.min_val.fill_(self.min_val * self.retain_constant + \
            (1-self.retain_constant) * torch.min(x))
        else:
          self.min_val.fill_(torch.min(x))

        if not torch.isnan(self.max_val):
          self.max_val.fill_(self.max_val * self.retain_constant + \
            (1-self.retain_constant) * torch.max(x))
        else:
          self.max_val.fill_(torch.max(x))
    #uniform quantization
    x = uni_quantizer_fun(x, self.min_val, self.max_val, self.levels)
    return x

# Uniform quantization module with fixed interval [-1, 1]
class UniQuantizerFixed(nn.Module):
  '''Uniform quantizer for all elements in a tensor
  '''

  def __init__(self, levels=256):
    '''
    Parameters
    -----------------
    levels : number of quantization levels, default 256
    '''
    super(UniQuantizerFixed, self).__init__()

    assert levels>1, f"quantization levels must be greater than 1 : leves={levels}"

    self.levels = levels

  def extra_repr(self):
    return f'UniQuantizerFixed levels: {self.levels}'

  def forward(self, x):
    '''uniform quantization
    '''
    #uniform quantization
    x = uni_quantizer_fun(x, -1, 1, self.levels)
    return x

# unifrom quantizaiton with fixed delta
class UniQuantizerFixedDelta(nn.Module):
  '''Uniform quantizer with fixed delta
     output = [input / delta]
  '''
  def __init__(self, delta=1, method='noise'):
    '''
    Parameters
    -----------------------
    delta: default 1 which means round operation
    method: 
      'ste': straight through estimation
      'noise': adding noise       
    '''
    super().__init__()

    assert delta > 0, f"delta must be a positive number"
    assert method in ['ste', 'noise'], f"only ste and noise method are supported"

    self.delta = delta
    self.method = method

    if method=='noise':
      self._noise = None

  def _get_noise_cached(self, x):
    if self._noise is None:
      self._noise = torch.zeros_like(x, requires_grad=False)
      #self._noise = x.new(x.size())
    self._noise.resize_(x.size())
    self._noise.uniform_(-0.5, 0.5)
    return self._noise

  def extra_repr(self):
    return f"UniQuantizerFixedDelta, delta: {self.delta}"

  def forward(self, x):
    if self.training and self.method=='noise':
      x = x + self._get_noise_cached(x)
    else:
      x = uni_quantizer_delta_fun(x, self.delta)
    return x
