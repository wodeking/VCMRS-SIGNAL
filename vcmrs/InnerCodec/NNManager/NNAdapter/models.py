# This file is covered by the license agreement found in the file “license.txt” in the root of this project.
import torch
from torch import nn
from e2evc.Utils import utils_nn

class injectorBlock(nn.Module):
  def __init__(self, \
      in_channels, 
      out_channels,
      inj_operator="concatenate"):
    super().__init__()
    self.blocks = nn.ModuleList()
    self.injection_type=inj_operator
    if inj_operator == "multiply":
      self.inj_operator = torch.mul
    elif inj_operator == "add":
      self.inj_operator = torch.add
    elif inj_operator == "concatenate":
      self.inj_operator = torch.cat
    else:
      raise TypeError(f"inj_operator can currently be only 'multiply', 'add' or 'concatenate' found {inj_operator}")
    self.blocks.append(nn.Linear(in_channels, out_channels))
    self.blocks.append(nn.PReLU())
    self.blocks.append(nn.Sequential())

  def forward(self, x):
    for m1 in self.blocks:
      x = m1(x)
    return x

class EncoderD(nn.Module):

  def __init__(self, 
      n_channels,
      strides,
      num_blocks = 5,
      bn='None',
      skip_connection="conv2d",
      lateral_connection="None",
      injections=[1,1],
      inj_out_chn=4,
      inj_operator="concatenate"):
    '''Initialize the encoder

    Parameters
    ----------------
    - `n_channels`: list of channel numbers for the layers
    - `strides`: list of stride of the layers
    - `bn` : `'None'` or `'group'`
    - `skip_connection`: can be either `identity` or `conv2d`
    - `lateral_connection`: can be either `None`, `identity` or `conv2d`
    '''
    super(EncoderD, self).__init__()
    
    self.n_channels = n_channels
    self.lateral_connection = lateral_connection
    io_channels = zip(n_channels, n_channels[1:], strides)
    modules = []
    inj_amt=injections.count(1)
    if inj_amt==0:
      inj_out_chn=0
    for i, (ic, oc, stride) in enumerate(io_channels):
      if i < len(strides) - 1:
        if inj_amt != 0:
          modules.append(injectorBlock(in_channels=inj_amt, 
                                                out_channels=inj_out_chn,
                                                inj_operator=inj_operator))
        modules.append(utils_nn.IntConv2d(ic+inj_out_chn, oc, kernel_size=3, stride=stride, padding=1))
        modules.append(utils_nn.BN(oc, bn, 16)) # no BN by default
        modules.append(nn.PReLU())
        modules.append(nn.Identity()) # Placeholder for lateral connections
        modules.append(utils_nn.ResBlocks(oc, num_blocks, skip_connection=skip_connection)) #no BN by default
        
      else: # Last layer
        modules.append(utils_nn.IntConv2d(ic, oc, kernel_size=3, stride=stride, padding=1))

    self.network = nn.Sequential(*modules)

  def forward(self, x,  injection_data=None):
    lateral_inputs = []
    for layer in self.network:
      if hasattr(layer, 'inj_operator'):
        l_out=layer(injection_data).unsqueeze(2).unsqueeze(3).repeat(1,1,x.shape[2],x.shape[3])
        if layer.injection_type == "concatenate":
          x = layer.inj_operator((x, l_out), 1)
        else:
          x = layer.inj_operator(x, l_out)  
      else:
        x = layer(x)
      if isinstance(layer, nn.Identity):
        if self.lateral_connection != "None":
          lateral_inputs.append(x)

    return x, lateral_inputs

class DecoderD(nn.Module):
  def __init__(self, \
      n_channels,
      strides,
      num_blocks = 5,
      bn='None',
      skip_connection="conv2d",
      lateral_connection="None",
      injections=[1,1],
      inj_out_chn=4,
      inj_operator="concatenate"):
    """
      See parameters documentation in class `EncoderD`
    """
    super(DecoderD, self).__init__()
    
    self.n_channels = n_channels
    self.lateral_connections = nn.ModuleList() #The elements of this list will be registered as model's params
    io_channels = zip(n_channels, n_channels[1:], strides)
    modules = []
    inj_amt=injections.count(1)
    if inj_amt==0:
      inj_out_chn=0
    for i, (ic, oc, stride) in enumerate(io_channels):
      if i < len(strides) - 1:
        if inj_amt != 0:
          modules.append(injectorBlock(in_channels=inj_amt, 
                                                out_channels=inj_out_chn,
                                                inj_operator=inj_operator))
        modules.append(utils_nn.IntTransposedConv2d(ic+inj_out_chn, oc, kernel_size=3, stride=stride, padding=1, output_padding=1))
        modules.append(utils_nn.BN(oc, bn, 16))
        modules.append(nn.PReLU())
        modules.append(nn.Identity()) # Placeholder for lateral connections
        modules.append(utils_nn.ResBlocks(oc, num_blocks, skip_connection=skip_connection)) #no BN by default
        
        if lateral_connection != "None":
          # NOTE: assuming the decoder mirrors the encoder
          con = utils_nn.IntConv2d(n_channels[i+1], oc, kernel_size=1) if skip_connection == "conv2d" else nn.Identity()
          self.lateral_connections.append(con)
      else:
        if inj_amt != 0:
          modules.append(injectorBlock(in_channels=inj_amt, 
                                                out_channels=inj_out_chn,
                                                inj_operator=inj_operator))
        modules.append(utils_nn.IntTransposedConv2d(ic+inj_out_chn, 3, kernel_size=3, stride=stride, padding=1, output_padding=1))

    self.network = nn.Sequential(*modules)
    
   
  def forward(self, x, lateral_inputs=[], injection_data=None):
    lateral_inputs = lateral_inputs[::-1] #Reversed order
    lateral_idx = -1
    for layer in self.network:
      if hasattr(layer, 'inj_operator'):
        l_out=layer(injection_data).unsqueeze(2).unsqueeze(3).repeat(1,1,x.shape[2],x.shape[3])
        if layer.injection_type == "concatenate":
          x = layer.inj_operator((x, l_out), 1)
        else:
          x = layer.inj_operator(x, l_out)  
      else:
        x = layer(x)
      if len(self.lateral_connections) > 0:
        if isinstance(layer, nn.Identity):
          lateral_idx += 1
          lateral = self.lateral_connections[lateral_idx](lateral_inputs[lateral_idx])
          x = x + lateral
        
    return x

class Autoencoder(nn.Module):
  def __init__(self, en_layer_channels, en_strides, resblocks_size, skip_connection="conv2d",
   lateral_connection="conv2d", injections=[1,1], inj_out_chn=4, inj_operator="concatenate"):
    """
      The decoder will be the mirror of the encoder configs, except output has 3 channels always
      
      Params:
        - `en_layer_channels`: list of channel numbers for the layers in the encoder
        - `en_strides`: list of strides of the layers in the encoder
        - `bn` : Batch normalization layer. Can be 'None' or 'group'
        - `skip_connection`: can be either `identity` or `conv2d`
        - `lateral`: can be either `None`, `identity` or `conv2d`
    """
    super().__init__()
    de_strides = en_strides[::-1]
    de_channels = en_layer_channels[::-1]
    assert len(en_layer_channels) > 1, f"The encoder should have more than one layer. Given {len(en_layer_channels)}."
    assert de_channels[0] == en_layer_channels[-1], f"Mismatched numbers of latent channels in encoder and decoder ({de_channels[0]} vs {en_layer_channels[-1]})."
    assert len(en_strides) == len(en_layer_channels) - 1, f"Numbers of strides and channels are not compatible in the encoder."
    
    self.encoder = EncoderD(n_channels=en_layer_channels, strides=en_strides, num_blocks=resblocks_size,
     skip_connection=skip_connection, lateral_connection=lateral_connection, injections=injections, inj_out_chn=inj_out_chn, inj_operator=inj_operator)

    self.decoder = DecoderD(n_channels=de_channels, strides=de_strides, num_blocks=resblocks_size,
     skip_connection=skip_connection, lateral_connection=lateral_connection, injections=injections, inj_out_chn=inj_out_chn, inj_operator=inj_operator)
  
  def forward(self,x,injection_data=None):
    x_orig = x
    y, lateral_inputs = self.encoder(x, injection_data)
    x_hat = self.decoder(y, lateral_inputs, injection_data)
    x_hat = x_hat + x_orig[:,0:x_hat.shape[1],:,:] # lateral connection
    return x_hat
