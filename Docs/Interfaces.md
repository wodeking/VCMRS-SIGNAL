# Pre- and post- inner codec components
The pre- and post- inner codec components shall be implemented as plugins to the VCMRS. The plugin interface is defined in the file `vcmrs/Utils/component.py`. 

Each plugin shall implement a proposed technology for a component. The plugins for one component should have unique names. Each plugin should inherit from the class `Component` defined in `vcmrs/Utils/component.py` and implement the `process` function. 

The `process` function takes the following input arguments. 

  - `input_fname`: if a file is given, the file may be a video file in YUV format with .yuv as the extension or an image file with .png or .jpg as the extension. If a directory is given, the frames in the directory shall be processed. The input frame file names have the format of `<video_id>_%06d.png`. For example,  `video_id_000000.png.` 

  - `output_fname`: the output file name or directory. For image compression, it is the output image file. For video compression, it is the output directory name. And the output frame file names shall have a format of `frame_%06d.png`. Index shall start from 0.  

 - `item`: a FileItem object containing the parameters related to the file being processed. This argument is an object of `io_utils.FileItem`. It's used to store the data related to the file being processed. For example, the input arguments to process the file item.

  - `ctx`: a `CodecContext` object, defined in `codec_context.py`. The object is used to store some global data for the codec.  For example, the system input arguments, such as debug mode. 

A pre- or post-inner codec component shall be able to handle video file in both YUV format and frames in png format. Note that the output format may be different from the input format. For example, a component may take video in YUV format and output frames in png format. This design reduces unnecessary color format conversion among multiple components. 

If a pre-inner codec component modifiles the properties of the input image or video, for example, changing the resolution or intra period, it shall modify the corresonding properties in the item.args. For example, `item.args.SourceWidth` and `items.args.SourceHeight` for the resolution. 

If a post-inner codec component modifes the properties of the input image or video, for example, changing the resolution or intra period, it shall modify the corresponding properties in item.video_info. For example, `item.video_info.resolution` or `item.video_info.frame_rate`.

Two example plugins are provided. One is the bypass plugin, which does not modify the input data and simply creates a symbolic link to the output folder. The implementation of the bypass plugin is at `vcmrs/Utils/component_bypass.py`. The other example is a simple spatial downsampling plugin, which downsamples an input image or video by a scale factor defined in the input argument `OversizedVideoScaleFactor` if the longest side of the input media exceeds the value specified in input argument `ResolutionThreshold`.


## Data in bitstream

A plugin can store data in the bitstream at the encoder side and retrieve the data from the bitstream at the decoder side. The store data in a bitstream, the plugin shall use the `item.add_parameter()` function. For example,

```
param_data = bytearray([1,2,3])
item.add_parameter('SpatialResample', param_data=param_data) 
 ```
 
Note that the first parameter of `item.add_parameter` must be one of the component name, i.e., SpatialResample, TemporalResample, ROI, and PostFilter. The second parameter `param_data`  must be a type of `bytesarray`. 

**Note that this function for each component can only be called once. A second call will cause an exception. **

To read the data from the bitstream, the following function call can be used. 

```
param_data = item.get_parameter('SpatialResample')
```
The returned data has the format of `bytearray`.

## Inner codec control signal

The following code shows how to send control signals from a pre-inner codec component to the inner codec.
 
 ```
item.add_inner_control_signal('SpatialResample', signal={'data1': 0, 'data2': 2})
```

At the current release, no control signal has been defined, so the inner codec does not know how to use the control signal. Thus if a plugin needs to modify the behavior of the inner codec, it must modify the source code of the encoder to take the signal into use. 

# Neural network-based intra frame codec

A Neural network-based intra frame codec (NNIC) shall implement the interface `class IntraCodecController` The implementation shall be in the file `vcmrs/InnerCodec/NNManager/NNIntraCodec/<NNIC_name>/intra_codec_controller.py` where `NNIC_name` is the unique name for the NNIC  component. 

The `class InraCodecController` shall implement two interface functions. 
```
@API_func
def code_image(self, input_fp, output_bitstream_fp, output_image_fp, intra_cfg)

@API_func
def decode_bitstream(self, input_fp, output_image_fp, intra_cfg)
```
The input arguments and output data for these two functions are shown in the example in `vcmrs/InnerCodec/NNManager/NNIntraCodec/ExampleIntraCodec/intra_codec_controller.py`



