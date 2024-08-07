# Spatial resampling Using Descriptors Manual

This manual is intended to facilitate the understanding of spatial resampling using Descriptors.

### Using Pre-Generated Spatial Descriptors

To execute a Spatial resampling process generating and using a spatial descriptor, the value of `--SpatialDescriptorMode` should be set to `GeneratingDescriptor` and `UsingDescriptor`, respectively. The spatial resampling is conducted without generating and using the spatial descriptor if the value of `--SpatialDescriptorMode` is set to `NoDescriptor`. You can set the name of the spatial descriptor with a path configured by `--SpatialDescriptor`. The spatial descriptor can be used for cross-checking.

`--SpatialDescriptorMode`
  - `GeneratingDescriptor`: save the spatial descriptor and exit.
  - `UsingDescriptor`: run the spatial resampling using the spatial descriptor.
  - `NoDescriptor (default)`: run the spatial resampling without generating and using the spatial descriptor.

`--SpatialDescriptor` : File name of spatial descriptor file to be saved and loaded.
  - `default` : `<VCM-RS root>/Data/spatial_descriptors`

### Default Behavior
The default of `--SpatialDescriptorMode` is `NoDescriptor`. Therefore, VCM-RS is conducted to run the spatial resampling without generating and using the spatial descriptor.
If the `--SpatialDescriptor` is not set, the spatial descriptor is located in `<VCM-RS root>/Data/spatial_descriptors` for generating and using.


## Download spatial descriptors for Anchor generation
The spatial descriptor for anchor generation can be downloaded from 
https://git.mpeg.expert/MPEG/Video/VCM/vcm-ctc/-/tree/main/spatial%20descriptors
