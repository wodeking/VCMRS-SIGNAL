# RoI Based Coding Using Descriptors Manual

This manual is intended to facilitate the understanding of RoI based coding using Descriptors.

### Using Pre-Generated RoI Descriptors

To execute an RoI based process using a pre-generated RoI descriptor, the value of `--RoIDescriptor` should be set to the descriptor file. When descriptor usage mode is activated, the object detection part from the video is omitted, and the RoI based process is carried out by reading the descriptor. This can be used for cross-checking using descriptors provided by the proponent.

### Default Behavior

If `--RoIDescriptor` is None, the RoI will be extracted from the video as usual, and the RoI based process will be executed.

## Download RoI descriptors for Anchor generation
The RoI descriptor for anchor generation can be downloaded from 
http://mpegx.int-evry.fr/software/MPEG/Video/VCM/vcm-ctc/-/tree/main/roi%20descriptors
