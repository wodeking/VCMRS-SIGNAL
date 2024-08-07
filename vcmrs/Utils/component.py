# This file is covered by the license agreement found in the file "license.txt" in the root of this project.


# component base class
class Component:
  def __init__(self, ctx):
    '''Initialization

    Args:
      ctx: context object 
    '''
    self.ctx = ctx

  def process(self, input_fname, output_fname, item, ctx):
    '''
    Process input data. If the file is not modified, symbolic link may be used. 

    Args: 
      input_fname: if a file is given, a single frame shall be processed. 
        if a directory is given, the frames in the directory shall be processed. The input frame
        file names have the format of frame_%06d.png. 

      output_fname: the output file name or directory. For image compression, it is the output
        image file. For video compression, it is the output directory name. And the output 
        frame file names shall have a format of `frame_%06d.png`. Index shall start from 0.  

      item: a FileItem object containing the parameters related to the file being processed. 
      This argument is an object of `io_utils.FileItem`. It's used to store the data related 
      to the file being processed. For example, the input arguments to process the file item.

      ctx: a `CodecContext` object, defined in `codec_context.py`. The object is used to store
      some global data for the codec.  For example, the system input arguments, such as debug
      mode. 

         
    '''
    raise NotImplementedError

