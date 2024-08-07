# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import setuptools
from distutils.core import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

setuptools.setup(
    name="e2evc", # End-to-end video compression
    version="0.0.1",
    author="Anonymous",
    author_email="Anonymous",
    description="NN Intra codec of the NN-VVC framework",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: :: ",
        "Operating System :: OS Independent",
    ],

    # install_requires = required,
    
    #Extension modules
    ext_modules = [
      # rans
      Extension(
        name='ans',
        sources=['e2evc/EntropyCodec/rans/rans_interface.cpp'],
        include_dirs=cpp_extension.include_paths(),
        language='c++'),

      # entropy model utils
      Extension(
        name='entropy_model_ext',
        sources=['e2evc/EntropyModel/cpp_exts/ops/ops.cpp'],
        include_dirs=cpp_extension.include_paths(),
        language='c++'),
 
    ],

    cmdclass = {
      'build_ext': BuildExtension
    },

    python_requires='>=3.7',
)
