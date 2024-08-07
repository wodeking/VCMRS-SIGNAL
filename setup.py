# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import setuptools
from distutils.core import setup, Extension
import subprocess
from pip._internal.req import parse_requirements


# will be enabled later

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vcmrs",
    version="0.8.1",
    author="MPEG VCM",
    author_email="",
    description="VCM reference software",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],

    install_reqs = parse_requirements('Requirements.txt', session='hack'),

    python_requires='>=3.8',
)
