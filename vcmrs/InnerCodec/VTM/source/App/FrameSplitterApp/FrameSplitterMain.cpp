/* The copyright in this software is being made available under the BSD
* License, included below. This software may be subject to other third party
* and contributor rights, including patent rights, and no such rights are
* granted under this license.
*
* Copyright (c) 2010-2021, ITU/ISO/IEC
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*  * Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
*    be used to endorse or promote products derived from this software without
*    specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGE.
*/

 /** \file     FrameSplitterMain.cpp
     \brief    Frame splitter main function and command line handling
 */

#include <cstdio>
#include <cctype>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ios>
#include <algorithm>
#include "Utilities/program_options_lite.h"
#include "FrameSplitterApp.h"


namespace po = df::program_options_lite;

 //! \ingroup FrameSplitterApp
 //! \{


/**
  - Parse command line parameters
*/
bool parseCmdLine(int argc, char* argv[], std::vector<SubpicParams> &subpics, std::string &outBaseFileName)
{
  bool doHelp = false;
  std::string inputFileName;

  po::Options opts;
  opts.addOptions()
    ("-help",                            doHelp,                                           false, "This help text")
    ("b",                                inputFileName,                          std::string(""), "File containing list of input pictures to be merged")
    ("o",                                outBaseFileName,                        std::string(""), "Output base file name")
    ;

  po::setDefaults(opts);
  po::ErrorReporter err;
  const std::list<const char*>& argvUnhandled = po::scanArgv(opts, argc, (const char**) argv, err);

  if (argc == 1 || doHelp)
  {
    /* argc == 1: no options have been specified */
    po::doHelp(std::cout, opts);
    std::cout << std::endl;
    std::cout << "Examples" << std::endl;
    std::cout << " Split VVC bitstream:" << std::endl;
    std::cout << "   FrameSplitterApp -b input_file_name.bin -o out_base_filename_" << std::endl;
    std::cout << std::endl;
    return false;
  }

  for (std::list<const char*>::const_iterator it = argvUnhandled.begin(); it != argvUnhandled.end(); it++)
  {
    std::cerr << "Unhandled argument ignored: `" << *it << "'" << std::endl;
  }

  subpics.emplace_back();
  SubpicParams &sf = subpics.back();
  sf.fp.open(inputFileName, std::ios_base::binary);
  if (!sf.fp.is_open())
  {
    std::cerr << "Error: cannot open input file " << inputFileName << " for reading" << std::endl;
    return false;
  }

  if (outBaseFileName.length() == 0)
  {
    std::cerr << "Error: no output base file name given" << std::endl;
    return false;
  }

  return true;
}


/**
  - Subpicture merge main()
 */
int main(int argc, char* argv[])
{
  std::vector<SubpicParams> subpics;
  std::string outBaseFileName;

  if (!parseCmdLine(argc, argv, subpics, outBaseFileName))
  {
    return 1;
  }

  FrameSplitterApp *frameSplitterApp = new FrameSplitterApp(subpics, outBaseFileName);

  frameSplitterApp->splitFrames();

  delete frameSplitterApp;

  return 0;
}

//! \}
