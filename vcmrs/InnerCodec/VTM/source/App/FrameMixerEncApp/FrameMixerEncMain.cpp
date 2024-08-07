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

 /** \file     FrameMixerEncMain.cpp
     \brief    Frame mixer enc main function and command line handling
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
#include "FrameMixerEncApp.h"


namespace po = df::program_options_lite;

 //! \ingroup FrameMixerEncApp
 //! \{


/**
  - Parse command line parameters
*/
bool parseCmdLine(int argc, char* argv[], std::vector<SubpicParams> &subpics, std::ofstream &outputStream, std::string &inputIntraBaseFileName, std::string &inputConfigBaseFileName, bool &removePs, std::string &interMachineAdapterConfigBaseFileName
#if ZJU_BIT_STRUCT
, std::string &inputRestorationDataFileName
#endif
)
{
  bool doHelp = false;
  std::string inputFileName;
  std::string outputFileName;

  po::Options opts;
  opts.addOptions()
    ("-help",                            doHelp,                                           false, "This help text")
    ("b",                                inputFileName,                          std::string(""), "Input bitstream to modify")
    ("o",                                outputFileName,                         std::string(""), "Output bitstream file")
    ("i",                                inputIntraBaseFileName,                 std::string(""), "Input intra base file name")
    ("c",                                inputConfigBaseFileName,                std::string(""), "Input intra config base file name")
    ("p",                                removePs,                                         false, "Remove parameter sets from other than first picture")
    ("s",                                interMachineAdapterConfigBaseFileName,  std::string(""), "Input inter machine adapter config base file name")
#if ZJU_BIT_STRUCT
    ("r",                                inputRestorationDataFileName,           std::string(""), "Input restoration data file name")
#endif
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
    std::cout << " Mix VVC bitstream with NN coded intra frames:" << std::endl;
    std::cout << "   FrameMixerEncApp -b input_file_name.bin -o out_filename.bin [-i input_intra_base_filename_ -c input_config_base_filename_] [-s inter_machine_adapter_config_base_filename_]" << std::endl;
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

  outputStream.open(outputFileName, std::ios_base::binary);
  if (!outputStream.is_open())
  {
    std::cerr << "Error: cannot open output file " << outputFileName << " for writing" << std::endl;
    return false;
  }

#if 0
  if (inputIntraBaseFileName.empty())
  {
    std::cerr << "Error: no input intra base file name given" << std::endl;
    return false;
  }
  if (inputConfigBaseFileName.empty())
  {
    std::cerr << "Error: no input intra config base file name given" << std::endl;
    return false;
  }
#else
  if (inputIntraBaseFileName.empty())
  {
    std::cout << "Warning: no input intra base file name given" << std::endl;
  }
  if (inputConfigBaseFileName.empty())
  {
    std::cout << "Warning: no input intra config base file name given" << std::endl;
  }
#endif

  return true;
}


/**
  - Subpicture merge main()
 */
int main(int argc, char* argv[])
{
  std::vector<SubpicParams> subpics;
  std::ofstream outputStream;
  std::string inputIntraBaseFileName;
  std::string inputConfigBaseFileName;
  bool removePs = false;
  std::string interMachineAdapterConfigBaseFileName;
#if ZJU_BIT_STRUCT
  std::string inputRestorationDataFileName;
#endif

  if (!parseCmdLine(argc, argv, subpics, outputStream, inputIntraBaseFileName, inputConfigBaseFileName, removePs, interMachineAdapterConfigBaseFileName
#if ZJU_BIT_STRUCT
  , inputRestorationDataFileName
#endif
  ))
  {
    return EXIT_FAILURE;
  }

  FrameMixerEncApp *frameMixerEncApp = new FrameMixerEncApp(subpics, outputStream, inputIntraBaseFileName, inputConfigBaseFileName, removePs, interMachineAdapterConfigBaseFileName
#if ZJU_BIT_STRUCT
  , inputRestorationDataFileName
#endif
  );

  int returnCode = EXIT_SUCCESS;
  try
  {
#if ZJU_BIT_STRUCT
    if(!inputRestorationDataFileName.empty() && inputIntraBaseFileName.empty())
    {
      frameMixerEncApp->genVCMBitstream();
    }
    else
    {
#endif
    frameMixerEncApp->mixFrames();
#if ZJU_BIT_STRUCT
    }
#endif
  }
  catch (const std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << std::endl;
    returnCode = EXIT_FAILURE;
  }

  delete frameMixerEncApp;

  return returnCode;
}

//! \}
