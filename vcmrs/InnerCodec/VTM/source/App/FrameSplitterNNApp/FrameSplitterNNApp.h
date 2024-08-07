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

 /** \file     FrameSplitterNN.h
     \brief    Frame splitter for NN coded intra frames
 */

#include <vector>
#include <fstream>

#define ZJU_BIT_STRUCT                      1

 //! \ingroup FrameSplitterNNApp
 //! \{



struct SubpicParams {
  int                                  width;
  int                                  height;
  int                                  topLeftCornerX;
  int                                  topLeftCornerY;
  std::ifstream                        fp;
};


struct Subpicture;
class InputByteStream;
class HLSyntaxReader;
class DCI;
class ParameterSetManager;
class PicHeader;
class InputNALUnit;
class Slice;
class OutputBitstream;
class VPS;
class SPS;
class PPS;
struct OutputNALUnit;
class AccessUnit;


class FrameSplitterNNApp
{
public:
  FrameSplitterNNApp(std::vector<SubpicParams> &subpicParams, std::string &outBaseFileName, std::string &outConfigBaseFileName, std::string &outInterMachineAdapterConfigBaseFileName
#if ZJU_BIT_STRUCT
  , std::string &outRestorationDataFileName
  , std::string & outCodedVideoDataFileName
#endif
);
  ~FrameSplitterNNApp();

  void splitFrames();
#if ZJU_BIT_STRUCT
  void splitVCMBitstream();
  void writeOneCVDPic(OutputBitstream &cvdBs);
#endif

private:
  std::vector<Subpicture> *m_subpics;
  std::ofstream m_outputStream;
  std::ofstream m_outputConfigStream;
  std::string m_outBaseFileName;
  std::string m_outConfigBaseFileName;
  std::string m_outInterMachineAdapterConfigBaseFileName;
  int m_prevPicPOC;
  int m_picWidth;
  int m_picHeight;
#if ZJU_BIT_STRUCT
  std::string m_outRestorationDataFileName;
  std::string m_outCodedVideoDataFileName;
  std::ofstream m_outRsdStream;
  std::ofstream m_outCvdStream;
#endif

  bool isNewPicture(std::ifstream *bitstreamFile, InputByteStream *bytestream, bool firstSliceInPicture);
  bool parseDCI(HLSyntaxReader &hlsReader, DCI &dci);
  int parseVPS(HLSyntaxReader &hlsReader, ParameterSetManager &psManager);
  int parseSPS(HLSyntaxReader &hlsReader, ParameterSetManager &psManager);
  int parsePPS(HLSyntaxReader &hlsReader, ParameterSetManager &psManager);
  void parseAPS(HLSyntaxReader &hlsReader, ParameterSetManager &psManager, int &apsId, int &apsType);
  void parsePictureHeader(HLSyntaxReader &hlsReader, PicHeader &picHeader, ParameterSetManager &psManager);
  void parseSliceHeader(HLSyntaxReader &hlsReader, InputNALUnit &nalu, Slice &slice, PicHeader &picHeader, OutputBitstream &sliceData, ParameterSetManager &psManager, int prevTid0Poc);
  void parseSliceHeaderNN(HLSyntaxReader &hlsReader, InputNALUnit &nalu, Slice &slice, PicHeader &picHeader, OutputBitstream &sliceData, ParameterSetManager &psManager, int prevTid0Poc);
  void parseSEIMessage(Subpicture &subpic, InputNALUnit &nalu);
  void decodeNalu(Subpicture &subpic, InputNALUnit &nalu);
  void parseSubpic(Subpicture &subpic, bool &morePictures);
  void generateMergedStreamVPSes(std::vector<VPS*> &vpsList);
  void generateMergedStreamSPSes(std::vector<SPS*> &spsList);
  void generateMergedStreamPPSes(ParameterSetManager &psManager, std::vector<PPS*> &ppsList);
  void updateSliceHeadersForMergedStream(ParameterSetManager &psManager);
  void copyInputNaluToOutputNalu(OutputNALUnit &outNalu, InputNALUnit &inNalu);
  void copyNalUnitsToAccessUnit(AccessUnit &accessUnit, std::vector<InputNALUnit> &nalus, int naluType);
  void writeOnePic();
  void generateFilename(std::string &generatedFilename, std::string &baseFilename, int n, const char* suffix);
};


//! \}
