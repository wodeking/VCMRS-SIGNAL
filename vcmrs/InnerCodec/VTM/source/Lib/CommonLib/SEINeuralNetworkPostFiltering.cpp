/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2023, ITU/ISO/IEC
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

/** \file    SEINeuralNetworkPostFiltering.cpp
   \brief    SEI NN post filtering (application) class
*/

#include "SEINeuralNetworkPostFiltering.h"

#if JVET_AC0074_USE_OF_NNPFC_FOR_PIC_RATE_UPSAMPLING

SEINeuralNetworkPostFiltering::SEINeuralNetworkPostFiltering()
: m_picList()
{

}

void SEINeuralNetworkPostFiltering::setPictureNnpfc(Picture* picture)
{
  SEIMessages seiList = getSeisByType(picture->SEIs, SEI::PayloadType::NEURAL_NETWORK_POST_FILTER_ACTIVATION);
  std::map<uint32_t, bool> tmpIsNnpfActivatedForPic;
  for (auto sei : seiList)
  {
    auto nnpfa = (SEINeuralNetworkPostFilterActivation*) sei;
    auto nnpfc = getNnpfcWithGivenId(m_clvsNnpfcSEIs, nnpfa->m_targetId);
    CHECK(nullptr == nnpfc, "There must be a NNPFC with nnpfc_id equal to nnpfa_target_id");
    if (nnpfa->m_cancelFlag)
    {
      tmpIsNnpfActivatedForPic[nnpfa->m_targetId] = false;
      m_isNnpfActiveForCLVS[nnpfa->m_targetId]    = false;
    }
    else
    {
      if (nnpfa->m_persistenceFlag)
      {
        m_isNnpfActiveForCLVS[nnpfa->m_targetId] = true;
      }
      else
      {
        tmpIsNnpfActivatedForPic[nnpfa->m_targetId] = true;
      }
    }
  }

  std::map<uint32_t, bool> isNnpfActivactedForPic = m_isNnpfActiveForCLVS;
  for (auto it: tmpIsNnpfActivatedForPic)
  {
    isNnpfActivactedForPic.insert_or_assign(it.first, it.second);
  }
  
  for (auto it : isNnpfActivactedForPic)
  {
    if (it.second)
    {
      auto nnpfc = getNnpfcWithGivenId(m_clvsNnpfcSEIs, it.first);
#if AVOID_DUPLICATE_NNPFC_ENTRY
      if (std::find(picture->SEIs.begin(), picture->SEIs.end(), nnpfc) == picture->SEIs.end())
      {
        picture->SEIs.push_back(nnpfc);
      }
#else
      picture->SEIs.push_back(nnpfc);
#endif
    }
  }
}

void SEINeuralNetworkPostFiltering::filterPictures(PicList& picList)
{
  m_picList = PicVector(picList.begin(), picList.end());
  for (Picture* currCodedPic: m_picList)
  {
    const NalUnitType picType = currCodedPic->getPictureType();

    if (picType == NAL_UNIT_CODED_SLICE_IDR_N_LP || picType == NAL_UNIT_CODED_SLICE_IDR_W_RADL
        || picType == NAL_UNIT_CODED_SLICE_CRA || picType == NAL_UNIT_CODED_SLICE_GDR)
    {
      m_clvsNnpfcSEIs = getSeisByType(currCodedPic->SEIs, SEI::PayloadType::NEURAL_NETWORK_POST_FILTER_CHARACTERISTICS);
      m_isNnpfActiveForCLVS.clear();
    }

    setPictureNnpfc(currCodedPic);

    const SPS *sps = currCodedPic->slices[0]->getSPS();
    const PPS* pps = currCodedPic->slices[0]->getPPS();
    const int subWidthC  = SPS::getWinUnitX(sps->getChromaFormatIdc());
    const int subHeightC = SPS::getWinUnitY(sps->getChromaFormatIdc());

    for (auto sei : currCodedPic->m_nnpfcActivated)
    {
      auto currNnpfc = (SEINeuralNetworkPostFilterCharacteristics*) sei;
#if JVET_AC0127_BIT_MASKING_NNPFC_PURPOSE
      bool pictureRateUpsamplingFlag = (currNnpfc->m_purpose & NNPC_PurposeType::FRAME_RATE_UPSAMPLING) != 0;
#else
      bool pictureRateUpsamplingFlag = currNnpfc->m_purpose == 5;
#endif
      if (pictureRateUpsamplingFlag)
      {
        uint32_t sourceWidth = pps->getPicWidthInLumaSamples()
                               - subWidthC
                                   * (pps->getConformanceWindow().getWindowLeftOffset()
                                      + pps->getConformanceWindow().getWindowRightOffset());
        uint32_t sourceHeight = pps->getPicHeightInLumaSamples()
                                - subHeightC
                                    * (pps->getConformanceWindow().getWindowTopOffset()
                                       + pps->getConformanceWindow().getWindowBottomOffset());

        uint32_t croppedWidth;
        uint32_t croppedHeight;
        auto *superResolutionNnpfc = getSuperResolutionNnpfc(currCodedPic->m_nnpfcActivated);
        if (nullptr != superResolutionNnpfc)
        {
          croppedWidth  = superResolutionNnpfc->m_picWidthInLumaSamples;
          croppedHeight = superResolutionNnpfc->m_picHeightInLumaSamples;
        }
        else
        {
          croppedWidth = pps->getPicWidthInLumaSamples()
                         - subWidthC
                             * (pps->getConformanceWindow().getWindowLeftOffset()
                                + pps->getConformanceWindow().getWindowRightOffset());
          croppedHeight = pps->getPicHeightInLumaSamples()
                          - subHeightC
                              * (pps->getConformanceWindow().getWindowTopOffset()
                                 + pps->getConformanceWindow().getWindowBottomOffset());
        }

        findFrameRateUpSamplingInputPics(currCodedPic, currNnpfc,
                                         sourceWidth, sourceHeight, croppedWidth, croppedHeight);
      }
    }
  }
}

void SEINeuralNetworkPostFiltering::findFrameRateUpSamplingInputPics(
  Picture* currCodedPic, const SEINeuralNetworkPostFilterCharacteristics* frameRateUpsamplingNnpfc,
  uint32_t sourceWidth, uint32_t sourceHeight, uint32_t croppedWidth, uint32_t croppedHeight)
{
#if JVET_AC0127_BIT_MASKING_NNPFC_PURPOSE
  uint32_t numInputPics = frameRateUpsamplingNnpfc->m_numberInputDecodedPicturesMinus1 + 1;
#else
  int numInputPics = frameRateUpsamplingNnpfc->m_numberInputDecodedPicturesMinus2 + 2;
#endif

  std::vector<uint32_t> inputPicPoc(numInputPics, 0);
  uint32_t currPicPOC = currCodedPic->getPOC();
  inputPicPoc[0] = currPicPOC;

  if (numInputPics > 1)
  {
    SEIMessages currPicFramePacking = getSeisByType(currCodedPic->SEIs, SEI::PayloadType::FRAME_PACKING);
    bool fpCurrPicArrangementTypeIsFive = false;
    bool fpCurrPicFrameIsFrame0Flag     = false;
    if (!currPicFramePacking.empty())
    {
      const SEIFramePacking* seiFramePacking = (SEIFramePacking*) *(currPicFramePacking.begin());
      fpCurrPicArrangementTypeIsFive = seiFramePacking->m_arrangementType == 5;
      fpCurrPicFrameIsFrame0Flag = seiFramePacking->m_currentFrameIsFrame0Flag;
    }

    for (int i = 1; i < numInputPics; i++)
    {
      inputPicPoc[i]    = inputPicPoc[i - 1];
      Picture* inputPic = m_picList[i - 1];
      if (fpCurrPicArrangementTypeIsFive)
      {
        SEIMessages inputPicFramePacking = getSeisByType(inputPic->SEIs, SEI::PayloadType::FRAME_PACKING);
        if (!inputPicFramePacking.empty())
        {
          const SEIFramePacking* seiFramePacking = (SEIFramePacking*) *(inputPicFramePacking.begin());
          CHECK((seiFramePacking->m_arrangementType != 5
                 || seiFramePacking->m_currentFrameIsFrame0Flag != fpCurrPicFrameIsFrame0Flag),
                "If currCodedPic is associated with a frame packing arrangement SEI message with fp_arrangement_type equal to 5, "
                "inputPicPoc[i] must have the same value for fp_arrangement_type and fp_current_frame_is_frame0_flag");
        }
      }

      if (sourceWidth != croppedWidth || sourceHeight != croppedHeight)
      {
        bool inputPicUsesNnpfResolutionAdaptation = false;
        for (auto sei : inputPic->m_nnpfcActivated)
        {
          const auto *inputNnpfc = (SEINeuralNetworkPostFilterCharacteristics*)sei;
#if JVET_AC0127_BIT_MASKING_NNPFC_PURPOSE
          if ((inputNnpfc->m_purpose & NNPC_PurposeType::RESOLUTION_UPSAMPLING) != 0)
#else
          if (nnpfc->m_purpose == 4)
#endif
          {
            CHECK(inputNnpfc->m_picWidthInLumaSamples != croppedWidth || inputNnpfc->m_picHeightInLumaSamples != croppedHeight, "Input picture shall have a super resolution NNPF activated");
            inputPicUsesNnpfResolutionAdaptation = true;
            break;
          }
        }
        CHECK(!inputPicUsesNnpfResolutionAdaptation, "Input picture does not uses a super resolution NNPF");
      }
    }
  }
}
#endif
