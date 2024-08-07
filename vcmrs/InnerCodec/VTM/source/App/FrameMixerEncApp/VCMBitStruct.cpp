/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2024-2034, Zhejiang University
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
 *  * Neither the name of the Zhejiang University nor the names of its contributors may
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

#include "VideoParameterSet.h"
#include "VCMBitStruct.h"

/**
 * @brief VCM parameter set
 * 
 */

VCMPS::VCMPS()
  : m_refVpsId(0)
  , m_log2MaxRestorationDataPOCLSB(0)
  , m_spatialResampleEnableFlag(false)
  , m_retargetingEnableFlag(false)
  , m_temporalResampleEnableFlag(false)
  , m_bitDepthShiftEnableFlag(false)
{}

void VCMUnit::writeHeader() 
{
  m_bitstream.write(m_vcmUnitType, 5);
  if (m_vcmUnitType > 0)
  {
    m_bitstream.write(m_refVpsId, 4);
    m_bitstream.write(0, 23);   
  }
  else
  {
    m_bitstream.write(0, 27);
  }
}

void VCMUnit::writeVCMPS()
{
  m_bitstream.clear();
  writeHeader();

  m_bitstream.write(0, 1);
  m_bitstream.write(0, 7);
  m_bitstream.write(0, 8);
  m_bitstream.write(0, 8);

  m_bitstream.write(m_vpsId, 4);
  m_bitstream.write(m_vpsBitsForPOCLsb-4, 4);
  m_bitstream.write(m_vpsSpatialFlag, 1);
  m_bitstream.write(m_vpsRetargetFlag, 1);
  m_bitstream.write(m_vpsTemporalFlag, 1);
  m_bitstream.write(m_vpsBitDepthShiftFlag, 1);
	m_bitstream.writeByteAlignment();
}

void VCMUnit::writeRSD(OutputBitstream& inRsd) 
{
  m_bitstream.clear();
  writeHeader();
  m_bitstream.addSubstream(&inRsd);
  //m_bitstream.writeByteAlignment();
}

void VCMUnit::writeCVD(OutputBitstream& inCvd) 
{
  m_bitstream.clear();
  writeHeader();
  m_bitstream.addSubstream(&inCvd);
  //m_bitstream.writeByteAlignment();
}

void VCMUnit::parseHeader() 
{
  uint32_t valueBits;
  uint32_t zeroBits;
  m_inBitstream.read(5, valueBits);
  m_vcmUnitType = VCMUnitType(valueBits);
  CHECK(m_vcmUnitType > 2, "Wrong VCM unit type" );

  if (m_vcmUnitType > 0)
  {
    m_inBitstream.read(4, m_refVpsId);
    m_inBitstream.read(23, zeroBits);
    CHECK(zeroBits != 0, "Found non-zero VCM unit header 1");
  }
  else
  {
    m_inBitstream.read(27, zeroBits);
    CHECK(zeroBits != 0, "Found non-zero VCM unit header 2");
  }
}

void VCMUnit::parseVCMPS() 
{
  uint32_t valueBits;

  m_inBitstream.read(1, valueBits);
  CHECK(valueBits != 0, "Found non-zero VCMPS 1");
  m_inBitstream.read(7, valueBits);
  CHECK(valueBits != 0, "Found non-zero VCMPS 2");
  m_inBitstream.read(8, valueBits);
  CHECK(valueBits != 0, "Found non-zero VCMPS 3");
  m_inBitstream.read(8, valueBits);
  CHECK(valueBits != 0, "Found non-zero VCMPS 4");

  m_inBitstream.read(4, m_vpsId);
  m_inBitstream.read(4, valueBits);
  m_vpsBitsForPOCLsb = valueBits + 4;
  m_inBitstream.read(1, m_vpsSpatialFlag);
  m_inBitstream.read(1, m_vpsRetargetFlag);
  m_inBitstream.read(1, m_vpsTemporalFlag);
  m_inBitstream.read(1, m_vpsBitDepthShiftFlag);
  m_inBitstream.readByteAlignment();
}

void VCMUnit::parseRSD(InputBitstream* inRsd) 
{
  InputBitstream *tmpBs;
  tmpBs = m_inBitstream.extractSubstream(m_inBitstream.getNumBitsLeft());
  inRsd->getFifo().insert(inRsd->getFifo().end(), tmpBs->getFifo().begin(), tmpBs->getFifo().end()); // add the bits to the end.
}

void VCMUnit::parseCVD(OutputBitstream& inCvd) 
{
  InputBitstream *tmpBs;
  OutputBitstream tmpOutBs;
  tmpBs = m_inBitstream.extractSubstream(m_inBitstream.getNumBitsLeft());
  tmpOutBs.getFifo() = tmpBs->getFifo();
  inCvd.addSubstream(&tmpOutBs); // add the bits to the end.
}

void VCMNalu::writeVCMNaluHeader()
{
  m_forbiddenZeroBit = 0;
  m_nuhReservedZeroBit = 0;

  m_bitstream.write(m_forbiddenZeroBit, 1);
  m_bitstream.write(m_nalUnitType, 6);
  m_bitstream.write(m_temporalId, 3);
  m_bitstream.write(m_nuhReservedZeroBit, 6);
}

void VCMNalu::writeSRD(std::vector<uint8_t>& rsd, uint32_t rsdSize)
{
  m_bitstream.clear();
  writeVCMNaluHeader();
  m_bitstream.write(m_srdId, 4);
  
  /* TODO: whether to write rsd information should be decided by the config. But currently the rsd information is not provided in readable form*/
  m_bitstream.write(rsdSize, 20);
  OutputBitstream tmpBs;
  tmpBs.getFifo() = rsd;
  m_bitstream.addSubstream(&tmpBs);
	m_bitstream.writeByteAlignment();
}

void VCMNalu::writePRD() 
{
  m_bitstream.clear();
  writeVCMNaluHeader();

  m_bitstream.write(m_prdRefSrdId, 4);
  m_bitstream.write(m_prdPocLsb, m_refSRD->m_refVPS->m_vpsBitsForPOCLsb);
	if (m_refSRD->m_refVPS->m_vpsSpatialFlag)
  {
    m_bitstream.write(m_prdSpatialFlag, 1);
  }
  if (m_refSRD->m_refVPS->m_vpsRetargetFlag)
  {
    m_bitstream.write(m_prdRetargetFlag, 1);
  }
  if (m_refSRD->m_refVPS->m_vpsTemporalFlag)
  {
    m_bitstream.write(m_prdTemporalFlag, 1);
  }
  if (m_refSRD->m_refVPS->m_vpsBitDepthShiftFlag)
  {
    m_bitstream.write(m_prdBitDepthShiftFlag, 1);
  }
	m_bitstream.writeByteAlignment();
}

void VCMNalu::writeSEI() 
{
  m_bitstream.clear();
  writeVCMNaluHeader();
  /* TODO */
}

void VCMNalu::writeEOSS() 
{
  m_bitstream.clear();
  writeVCMNaluHeader();
}

void VCMNalu::parseVCMNaluHeader() 
{
  uint32_t valueBits;

  m_inBitstream.read(1, m_forbiddenZeroBit);
  CHECK(m_forbiddenZeroBit != 0, "Found non-zero VCM NAL header 1");

  m_inBitstream.read(6, valueBits);
  m_nalUnitType = VCMNaluType(valueBits);
  CHECK(!(m_nalUnitType == VCM_NAL_SRD || m_nalUnitType == VCM_NAL_PRD || m_nalUnitType == VCM_NAL_SEI || m_nalUnitType == VCM_NAL_EOSS), "Found wrong VCM NAL type");
  
  m_inBitstream.read(3, m_temporalId);
  
  m_inBitstream.read(6, m_nuhReservedZeroBit);
  CHECK(m_nuhReservedZeroBit != 0, "Found non-zero VCM NAL header 2");
}

void VCMNalu::parseSRD(std::vector<uint8_t>& rsd) 
{
  m_inBitstream.read(4, m_srdId);

  uint32_t sizeRsd;
  InputBitstream *tmpBs;
  m_inBitstream.read(20, sizeRsd);
  tmpBs = m_inBitstream.extractSubstream(sizeRsd * 8);
  rsd.clear();
  rsd.insert(rsd.end(), tmpBs->getFifo().begin(), tmpBs->getFifo().end());

  m_inBitstream.readByteAlignment();
}

void VCMNalu::parsePRD() 
{
  m_inBitstream.read(4, m_prdRefSrdId);
  CHECK(m_prdRefSrdId != m_refSRD->m_srdId, "Found wrong referenced SRD ID in PRD"); // temporary solution.

  m_inBitstream.read(m_refSRD->m_refVPS->m_vpsBitsForPOCLsb, m_prdPocLsb);

	if (m_refSRD->m_refVPS->m_vpsSpatialFlag)
  {
    m_inBitstream.read(1, m_prdSpatialFlag);
    CHECK(m_prdSpatialFlag != 0, "Found wrong flag for spatial"); // temporary solution.
  }
  if (m_refSRD->m_refVPS->m_vpsRetargetFlag)
  {
    m_inBitstream.read(1, m_prdRetargetFlag);
    CHECK(m_prdRetargetFlag != 0, "Found wrong flag for retarget"); // temporary solution.
  }
  if (m_refSRD->m_refVPS->m_vpsTemporalFlag)
  {
    m_inBitstream.read(1, m_prdTemporalFlag);
    CHECK(m_prdTemporalFlag != 0, "Found wrong flag for temporal"); // temporary solution.
  }
  if (m_refSRD->m_refVPS->m_vpsBitDepthShiftFlag)
  {
    m_inBitstream.read(1, m_prdBitDepthShiftFlag);
    CHECK(m_prdBitDepthShiftFlag != 0, "Found wrong flag for bit depth shift"); // temporary solution.
  }
  m_inBitstream.readByteAlignment();
}

void VCMNalu::parseSEI() 
{
  // TODO
}
