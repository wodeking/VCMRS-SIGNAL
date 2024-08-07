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

#pragma once

#include <vector>
#include <stdint.h>
#include "VLCWriter.h"

//------------------------------------------------------------------
//    VCM unit
//------------------------------------------------------------------
enum VCMUnitType
{
  VCM_VPS = 0,       // 0
  VCM_RSD,           // 1
  VCM_CVD,           // 2
  VCM_RSV_3  = 3,    // 3
  VCM_RSV_31 = 31,   // 31
  VCM_UNIT_INVALID   // 32
};

struct VCMUnit
{
  VCMUnitType     m_vcmUnitType;   ///< vuh_unit_type
  uint32_t        m_refVpsId;         ///< vuh_vps_id
  OutputBitstream m_bitstream;
  InputBitstream m_inBitstream;

  /** VPS syntax */
  uint32_t m_vpsId;
  uint32_t m_vpsBitsForPOCLsb;
	uint32_t m_vpsSpatialFlag;
	uint32_t m_vpsRetargetFlag;
	uint32_t m_vpsTemporalFlag;
	uint32_t m_vpsBitDepthShiftFlag;

  VCMUnit(VCMUnit& src)
    : m_vcmUnitType(src.m_vcmUnitType)
    , m_refVpsId(src.m_refVpsId)
  {m_bitstream.addSubstream(&src.m_bitstream);}
  /** construct an VCMUnit structure with given header values. */
  VCMUnit(VCMUnitType nalUnitType, int vpsId = 0)
    : m_vcmUnitType(nalUnitType)
    , m_refVpsId(vpsId)
  {}

  /** default constructor - no initialization; must be performed by user */
  VCMUnit() {}

  virtual ~VCMUnit() {}

  /** returns true if the VCM unit is a VPS */
  bool isVPS() { return m_vcmUnitType == VCM_VPS; }
  /** returns true if the VCM unit is a RSD */
  bool isRSD() { return m_vcmUnitType == VCM_RSD; }
  /** returns true if the VCM unit is a CVD */
  bool isCVD() { return m_vcmUnitType == VCM_CVD; }
  /** write VCM unit data */
  void writeHeader();
  void writeVCMPS();
  void writeRSD(OutputBitstream &inRsd);
  void writeCVD(OutputBitstream &inCvd);
  /** parse VCM unit data */
  void parseHeader();
  void parseVCMPS();
  void parseRSD(InputBitstream *inRsd);
  void parseCVD(OutputBitstream &inCvd);
};

//------------------------------------------------------------------
//    VCM NAL unit
//------------------------------------------------------------------
enum VCMNaluType
{
  VCM_NAL_SRD = 0,       // 0
  VCM_NAL_PRD,           // 1
  VCM_NAL_RSV_2,         // 2
  VCM_NAL_RSV_31 = 31,   // 31
  VCM_NAL_EOSS,           // 32
  VCM_NAL_SEI,           // 33

  VCM_NAL_RSV_34,   // 34
  VCM_NAL_RSV_59,   // 59
  VCM_NAL_UNSPEC_60,
  VCM_NAL_UNSPEC_63 = 63,   // 63
  VCM_NAL_UNIT_INVALID          // 64
};

struct VCMNalu
{
  VCMNaluType     m_nalUnitType;   ///< nal_unit_type
  uint32_t        m_temporalId;    ///< temporal_id
  uint32_t        m_forbiddenZeroBit;
  uint32_t        m_nuhReservedZeroBit;
  OutputBitstream m_bitstream;
  InputBitstream m_inBitstream;

  /* SRD syntax*/
  uint32_t  m_srdId;
  VCMUnit *m_refVPS;

  /* PRD syntax*/
  uint32_t  m_prdRefSrdId;
  uint32_t  m_prdPocLsb;
  uint32_t  m_prdSpatialFlag;
  uint32_t  m_prdRetargetFlag;
  uint32_t  m_prdTemporalFlag;
  uint32_t  m_prdBitDepthShiftFlag;
  VCMNalu *m_refSRD;

  /* SEI syntax*/
  /* TODO */

  VCMNalu(VCMNalu& src)
    : m_nalUnitType(src.m_nalUnitType)
    , m_temporalId(src.m_temporalId)
    , m_forbiddenZeroBit(src.m_forbiddenZeroBit)
    , m_nuhReservedZeroBit(src.m_nuhReservedZeroBit)
  {m_bitstream.addSubstream(&src.m_bitstream);}
  /** construct an NALunit structure with given header values. */
  VCMNalu(VCMNaluType nalUnitType, int temporalId = 0, uint32_t nuhReservedZeroBit = 0, uint32_t forbiddenZeroBit = 0)
    : m_nalUnitType(nalUnitType)
    , m_temporalId(temporalId)
    , m_forbiddenZeroBit(forbiddenZeroBit)
    , m_nuhReservedZeroBit(nuhReservedZeroBit)
  {}

  /** default constructor - no initialization; must be performed by user */
  VCMNalu() {}

  virtual ~VCMNalu() {}

  /** returns true if the NALunit is a RDL NALunit */
  bool isRdl() { return m_nalUnitType == VCM_NAL_SRD || m_nalUnitType == VCM_NAL_PRD; }
  /** returns true if the NALunit is a SEI NALunit */
  bool isSei() { return m_nalUnitType == VCM_NAL_SEI; }
  /** write VCM NAL unit data */
  void writeVCMNaluHeader();
  void writeSRD(std::vector<uint8_t> &rsd, uint32_t rsdSize);
  void writePRD();
  void writeSEI();
  void writeEOSS();
  /** parse VCM NAL unit data */
  void parseVCMNaluHeader();
  void parseSRD(std::vector<uint8_t> &rsd);
  void parsePRD();
  void parseSEI();
};

//------------------------------------------------------------------
//    VCM profile
//------------------------------------------------------------------
class VCMProfile
{
private:
  bool m_ptlTierFlag;
  int  m_ptlProfileCodecGroupIdc;
  int  m_ptlProfileRestorationIdc;
  int  m_ptlLevelIdc;

public:
  VCMProfile();
  virtual ~VCMProfile(){};

  int  getVPSId() const { return m_ptlProfileCodecGroupIdc; }
  void setVPSId(int i) { m_ptlProfileCodecGroupIdc = i; }
};

//------------------------------------------------------------------
//    VCM parameter set
//------------------------------------------------------------------
class VCMPS
{
private:
  int  m_refVpsId;
  int  m_log2MaxRestorationDataPOCLSB;
  bool m_spatialResampleEnableFlag;
  bool m_retargetingEnableFlag;
  bool m_temporalResampleEnableFlag;
  bool m_bitDepthShiftEnableFlag;

public:
  VCMPS();
  virtual ~VCMPS(){};

  int  getVPSId() const { return m_refVpsId; }
  void setVPSId(int i) { m_refVpsId = i; }
};

//------------------------------------------------------------------
//    VCM NAL sequence restoration data
//------------------------------------------------------------------
class SRD
{
private:
  int m_srdId;

public:
  SRD();
  virtual ~SRD(){};

  int  getSRDId() const { return m_srdId; }
  void setSRDId(int i) { m_srdId = i; }
};

//------------------------------------------------------------------
//    VCM NAL picture restoration data
//------------------------------------------------------------------
class PRD
{
private:
  int  m_refSRDId;
  bool m_spatialResampleEnableFlag;
  bool m_retargetingEnableFlag;
  bool m_temporalResampleEnableFlag;
  bool m_bitDepthShiftEnableFlag;

public:
  PRD();
  virtual ~PRD(){};

  int  getRefSRDId() const { return m_refSRDId; }
  void setRefSRDId(int i) { m_refSRDId = i; }
};

//------------------------------------------------------------------
//    VCM NAL picture restoration data
//------------------------------------------------------------------
class VCMHLSWriter : public HLSWriter
{
public:
  VCMHLSWriter() {}
  virtual ~VCMHLSWriter() {}

public:
  void codeVCMPS();
  void codeRSD();
  void codeCVD();
  void codeSRD();
  void codePRD();
  void codeSEI();
  void codeEOSS();

private:
};