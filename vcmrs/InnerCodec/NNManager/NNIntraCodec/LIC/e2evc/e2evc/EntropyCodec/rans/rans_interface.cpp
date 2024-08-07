/* Copyright (c) 2021-2022, InterDigital Communications, Inc
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of InterDigital Communications, Inc nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 * THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "rans_interface.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

#include "ryn_rans/rans64.h"

namespace py = pybind11;

/* probability range, this could be a parameter... */
constexpr int precision = 16;

constexpr uint16_t bypass_precision = 4; /* number of bits in bypass mode */
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

// Out-of-range symbols CDF index in a CDF array
constexpr int OOR_CDF_IDX = 30;

namespace {

/* We only run this in debug mode as its costly... */
void assert_cdfs(const std::vector<std::vector<int>> &cdfs,
                 const std::vector<int> &cdfs_sizes) {
  for (int i = 0; i < static_cast<int>(cdfs.size()); ++i) {
    assert(cdfs[i][0] == 0);
    assert(cdfs[i][cdfs_sizes[i] - 1] == (1 << precision));
    for (int j = 0; j < cdfs_sizes[i] - 1; ++j) {
      assert(cdfs[i][j + 1] > cdfs[i][j]);
    }
  }
}

/* Support only 16 bits word max */
inline void Rans64EncPutBits(Rans64State *r, uint32_t **pptr, uint32_t val,
                             uint32_t nbits) {
  assert(nbits <= 16);
  assert(val < (1u << nbits));

  /* Re-normalize */
  uint64_t x = *r;
  uint32_t freq = 1 << (16 - nbits);
  uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
  if (x >= x_max) {
    *pptr -= 1;
    **pptr = (uint32_t)x;
    x >>= 32;
    Rans64Assert(x < x_max);
  }

  /* x = C(s, x) */
  *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr,
                                 uint32_t n_bits) {
  uint64_t x = *r;
  uint32_t val = x & ((1u << n_bits) - 1);

  /* Re-normalize */
  x = x >> n_bits;
  if (x < RANS64_L) {
    x = (x << 32) | **pptr;
    *pptr += 1;
    Rans64Assert(x >= RANS64_L);
  }

  *r = x;

  return val;
}
} // namespace

void BufferedRansEncoder::encode_symbol(
    const std::vector<int32_t> cdf,
    int32_t value) {
  _syms.push_back({static_cast<uint16_t>(cdf[value]),
                     static_cast<uint16_t>(cdf[value + 1] - cdf[value]),
                     false});
}

void BufferedRansEncoder::encode_oor_bypass(const int32_t value, 
       const int32_t max_value) {
    // Bypass coding mode 
    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = -2 * value - 1;
    } else if (value >= max_value) {
      raw_val = 2 * (value - max_value);
    }

    // Determine the number of bypasses (in bypass_precision size) needed to
    //   encode the raw value. 
    int32_t n_bypass = 0;
    while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
      ++n_bypass;
    }

    /* Encode number of bypasses */
    int32_t val = n_bypass;
    while (val >= max_bypass_val) {
      _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
      val -= max_bypass_val;
    }
    _syms.push_back(
        {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

    /* Encode raw value */
    for (int32_t j = 0; j < n_bypass; ++j) {
      const int32_t val =
          (raw_val >> (j * bypass_precision)) & max_bypass_val;
      _syms.push_back(
          {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});
    }
}

void BufferedRansEncoder::encode_oor_cmpr(int32_t value, 
      const int32_t max_value,
      const std::vector<std::vector<int32_t>> &cdfs,
      const std::vector<int32_t> &cdfs_sizes,
      const std::vector<int32_t> &offsets,
      const int32_t cdf_idx) {

    // out of range value cdf
    const int idx_oor = std::max(OOR_CDF_IDX, cdf_idx);
    const auto &cdf_oor = cdfs[idx_oor];
    const int32_t max_value_oor = cdfs_sizes[idx_oor] - 2;
    const int32_t offset_oor = offsets[idx_oor];

    if (value >= max_value) {
      value = (value - max_value) - offset_oor;

      // encode remaining using cdf_oor
      while (value >= max_value_oor) {
        encode_symbol(cdf_oor, max_value_oor);
        value = (value - max_value_oor) - offset_oor;
      }
      encode_symbol(cdf_oor, value);
    } else if (value<0) {
      value = value - offset_oor; 

      while (value < 0) {
        encode_symbol(cdf_oor, max_value_oor);
        value = value - offset_oor; 
      }
      encode_symbol(cdf_oor, value);
    }
}

void BufferedRansEncoder::encode_with_indexes(
    const std::vector<int32_t> &symbols, const std::vector<int32_t> &indexes,
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes,
    const std::vector<int32_t> &offsets, 
    const bool oor_bypass) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  // backward loop on symbols from the end;
  for (size_t i = 0; i < symbols.size(); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());
    const int32_t offset = offsets[cdf_idx];

    int32_t value = symbols[i] - offset;

    // out of range value cdf
    if (value<0 || value>=max_value) {

      // set OOR tag
      encode_symbol(cdf, max_value);

      if (oor_bypass) {
        encode_oor_bypass(value, max_value);
      } else {
        encode_oor_cmpr(value, max_value, 
          cdfs, cdfs_sizes, offsets, cdf_idx);
      }
    } else {
      // normal encoding
      encode_symbol(cdf, value);
    }
  }
}

py::bytes BufferedRansEncoder::flush() {
  Rans64State rans;
  Rans64EncInit(&rans);

  std::vector<uint32_t> output(_syms.size()+2, 0xCC); // too much space ?
  uint32_t *ptr = output.data() + output.size();
  assert(ptr != nullptr);

  while (!_syms.empty()) {
    const RansSymbol sym = _syms.back();

    if (!sym.bypass) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    } else {
      // unlikely...
      Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    }

    _syms.pop_back();
  }

  Rans64EncFlush(&rans, &ptr);


  const int nbytes =
      std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);

  return std::string(reinterpret_cast<char *>(ptr), nbytes);
}

py::bytes
RansEncoder::encode_with_indexes(const std::vector<int32_t> &symbols,
                                 const std::vector<int32_t> &indexes,
                                 const std::vector<std::vector<int32_t>> &cdfs,
                                 const std::vector<int32_t> &cdfs_sizes,
                                 const std::vector<int32_t> &offsets,
                                 const bool oor_bypass) {

  BufferedRansEncoder buffered_rans_enc;
  buffered_rans_enc.encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes,
                                        offsets, oor_bypass);
  return buffered_rans_enc.flush();
}

void RansDecoder::set_stream(const std::string &encoded) {
  _stream = encoded;
  uint32_t *ptr = (uint32_t *)_stream.data();
  assert(ptr != nullptr);
  _ptr = ptr;
  Rans64DecInit(&_rans, &_ptr);
}

int32_t RansDecoder::decode_symbol(const std::vector<int32_t> &cdf,
                                   const int32_t cdf_size,
                                   const int32_t offset){
    const uint32_t cum_freq = Rans64DecGet(&_rans, precision);

    const auto cdf_end = cdf.begin() + cdf_size;
    const auto it = std::find_if(cdf.begin(), cdf_end,
                                 [cum_freq](int v) { return v > cum_freq; });
    assert(it != cdf_end + 1);
    const uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&_rans, &_ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);
    return value;
}

int32_t RansDecoder::decode_oor_bypass(const int32_t max_value,
                        const int32_t offset){
    /* Bypass decoding mode */
    int32_t val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
    int32_t n_bypass = val;

    while (val == max_bypass_val) {
      val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
      n_bypass += val;
    }

    int32_t raw_val = 0;
    for (int j = 0; j < n_bypass; ++j) {
      val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
      assert(val <= max_bypass_val);
      raw_val |= val << (j * bypass_precision);
    }
    int32_t value = raw_val >> 1;
    if (raw_val & 1) {
      value = -value - 1;
    } else {
      value += max_value;
    }

    return value + offset;
 
}

int32_t RansDecoder::decode_oor_cmpr(
      const std::vector<std::vector<int32_t>> &cdfs,
      const std::vector<int32_t> &cdfs_sizes,
      const std::vector<int32_t> &offsets,
      const int32_t cdf_idx, 
      const int32_t max_value,
      const int32_t offset) {

    int32_t real_pos_value = 0;
    int32_t real_neg_value = 0;
    int32_t real_value = 0;

    // OOR cdf
    const int idx_oor = std::max(OOR_CDF_IDX, cdf_idx);
    const auto &cdf_oor = cdfs[idx_oor];
    const int32_t max_value_oor = cdfs_sizes[idx_oor] - 2;
    const int32_t offset_oor = offsets[idx_oor];


    real_pos_value += max_value + offset;
    real_neg_value += offset;
  
    int32_t value = decode_symbol(cdf_oor,  cdfs_sizes[idx_oor], offset_oor);
  
    while (value == max_value_oor) {
          real_pos_value += max_value_oor + offset_oor;
          real_neg_value += offset_oor;
          value = decode_symbol(cdf_oor,  cdfs_sizes[idx_oor], offset_oor);
    }
    if (value + offset_oor >=0) {
        real_value = real_pos_value;
    } else {
        real_value = real_neg_value;
    }
    real_value += value + offset_oor;
    return real_value;
}


std::vector<int32_t>
RansDecoder::decode_stream(const std::vector<int32_t> &indexes,
                           const std::vector<std::vector<int32_t>> &cdfs,
                           const std::vector<int32_t> &cdfs_sizes,
                           const std::vector<int32_t> &offsets,
                           const bool oor_bypass) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  std::vector<int32_t> output(indexes.size());

  assert(_ptr != nullptr);

  for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    const int32_t offset = offsets[cdf_idx];
    int32_t value = decode_symbol(cdf,  cdfs_sizes[cdf_idx], offset);

    int32_t real_value = 0;

    // process out-of-range values
    if (value == max_value) {
      if (oor_bypass) {
        real_value = decode_oor_bypass(max_value, offset);
      } else {
        real_value = decode_oor_cmpr(cdfs, cdfs_sizes, offsets, cdf_idx, max_value, offset);
      }
    } else {
      real_value += value + offset;
    }

    output[i] = real_value;
  }

  return output;
}

PYBIND11_MODULE(ans, m) {
  m.attr("__name__") = "ans";

  m.doc() = "range Asymmetric Numeral System python bindings";

  py::class_<BufferedRansEncoder>(m, "BufferedRansEncoder")
      .def(py::init<>())
      .def("encode_with_indexes", &BufferedRansEncoder::encode_with_indexes)
      .def("flush", &BufferedRansEncoder::flush);

  py::class_<RansEncoder>(m, "RansEncoder")
      .def(py::init<>())
      .def("encode_with_indexes", &RansEncoder::encode_with_indexes);

  py::class_<RansDecoder>(m, "RansDecoder")
      .def(py::init<>())
      .def("set_stream", &RansDecoder::set_stream)
      .def("decode_stream", &RansDecoder::decode_stream);

}
