# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import ans

class RansEntropyCoder:
    """Proxy class to an actual entropy coder class.
    """
    def __init__(self):
        self.encoder = ans.BufferedRansEncoder()
        self.decoder = ans.RansDecoder()

    #################
    # encode  to stream
    #const std::vector<int32_t> &symbols, 
    #const std::vector<int32_t> &indexes,
    #const std::vector<std::vector<int32_t>> &cdfs,
    #const std::vector<int32_t> &cdfs_sizes,
    #const std::vector<int32_t> &offsets) {
 
    def encode_with_indexes(self, symbols, indexes, cdfs, cdf_sizes, offsets, oor_bypass=False):
        return self.encoder.encode_with_indexes(
          symbols, indexes, cdfs, cdf_sizes, offsets, oor_bypass)

    # return bytes
    def flush(self):
        return self.encoder.flush()

    #################
    # decode from a bitstream

    def set_stream(self, bitstream):
        return self.decoder.set_stream(bitstream)

    def decode_stream(self, indexes, cdfs, cdfs_sizes, offsets, oor_bypass=False):
        return self.decoder.decode_stream(indexes, cdfs, cdfs_sizes, offsets, oor_bypass)




