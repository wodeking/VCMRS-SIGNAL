# convert png to yuv by ffmpeg


cd ..

ffmpeg -i lighthouse.png \
  -y \
  -f rawvideo \
  -pix_fmt 'yuv420p' \
  -dst_range 1 \
  lighthouse_512x640_420p.yuv

ffmpeg -i lighthouse.png \
  -y \
  -f rawvideo \
  -pix_fmt 'yuv420p10le' \
  -dst_range 1 \
  lighthouse_512x640_420p_10bit.yuv

ffmpeg -i lighthouse.png \
  -y \
  -f rawvideo \
  -pix_fmt 'yuv444p' \
  -dst_range 1 \
  lighthouse_512x640_444p.yuv

ffmpeg -i lighthouse.png \
  -y \
  -f rawvideo \
  -pix_fmt 'yuv444p10le' \
  -dst_range 1 \
  lighthouse_512x640_444p_10bit.yuv

