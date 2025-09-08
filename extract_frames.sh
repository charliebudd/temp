#!/bin/bash
find sarrarp -type f -name "*.avi" -exec bash -c '
for video; do
    base_dir="$(dirname "$video")"
    frames_dir="$base_dir/frames"
    mkdir -p "$frames_dir"
    echo "$video $frames_dir"
    ffmpeg -i "$video" -vf "select=not(mod(n\,60))" -vsync vfr \
        -frame_pts 1 "$frames_dir/%09d.png"
done
' _ {} +
