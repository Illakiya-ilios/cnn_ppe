"""
Download a YouTube clip as a single MP4 for PPE video testing.

Uses a progressive (pre-merged) format so NO ffmpeg is required. Audio is
irrelevant for detection, so we prefer a single stream that already contains
video+audio, falling back to best video-only if needed.

Usage:
    python youtube.py                      # downloads the default URL below
    python youtube.py <youtube_url>        # downloads a specific URL

Output: test_clip.mp4 in the project folder, ready for:
    python test_video.py --source test_clip.mp4
"""

import sys

from yt_dlp import YoutubeDL

DEFAULT_URL = "https://youtu.be/PiklWx68dSI?si=aahQSyoqMCZx9mfu"

url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL

ydl_opts = {
    # No ffmpeg available, and (without a JS runtime) YouTube exposes no
    # progressive combined streams for many videos. PPE detection needs no
    # audio, so download a SINGLE video-only MP4 over https (not m3u8) -- this
    # requires no merge and no ffmpeg.
    #
    # Preference order (all single-file, no merge):
    #   1. 480p mp4 video-only over https   (good detail/size balance)
    #   2. 720p mp4 video-only over https
    #   3. 360p mp4 video-only over https
    #   4. any best progressive mp4 (if one exists)
    #   5. best mp4 video-only
    "format": (
        "135[protocol^=http]/136[protocol^=http]/134[protocol^=http]/"
        "best[ext=mp4][acodec!=none][vcodec!=none]/"
        "bestvideo[ext=mp4][protocol^=http]"
    ),
    "outtmpl": "test_clip.%(ext)s",
    "noplaylist": True,
}

print(f"Downloading: {url}")
with YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

# Re-mux via OpenCV so the file plays in any Windows media player.
# The dash-only download is video-only and some players reject it.
import cv2, os
raw = "test_clip.mp4"
playable = "test_clip_playable.mp4"
if os.path.exists(raw):
    cap = cv2.VideoCapture(raw)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    out = cv2.VideoWriter(playable, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.write(f)
    cap.release()
    out.release()
    print(f"Playable copy: {playable} ({w}x{h})")

print("\nDone.")
print(f"  For detection:  python test_video.py --source {raw}")
print(f"  For watching:   open {playable}")
