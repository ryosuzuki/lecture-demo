#!/usr/bin/env python3
"""Download retained release videos and build a fully local playback page."""
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parent
BASE="https://github.com/ryosuzuki/lecture-demo/releases/download/physical-ai-demo-videos-20261014/"
media=ROOT/"media"
media.mkdir(exist_ok=True)
s=(ROOT/"index.html").read_text()
for name in ["robot","vla","road","detect","segment","openpilot","stretch","failure"]:
 p=media/(name+".mp4")
 if not p.exists():
  temp=p.with_suffix(".part")
  urllib.request.urlretrieve(BASE+p.name,temp)
  temp.replace(p)
 s=s.replace(BASE+p.name,"media/"+p.name)
(ROOT/"offline.html").write_text(s)
print("Open",ROOT/"offline.html")
