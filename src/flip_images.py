"""Flips dataset images to support recognition for left hand"""

from pathlib import Path
from PIL import Image

paths = input("Directories seperated by spaces: ")
paths = paths.split()

for x in paths:
    path = Path(x)
    for file in path.rglob("*"):
        if file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        img = Image.open(file)
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped.save(f"{path}/{file.stem}_left{file.suffix}")
