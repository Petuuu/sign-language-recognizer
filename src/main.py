import os
from pathlib import Path
from src.image_pipeline import image_detect, stream_detect


def main() -> None:
    """Main function of program: Runs appropriate landmark detection based
    on user input and predicts the letter signed"""

    n = input("Detect from images (1, default) or video (2)? ")
    if n == "1" or n == "":
        path = input(
            "Path to image file/directory from project root (default dataset/sample/): "
        )
        if path == "":
            path = "dataset/sample/"

        if os.path.exists(path):
            image_detect(Path(path))
        else:
            print("Incorrect path. Exiting...")

    else:
        stream_detect()


if __name__ == "__main__":
    main()
