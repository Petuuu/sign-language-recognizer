"""Main program"""

import os
from pathlib import Path
from src.image_pipeline import landmarks_to_csv, image_detect, stream_detect
from src.model.train import handle_training


def main() -> None:
    """Main function of program: Runs appropriate action based
    on user input and predicts the letter signed"""

    n = input(
        "Save landmarks from images (0), detect from images (1, default) or video (2), or train model (3)? "
    )
    if n == "0":
        path = input(
            "Path to image file/directory from project root (default 'dataset/sample/'): "
        )
        output_path = input(
            "Path to output file from project root (default 'dataset/sample.csv'): "
        )
        if path == "":
            path = "dataset/sample/"
        if output_path == "":
            output_path = "dataset/sample.csv"

        if os.path.exists(path):
            landmarks_to_csv(Path(path), output_path)
        else:
            print("Incorrect path. Exiting...")

    elif n in ("1", ""):
        path = input(
            "Path to image file/directory from project root (default 'dataset/sample/'): "
        )
        if path == "":
            path = "dataset/sample/"

        if os.path.exists(path):
            image_detect(Path(path))
        else:
            print("Incorrect path. Exiting...")

    elif n == "2":
        stream_detect()

    elif n == "3":
        handle_training()

    else:
        print("Invalid input. Exiting...")


if __name__ == "__main__":
    main()
