"""Methods for image and video processing and landmarking"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarkerResult,
)
import cv2 as cv
from src.model.architecture import MLP, load_model
from src.model.helpers import classify
from src.helpers import check_file_exists

BASE_OPTIONS = mp.tasks.BaseOptions(model_asset_path="models/hand_landmarker.task")
LETTER_TO_LABEL = {
    "UNK": 0,
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "U": 20,
    "V": 21,
    "W": 22,
    "X": 23,
    "Y": 24,
}
LABEL_TO_LETTER = {
    0: "UNK",
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
}
HANDEDNESS_INDEX_TO_NAME = {0: "Right", 1: "Left"}


def normalize_landmarks(landmarks: list[NormalizedLandmark]) -> list[float]:
    """Center landmarks at the wrist and normalize their scale using the distance
    between the wrist and the middle finger's MCP joint

    Args:
        landmarks (list[NormalizedLandmark]): landmarks to normalize

    Returns:
        points (np.ndarray): normalized landmarks excluding wrist
    """
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)

    # points[0] is the landmark for the wrist
    points -= points[0]

    # points[9] is the landmark for the middle finger's MCP joint
    # `None` is returned if scale is too small
    scale = np.linalg.norm(points[9])
    if scale <= 1e-6:
        return None

    points /= scale
    points = points[1:].flatten()
    return points.tolist()


def draw_landmarks(img: np.ndarray, res: HandLandmarkerResult) -> np.ndarray:
    """Draw landmarks and handedness (which hand is in picutre) onto given image

    Args:
        img (numpy.ndarray): image to draw landmarks onto
        res (HandLandmarkerResult): result of landmark detection

    Returns:
        annotated (numpy.ndarray): copy of original image including landmarks and handedness
    """
    mp_hands = vision.HandLandmarksConnections
    mp_drawing = vision.drawing_utils
    mp_drawing_styles = vision.drawing_styles
    annotated = np.copy(img)
    if not res.hand_landmarks:
        return annotated

    landmarks = res.hand_landmarks[0]

    mp_drawing.draw_landmarks(
        annotated,
        landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )

    return annotated


def landmarks_to_csv(path: str, output_path: str = "dataset/sample.csv") -> None:
    """Detect landmarks from images and save them to CSV file. Label is extracted from the first
    letter of image files. Image must be jpg, jpeg, or png

    Args:
        path (str): path to an image directory or file
        output_path (str): path to output CSV file. Defaults to "dataset/sample.csv"
    """
    check_file_exists(output_path)
    parent = Path(output_path).parent
    if not parent.exists():
        raise FileNotFoundError(f"Output directory '{parent}' does not exist.")

    options = vision.HandLandmarkerOptions(
        base_options=BASE_OPTIONS, num_hands=2, running_mode=vision.RunningMode.IMAGE
    )
    with vision.HandLandmarker.create_from_options(options) as detector:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                if os.path.isdir(path):
                    for file in path.rglob("*"):
                        if file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                            continue

                        # OpenCV uses BGR for images, while MediaPipe uses RGB
                        # -> colors need to be converted
                        img = cv.imread(str(file))
                        as_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=as_rgb)
                        res = detector.detect(mp_img)

                        # format result and save to file
                        if len(res.hand_landmarks) > 1:
                            raise ValueError(
                                "Only one hand allowed in a single picture"
                            )
                        label = LETTER_TO_LABEL[file.stem[0].upper()]
                        handedness = res.handedness[0][0].index
                        normalized = ",".join(
                            [str(x) for x in normalize_landmarks(res.hand_landmarks[0])]
                        )
                        f.write(f"{label},{handedness},{normalized}\n")

                else:
                    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        print("Given file is not an image. Exiting...")
                        sys.exit()

                    img = cv.imread(str(path))
                    as_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=as_rgb)
                    res = detector.detect(mp_img)

                    if len(res.hand_landmarks) > 1:
                        raise ValueError("Only one hand allowed in a single picture")
                    label = LETTER_TO_LABEL[path.stem[0].upper()]
                    handedness = res.handedness[0][0].index
                    normalized = ",".join(
                        [str(x) for x in normalize_landmarks(res.hand_landmarks[0])]
                    )
                    f.write(f"{label},{handedness},{normalized}\n")

        except Exception:
            print(f"File '{output_path}' could not be opened. Exiting...")


def predict(model: MLP, res: HandLandmarkerResult) -> None:
    """Predicts and outputs prediction and probabilities from HandLandmarker
    objects result

    Args:
        model (MLP)               : model to make prediction with
        res (HandLandmarkerResult): HandLandmarker object's result
    """
    landmarks = normalize_landmarks(res.hand_landmarks[0])
    model_input = np.array([res.handedness[0][0].index, *landmarks])
    probas, pred = classify(model(model_input))
    print("\nPROBABILITIES:\n", probas)
    print("PREDICTION:", LABEL_TO_LETTER[pred])


def image_detect(path: str, model_path: str) -> None:
    """Detect landmarks from images, make prediction of letter, and display them.
    Image must be jpg, jpeg, or png

    Args:
        path       (str): path to an image directory or file
        model_path (str): path to model parameters
    """
    model = MLP()
    load_model(model, model_path)

    options = vision.HandLandmarkerOptions(
        base_options=BASE_OPTIONS, num_hands=1, running_mode=vision.RunningMode.IMAGE
    )
    with vision.HandLandmarker.create_from_options(options) as detector:
        if os.path.isdir(path):
            for file in path.rglob("*"):
                if file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                img = cv.imread(str(file))
                as_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=as_rgb)
                res = detector.detect(mp_img)

                if res.hand_landmarks:
                    predict(model, res)

                annotated = draw_landmarks(mp_img.numpy_view(), res)
                as_bgr = cv.cvtColor(annotated, cv.COLOR_RGB2BGR)
                cv.imshow("Image", as_bgr)
                cv.waitKey(0)
                cv.destroyAllWindows()

        else:
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                print("Given file is not an image")
                sys.exit()

            img = cv.imread(str(path))
            as_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=as_rgb)
            res = detector.detect(mp_img)

            if res.hand_landmarks:
                predict(model, res)

            annotated = draw_landmarks(mp_img.numpy_view(), res)
            as_bgr = cv.cvtColor(annotated, cv.COLOR_RGB2BGR)
            cv.imshow("Image", as_bgr)
            cv.waitKey(0)
            cv.destroyAllWindows()


def stream_detect(model_path: str) -> None:
    """Detect landmarks from video stream and display them

    Args:
        model_path (str): path to model parameters
    """
    model = MLP()
    load_model(model, model_path)

    options = vision.HandLandmarkerOptions(
        base_options=BASE_OPTIONS, num_hands=1, running_mode=vision.RunningMode.VIDEO
    )
    with vision.HandLandmarker.create_from_options(options) as detector:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            sys.exit()

        # OpenCV video capture loop
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Cannot receive frame (stream end?). Exiting...")
                break

            as_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=as_rgb)
            timestamp_ms = time.monotonic_ns() // 1000000
            res = detector.detect_for_video(mp_img, timestamp_ms)

            if res.hand_landmarks:
                predict(model, res)

            annotated = draw_landmarks(mp_img.numpy_view(), res)
            as_bgr = cv.cvtColor(cv.flip(annotated, 1), cv.COLOR_RGB2BGR)
            cv.imshow("Image", as_bgr)
            if cv.waitKey(1) == ord("q"):
                break

        cap.release()
        cv.destroyAllWindows()
