import os
import re
import io
import cv2
import base64
import numpy as np

from PIL import Image

STATIC_PATH = "static"


def decode_image(imageData) -> np.ndarray:
    header, imageData = imageData.split(",")[0], imageData.split(",")[1]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(imageData))))
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2BGR)
    return header, image


def encode_image_to_base64(header: str = "data:image/png;base64", image: np.ndarray = None) -> str:
    assert image is not None, "Image is None"
    _, imageData = cv2.imencode(".jpeg", image)
    imageData = base64.b64encode(imageData)
    imageData = str(imageData).replace("b'", "").replace("'", "")
    imageData = header + "," + imageData
    return imageData


def gray(image: np.ndarray, rgb: bool=False) -> np.ndarray:
    if rgb: return cv2.cvtColor(src=image, code=cv2.COLOR_RGB2GRAY)
    else: return cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)


class Model(object):
    def __init__(self, model_type="face"):
        self.model_type = model_type

        assert re.match(r"^face$", self.model_type, re.IGNORECASE) \
            or re.match(r"^eye$", self.model_type, re.IGNORECASE), "Invalid mode"

        if re.match(r"^face$", self.model_type, re.IGNORECASE):
            self.model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        elif re.match(r"^eye$", self.model_type, re.IGNORECASE):
            self.model_1 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.model_2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def detect(self, image):
        temp_image = gray(image.copy())
        if re.match(r"face", self.model_type, re.IGNORECASE):
            detections = self.model.detectMultiScale(image=temp_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            return detections, None
        elif re.match(r"eye", self.model_type, re.IGNORECASE):
            eye_detections = None
            face_detections = self.model_1.detectMultiScale(image=temp_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in face_detections:
                roi_image = gray(image[y:y+h, x:x+w].copy())
                eye_detections = self.model_2.detectMultiScale(image=roi_image, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            return face_detections, eye_detections