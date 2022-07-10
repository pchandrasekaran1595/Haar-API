from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from static.utils import Model, decode_image, encode_image_to_base64


VERSION: str = "1.0.0"
STATIC_PATH: str = "static"


class Image(BaseModel):
    imageData: str


origins = [
    "http://localhost:10011",
]

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "statusText" : "Root Endpoint of Haar API",
        "statusCode" : 200,
        "version" : VERSION,
    }


@app.get("/version")
async def get_version():
    return {
        "statusText" : "Version Fetch Successful",
        "statusCode" : 200,
        "version" : VERSION,
    }


@app.get("/detect/face")
async def get_detect_face():
    return {
        "statusText" : "Face Detection Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


@app.get("/detect/eye")
async def get_detect_eye():
    return {
        "statusText" : "Face and Eye Detection Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


@app.post("/detect/face")
async def post_detect_face(image: Image):
    _, image = decode_image(image.imageData)

    model = Model(model_type="face")
    face_detections_np, _ = model.detect(image)

    face_detections: list = []
    for (x, y, w, h) in face_detections_np:
        face_detections.append([int(x), int(y), int(w), int(h)])
    

    return {
        "statusText" : "Face Detection Successful",
        "statusCode" : 200,
        "face_detections" : face_detections,
    }


@app.post("/detect/eye")
async def post_detect_eye(image: Image):
    _, image = decode_image(image.imageData)

    model = Model(model_type="eye")
    face_detections_np, eye_detections_np = model.detect(image)

    face_detections: list = []
    eye_detections: list  = []

    for (x, y, w, h) in face_detections_np:
        face_detections.append([int(x), int(y), int(w), int(h)])
    
    for (x, y, w, h) in eye_detections_np:
        eye_detections.append([int(x), int(y), int(w), int(h)])

    return {
        "statusText" : "Face Detection Successful",
        "statusCode" : 200,
        "face_detections" : face_detections,
        "eye_detections" : eye_detections,
    }
