from fastapi.testclient import TestClient
from main import app, VERSION


client = TestClient(app)


def test_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Root Endpoint of Haar API",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_version():
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Version Fetch Successful",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_detect_face():
    response = client.get("/detect/face")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Face Detection Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_detect_eye():
    response = client.get("/detect/eye")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Face and Eye Detection Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }