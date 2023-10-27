from fastapi import FastAPI
from pydantic import BaseModel

class VideoSimilarityRequest(BaseModel):
    tutorial_uri: str
    performance_video_uri: str
    selected_model_name: str 

class SignRecognitionRequest(BaseModel):
    sign_image_uri: str
    selected_model_name: str

app = FastAPI()

@app.get("/", response_model=str)
async def root_requests():
    return "FastAPI Server is running."

@app.post("/api/v1/video/similarity", response_model=float)
async def video_similarity(videos : VideoSimilarityRequest):
    return 0.993

@app.post("/api/v1/image/recognize/sign", response_model=str)
async def video_similarity(request: SignRecognitionRequest):
    return "Ka"


