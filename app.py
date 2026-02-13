from fastapi import FastAPI, UploadFile, File
from inference import run_inference
from PIL import Image
import io

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    img_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    result = run_inference(pil_img)   # must return a dict
    return result
