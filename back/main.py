from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from inference import _inference

app = FastAPI()

origins = [
    "http://localhost:3000",
]
# react localhost

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/uploadfile") 
async def infereceImage(myFile: UploadFile = File(...)):
    # response = await fetch_gt(idx)
    out_path = './images/test.png'
    async with aiofiles.open(out_path, 'wb') as out_file:
        content = await myFile.read()
        await out_file.write(content)
    
    prediction = await _inference(out_path)

    return prediction

