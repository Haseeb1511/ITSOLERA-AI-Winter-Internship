from fastapi import FastAPI,UploadFile,File
from fastapi import HTTPException
import uvicorn
import joblib
import numpy as np
from skimage.feature import hog
import io
from PIL import Image
import os
import sqlite3
from datetime import datetime


app = FastAPI()  # fastapi object

svm = joblib.load("models/svm_pipeline")
rf = joblib.load("models/rf_pipeline")
knn = joblib.load("models/knn_pipeline")

models = {
    "svm": svm,
    "rf": rf,
    "knn": knn
}

# ============================ Database  part ================================
DB_PATH = "/app/data/predictions.db" # must be same in streamlit (both share same database-> also change in docker compose)

# create table if not exists
def init_db():
    conn = sqlite3.connect(DB_PATH)
    # create pointer it let us execute sql commands
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS predictions(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   image_name TEXT,
                   prediction text,
                   timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                   )
                   ''')
    # save changes to database
    conn.commit() 
    conn.close()

# execute
init_db()

# log prediction and image name in to database
def log_prediction(image_name,prediciton):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions(image_name,prediction) VALUES (?,?)",
        (image_name,prediciton)
    )
    conn.commit()
    conn.close()



# ============================================== Model Part =====================

@app.get("/")
def health_check():
    return {"message":"Good Health"}


@app.post("/predict")
async def predict(model_name: str, file: UploadFile = File(...)):

    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # read image
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")
    
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format")

    #preprocessing
    image = image.resize((64,64))
    img_arr = np.array(image,dtype=np.float32)/ 255.0

    hog_feat = hog(
        img_arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        channel_axis=None
    )

    if model_name not in models:
        raise HTTPException(status_code=400, detail="Invalid model name")

    model = models[model_name]

    pred = model.predict([hog_feat])[0]

    label = "CAT" if pred == 0 else "DOG"

    log_prediction(file.filename,label)  # add output to database
    return {
        "prediction":label
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)