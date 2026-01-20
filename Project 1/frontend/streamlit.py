import streamlit as st
import requests
import os
import sqlite3

st.title("Image Classification Using Machine Learning")
st.header("Cats & Dogs Classification")



DB_PATH = "/app/data/predictions.db" # must be same in fastapi (both share same database-> also change in docker compose)

def get_predictions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    data = cursor.fetchall()
    conn.close()
    return data






choice = st.sidebar.radio(
    "Choose Model",
    ["Support Vector Machine", "Random Forest", "KNN"]
)

model_map = {
    "Support Vector Machine": "svm",
    "Random Forest": "rf",
    "KNN": "knn"
}


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000")  # for server (aws) or docker compose this work
# API_BASE_URL = "http://localhost:8000"
#api = "http://127.0.0.1:8000/predict"

ENDPOINTS = {
    "predict": "/predict",
    "health": "/"
}




if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=512)

    if st.button("Get Prediction"):
        files = {"file": uploaded_file.getvalue()}
        url = f"{API_BASE_URL}{ENDPOINTS['predict']}"
        
        # parameter to pass to predict api
        param = {"model_name":model_map[choice]}
        
        # we need to parameter for api 
        response = requests.post(
            url,
            params=param,
            files={"file": (uploaded_file.name,uploaded_file,uploaded_file.type)}
        )

        if response.status_code != 200:
            st.error(response.text)
        else:
            result = response.json()
            st.success(f"The image is a {result['prediction']}")

            st.subheader("Logged Predictions")
            predictions = get_predictions()
            for p in predictions:
                st.write(f"Image: {p[1]}, Prediction: {p[2]}, Time: {p[3]}")
