import streamlit as st
import requests


st.title("Image Classification Using Machine Learning")
st.header("Cats & Dogs Classification")


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

API_BASE_URL = "http://localhost:8000"
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
            files={"file": uploaded_file.getvalue()}
        )

        if response.status_code != 200:
            st.error(response.text)
        else:
            result = response.json()
            st.success(f"The image is a {result['prediction']}")
