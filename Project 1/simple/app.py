import streamlit as st
import joblib
from PIL import Image
import numpy as np
from skimage.feature import hog

# svm = joblib.load("models/svm_pipeline")
svm=joblib.load("models/svm_pipeline")
rf =  joblib.load("models/rf_pipeline")
knn = joblib.load("models/knn_pipeline")

st.title("Image Classification Using Machine Leaning Algorithms")
st.header("Cats & Dogs classification")


choice = st.sidebar.radio("Choose Model",
    ["Support Vector Machine", "Random Forest", "KNN"]
)

model_chose = {
            "Support Vector Machine":svm,
            "Random Forest":rf,
            "KNN":knn
            }

pipeline = model_chose[choice]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=512)
    image = Image.open(uploaded_file).convert("L")

    if st.button("Get Prediction"):
        # preprocessing 
        image = image.resize((64,64)) # resize image
        img_arr = np.array(image, dtype=np.float32) / 255.0
        hog_feat = hog(
            img_arr,
            orientations=9 ,
            pixels_per_cell = (8,8), 
            cells_per_block = (2,2), 
            block_norm = "L2-Hys", 
            channel_axis=None 
            )

        pred = pipeline.predict([hog_feat])
        if pred == 0:
            st.success("The image is a CAT")
        else:
            st.success("The image is a DOG")
        



