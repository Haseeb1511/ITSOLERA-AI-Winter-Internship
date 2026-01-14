
# Cat vs Dog Classification

This project classifies images of cats and dogs using traditional ML algorithms (SVM, Random Forest, KNN).

---
# APP UI

![Alt text](images/1.png)
![Alt text](images/2.png)
![Alt text](images/3.png)

## Project Structure

.
├─ app.py               # Streamlit app
├─ requirements.txt     # Required packages
├─ src/
│   └─ utilities.py     # Helper functions
└─ notebook/notebook/Porject_1_Image_class_ML.ipynb


## Dataset

Cat & Dog images used for training and evaluation.

---

## Results

**SVM**  
- Accuracy Score: 0.683  
- Precision Score: {precision}  
- Recall Score: 0.617  
- Confusion Matrix:  
![SVM Confusion Matrix](images/svm_cm.png)


**Random Forest**  
- Accuracy Score: 0.677  
- Precision Score: {precision}  
- Recall Score: 0.606  
- Confusion Matrix:  
![RF Confusion Matrix](images/rf.png)

**KNN**  
- Accuracy Score: 0.62  
- Precision Score: {precision}  
- Recall Score: 0.569  
- Confusion Matrix:  
![KNN Confusion Matrix](images/knn.png)

---

## Run App

```bash
streamlit run app.py

```

