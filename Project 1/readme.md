
# Cat vs Dog Classification

This project classifies images of cats and dogs using traditional ML algorithms (SVM, Random Forest, KNN).

---
# APP UI

![Alt text](images/1.png)
![Alt text](images/2.png)
![Alt text](images/3.png)

## Project Structure
```bash
.
├─ app.py               # Streamlit app
├─ requirements.txt     # Required packages
├─ src/
│   └─ utilities.py     # Helper functions
└─ notebook/notebook/Porject_1_Image_class_ML.ipynb
```

## Dataset

Cat & Dog images used for training and evaluation.

---

## Results

**SVM**  
- Accuracy Score: 0.78 
- Precision Score: 0.77  
- Recall Score: 0.78 
- Cross val score: 0.77
- Confusion Matrix:  
![SVM Confusion Matrix](images/svm_cm.png)


**Random Forest**  
- Accuracy Score: 0.72
- Precision Score: 0.71
- Recall Score: 0.74
- Cross val score: 0.71
- Confusion Matrix:  
![RF Confusion Matrix](images/rf_cm.png)

**KNN**  
- Accuracy Score: 0.65 
- Precision Score: 0.60
- Recall Score: 0.86
- Cross val score: 0.64  
- Confusion Matrix:  
![KNN Confusion Matrix](images/knn_cm.png)

---

## Run App

```bash
streamlit run app.py

```

