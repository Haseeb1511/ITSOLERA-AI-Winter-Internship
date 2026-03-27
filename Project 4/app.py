import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
import warnings

warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(page_title="Adaptive Data Augmentation", layout="wide")

# Styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .report-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Adaptive Synthetic Data Augmentation Toolkit")
st.markdown("Improving model performance on imbalanced healthcare data using a closed-loop feedback system with SDV.")

# Sidebar
st.sidebar.header("Configuration")
MAX_ITER = st.sidebar.slider("Max Iterations", 1, 5, 2)
AUGMENT_SIZE = st.sidebar.number_input("Augmentation Size per Iteration", 10, 200, 50)

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=columns, na_values='?')
    df.dropna(inplace=True)
    df['target'] = (df['target'] > 0).astype(int)
    return df

def preprocess(X_tr, X_te):
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    return X_tr_scaled, X_te_scaled

def get_undersampled_class(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    classes = [c for c in report.keys() if c.isdigit()]
    recalls = {c: report[c]['recall'] for c in classes}
    weak_class = min(recalls, key=recalls.get)
    return int(weak_class), recalls[weak_class]

def train_synthesizer(train_data, metadata):
    synthesizer = TVAESynthesizer(metadata)
    synthesizer.fit(train_data)
    return synthesizer

# 1. Data Loading
df = load_data()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head(), width='stretch')
    st.write(f"**Total Samples:** {len(df)}")

with col2:
    st.subheader("🎯 Target Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='viridis', ax=ax)
    st.pyplot(fig)

# Session State for tracking
if 'baseline_run' not in st.session_state:
    st.session_state.baseline_run = False
if 'loop_run' not in st.session_state:
    st.session_state.loop_run = False

st.divider()

# 2. Baseline Model
if st.button("🚀 Train Baseline Model"):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.baseline_f1 = f1_score(y_test, y_pred)
    st.session_state.baseline_report = classification_report(y_test, y_pred, output_dict=True)
    st.session_state.baseline_run = True
    st.session_state.clf = clf

if st.session_state.baseline_run:
    st.subheader("📉 Baseline Performance")
    b_report = st.session_state.baseline_report
    cols = st.columns(3)
    cols[0].metric("F1-Score", f"{st.session_state.baseline_f1:.4f}")
    cols[1].metric("Recall (Class 0)", f"{b_report['0']['recall']:.4f}")
    cols[2].metric("Recall (Class 1)", f"{b_report['1']['recall']:.4f}")

    # 3. Adaptive Loop
    st.divider()
    if st.button("🔄 Start Adaptive Augmentation Loop"):
        st.session_state.loop_run = True
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        history = []
        current_X_train = st.session_state.X_train.copy()
        current_y_train = st.session_state.y_train.copy()
        current_clf = st.session_state.clf
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=current_X_train.join(current_y_train))

        for i in range(MAX_ITER):
            status_text.text(f"Running Iteration {i+1}/{MAX_ITER}...")
            
            # Error Analysis
            _, X_test_s = preprocess(current_X_train, X_test)
            y_pred_iter = current_clf.predict(X_test_s)
            weak_class, min_recall = get_undersampled_class(y_test, y_pred_iter)
            f1 = f1_score(y_test, y_pred_iter)
            
            history.append({'iteration': i, 'f1': f1, 'min_recall': min_recall, 'class': weak_class})
            
            # Targeted Generation
            synthesizer = train_synthesizer(current_X_train.join(current_y_train), metadata)
            targeted_condition = Condition(
                num_rows=AUGMENT_SIZE,
                column_values={'target': weak_class}
            )
            synthetic_data = synthesizer.sample_from_conditions(conditions=[targeted_condition])
            
            # Augment
            current_X_train = pd.concat([current_X_train, synthetic_data.drop('target', axis=1)])
            current_y_train = pd.concat([current_y_train, synthetic_data['target']])
            
            current_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            X_train_s, _ = preprocess(current_X_train, X_test)
            current_clf.fit(X_train_s, current_y_train)
            
            progress_bar.progress((i + 1) / MAX_ITER)

        status_text.text("Adaptive Loop Complete!")
        st.session_state.history = history
        st.session_state.final_clf = current_clf
        st.session_state.current_X_train = current_X_train

if st.session_state.loop_run and 'history' in st.session_state:
    st.subheader("📈 Performance Improvement")
    hist_df = pd.DataFrame(st.session_state.history)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=hist_df, x='iteration', y='f1', marker='o', label='F1 Score', ax=ax1)
    ax2 = ax1.twinx()
    sns.lineplot(data=hist_df, x='iteration', y='min_recall', marker='s', color='orange', label='Min Recall', ax=ax2)
    
    ax1.set_title("Metrics Over Iterations")
    st.pyplot(fig)
    
    # Final Report
    st.divider()
    st.subheader("🏁 Final Comparison")
    _, X_test_s = preprocess(st.session_state.current_X_train, st.session_state.X_test)
    y_final_pred = st.session_state.final_clf.predict(X_test_s)
    
    f_report = classification_report(st.session_state.y_test, y_final_pred, output_dict=True)
    final_f1 = f1_score(st.session_state.y_test, y_final_pred)
    improvement = (final_f1 - st.session_state.baseline_f1) / st.session_state.baseline_f1 * 100
    
    cols = st.columns(3)
    cols[0].metric("Final F1-Score", f"{final_f1:.4f}", delta=f"{improvement:.2f}%")
    cols[1].metric("Final Recall (C0)", f"{f_report['0']['recall']:.4f}")
    cols[2].metric("Final Recall (C1)", f"{f_report['1']['recall']:.4f}")

    st.text("Classification Report:")
    st.code(classification_report(st.session_state.y_test, y_final_pred))
