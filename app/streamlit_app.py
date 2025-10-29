import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, auc, f1_score, precision_score, recall_score
import urllib.request
import os

st.set_page_config(page_title="Spam Email Classifier", layout="wide")
st.title("Spam Email Classifier (SVM Baseline & Visualization)")

# 下載資料集
DATA_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
DATA_PATH = "datasets/sms_spam_no_header.csv"
os.makedirs("datasets", exist_ok=True)
if not os.path.exists(DATA_PATH):
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

df = pd.read_csv(DATA_PATH, encoding='latin-1', header=None, names=['label', 'text'])
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# 分割資料
train_size = st.sidebar.slider('Train/Test Split', 0.5, 0.95, 0.8, 0.01)
df_train, df_test = train_test_split(df, test_size=1-train_size, random_state=42, stratify=df['label_num'])

# 向量化
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(df_train['text'])
X_test = vectorizer.transform(df_test['text'])
y_train = df_train['label_num']
y_test = df_test['label_num']

# 訓練模型
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.decision_function(X_test)

# --- Data Overview ---
st.header("1. Data Overview")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='label', ax=ax)
    st.pyplot(fig)
with col2:
    st.subheader("Message Length Distribution")
    df['length'] = df['text'].apply(len)
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='length', hue='label', bins=40, kde=True, element='step', ax=ax)
    st.pyplot(fig)

# --- Top Tokens by Class ---
st.header("2. Top Tokens by Class")
N = st.slider('Top N tokens', 5, 50, 20, 1)
from collections import Counter
col1, col2 = st.columns(2)
for label, col in zip(['ham', 'spam'], [col1, col2]):
    tokens = ' '.join(df[df['label'] == label]['text']).lower().split()
    counter = Counter(tokens)
    most_common = counter.most_common(N)
    tokens_, counts_ = zip(*most_common)
    fig, ax = plt.subplots(figsize=(5, N//2))
    sns.barplot(x=list(counts_), y=list(tokens_), ax=ax, orient='h')
    ax.set_title(f'Top {N} Tokens: {label}')
    col.pyplot(fig)

# --- Model Performance ---
st.header("3. Model Performance (Test)")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
with col2:
    st.subheader("ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    st.pyplot(fig)

st.subheader("Threshold Sweep (Precision/Recall/F1)")
thresholds = np.linspace(min(y_pred_proba), max(y_pred_proba), 100)
precisions, recalls, f1s = [], [], []
for t in thresholds:
    preds = (y_pred_proba > t).astype(int)
    precisions.append(precision_score(y_test, preds))
    recalls.append(recall_score(y_test, preds))
    f1s.append(f1_score(y_test, preds))
fig, ax = plt.subplots()
ax.plot(thresholds, precisions, label='Precision')
ax.plot(thresholds, recalls, label='Recall')
ax.plot(thresholds, f1s, label='F1-score')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Threshold Sweep: Precision/Recall/F1')
ax.legend()
st.pyplot(fig)

# --- Live Inference ---
st.header("4. Live Inference")
st.write("可手動輸入訊息或點選按鈕產生 spam/ham 範例，並即時顯示預測與機率")
col1, col2 = st.columns([3,1])
with col2:
    if st.button('Use spam example'):
        example = df[df['label']=='spam']['text'].sample(1).values[0]
        st.session_state['input_text'] = example
    if st.button('Use ham example'):
        example = df[df['label']=='ham']['text'].sample(1).values[0]
        st.session_state['input_text'] = example
with col1:
    input_text = st.text_area('請輸入訊息', value=st.session_state.get('input_text', ''), height=80)
    if st.button('Predict'):
        if input_text.strip():
            X_vec = vectorizer.transform([input_text])
            pred = clf.predict(X_vec)[0]
            proba = clf.decision_function(X_vec)[0]
            proba_sigmoid = 1 / (1 + np.exp(-proba))
            st.write(f'預測結果: **{"spam" if pred==1 else "ham"}**')
            st.write(f'預測機率 (sigmoid): **{proba_sigmoid:.3f}**')
        else:
            st.warning('請輸入訊息')
