import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# Page config
st.set_page_config(page_title="Smart ML System", layout="wide")

st.title("🤖 Smart Machine Learning Prediction System")
st.write("Works for ANY dataset + Special handling for Fraud Detection datasets")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# ----------- SAFE DATA LOADING -----------
@st.cache_data
def load_data(file):
    return pd.read_csv(file, nrows=50000)  # limit rows for performance

# ----------- DEFAULT DATA OPTION -----------
if uploaded_file is None:
    st.info("👆 Upload a dataset to start OR using default sample data")
    try:
        data = pd.read_csv("default.csv", nrows=50000)
    except:
        st.stop()
else:
    # File size warning
    if uploaded_file.size > 100 * 1024 * 1024:
        st.warning("⚠️ Large file detected. Using only first 50,000 rows for performance.")

    data = load_data(uploaded_file)

# ----------- PREVIEW -----------
st.subheader("📄 Dataset Preview")
st.write(data.head())

# ----------- TARGET DETECTION -----------
if "Class" in data.columns:
    target = "Class"
    st.success("🎯 Target column automatically selected: Class (Fraud Detection)")
else:
    target = st.selectbox("🎯 Select Target Column", data.columns)

# ----------- TRAIN BUTTON -----------
if st.button("🚀 Train Model"):

    with st.spinner("⏳ Training model... please wait"):

        df = data.copy()

        # Encode categorical columns
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        X = df.drop(target, axis=1)
        y = df[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Lightweight model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

    st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")

    # ----------- PREDICTIONS -----------
    predictions = model.predict(X_test)

    result_df = X_test.copy()
    result_df["Actual"] = y_test
    result_df["Predicted"] = predictions

    st.subheader("🔍 Predictions")
    st.write(result_df.head())

    # ----------- SUMMARY -----------
    st.subheader("📊 Summary")

    col1, col2 = st.columns(2)
    col1.metric("Total Rows Used", len(data))

    if target == "Class":
        fraud_count = (result_df["Predicted"] == 1).sum()
        col2.metric("Fraud Predictions", fraud_count)

        if fraud_count > 0:
            st.error(f"🚨 ALERT: {fraud_count} Fraud Transactions Detected!")
        else:
            st.success("✅ No Fraudulent Transactions Found")

    # ----------- BAR CHART -----------
    st.subheader("📈 Prediction Distribution")

    fig, ax = plt.subplots()
    result_df["Predicted"].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Prediction Counts")
    st.pyplot(fig, clear_figure=True)

    # ----------- PIE CHART -----------
    if target == "Class":
        st.subheader("🥧 Fraud vs Normal (Pie Chart)")

        fig1, ax1 = plt.subplots()
        counts = result_df["Predicted"].value_counts()

        labels = ["Normal", "Fraud"] if len(counts) == 2 else counts.index

        ax1.pie(counts, labels=labels, autopct='%1.1f%%')
        ax1.set_title("Fraud vs Normal Distribution")

        st.pyplot(fig1)

    # ----------- ACTUAL VS PREDICTED -----------
    st.subheader("📊 Actual vs Predicted Comparison")

    fig2, ax2 = plt.subplots()

    actual_counts = result_df["Actual"].value_counts().sort_index()
    pred_counts = result_df["Predicted"].value_counts().sort_index()

    labels = list(actual_counts.index)

    ax2.bar(np.arange(len(labels)) - 0.2, actual_counts, width=0.4, label='Actual')
    ax2.bar(np.arange(len(labels)) + 0.2, pred_counts, width=0.4, label='Predicted')

    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.legend()

    st.pyplot(fig2)

    # ----------- CONFUSION MATRIX -----------
    st.subheader("📉 Confusion Matrix")

    cm = confusion_matrix(result_df["Actual"], result_df["Predicted"])

    fig3, ax3 = plt.subplots()
    ax3.matshow(cm)

    for (i, j), val in np.ndenumerate(cm):
        ax3.text(j, i, val, ha='center', va='center')

    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    st.pyplot(fig3)

    # ----------- CLASSIFICATION REPORT -----------
    st.subheader("📄 Classification Report")
    report = classification_report(result_df["Actual"], result_df["Predicted"])
    st.text(report)

    # ----------- FEATURE IMPORTANCE -----------
    st.subheader("🔥 Feature Importance")

    importances = model.feature_importances_
    feature_names = X.columns

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig4, ax4 = plt.subplots()
    ax4.barh(feat_df["Feature"], feat_df["Importance"])
    ax4.invert_yaxis()

    st.pyplot(fig4)

    # ----------- FRAUD TABLE -----------
    if target == "Class":
        st.subheader("⚠️ Detected Fraud Transactions")
        fraud_data = result_df[result_df["Predicted"] == 1]
        st.write(fraud_data)

    # ----------- DOWNLOAD -----------
    csv = result_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "📥 Download Prediction Report",
        csv,
        "prediction_report.csv",
        "text/csv"
    )