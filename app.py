# diabetes_app.py

# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# --- Setup Page ---
st.set_page_config(page_title="Diabetes Predictor", layout="wide")
st.markdown("""
    <style>
        body { background-color: #f0f2f6; color: #1c1c1e; }
        .stButton button {
            background-color: #4CAF50; color: white; border-radius: 10px;
            height: 3em; width: 100%;
        }
        h1, h2, h3, h4, h5 { color: #003366; }
    </style>
""", unsafe_allow_html=True)
st.title("🧠 Diabetes Prediction App")

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, df[col].median())
    return df

df = load_data()

# Prepare features
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

explainer = shap.Explainer(model)

# --- User Input Form ---
st.subheader("📋 Patient Info")
with st.form("form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20)
        Glucose = st.number_input("Glucose", 0, 200)
        BloodPressure = st.number_input("Blood Pressure", 0, 150)
    with col2:
        SkinThickness = st.number_input("Skin Thickness", 0, 100)
        Insulin = st.number_input("Insulin", 0, 900)
        BMI = st.number_input("BMI", 0.0, 70.0)
    with col3:
        DiabetesPedigreeFunction = st.number_input("Pedigree Function", 0.0, 3.0)
        Age = st.number_input("Age", 1, 120)

    predict_btn = st.form_submit_button("🔍 Predict Now")

# --- Prediction ---
if predict_btn:
    input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                                BMI, DiabetesPedigreeFunction, Age]],
                              columns=X.columns)
    scaled_data = scaler.transform(input_data)
    pred = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][pred]

    if pred == 1:
        st.error(f"😔 Likely Diabetic (Confidence: {prob:.2%})")
    else:
        st.success(f"😊 Likely Not Diabetic (Confidence: {prob:.2%})")

    # --- Health Tips ---
    st.subheader("💡 Personalized Health Advice")
    tips = []
    if Glucose > 140:
        tips.append("🔸 High glucose detected. Reduce sugary foods and check with a doctor.")
    if BMI > 30:
        tips.append("🔸 High BMI suggests obesity. Consider regular exercise and a balanced diet.")
    if BloodPressure > 130:
        tips.append("🔸 Elevated blood pressure. Limit salt and manage stress.")
    if Age > 45:
        tips.append("🔸 Older age is a risk factor. Regular screening is important.")
    if tips:
        for tip in tips:
            st.markdown(tip)
    else:
        st.markdown("🌿 Your data looks good! Keep maintaining a healthy lifestyle.")

    # --- Traffic Light Recommendations ---
    st.subheader("🚦 Traffic Light Recommendations")

    def risk_tip(feature, value):
        if feature == "Glucose":
            if value >= 140:
                return ("🔴", "High glucose detected. Reduce sugary foods and monitor levels.")
            elif value >= 100:
                return ("🟡", "Pre-diabetic glucose level. Watch your sugar intake.")
            else:
                return ("🟢", "Glucose is in a healthy range.")

        elif feature == "BMI":
            if value >= 30:
                return ("🔴", "High BMI suggests obesity. Exercise and a healthy diet are recommended.")
            elif value >= 25:
                return ("🟡", "Slightly overweight. Try to maintain a balanced diet and activity.")
            else:
                return ("🟢", "BMI is in a good range.")

        elif feature == "BloodPressure":
            if value >= 90:
                return ("🔴", "High blood pressure. Reduce salt, manage stress, and consult a doctor.")
            elif value >= 80:
                return ("🟡", "Borderline blood pressure. Monitor regularly.")
            else:
                return ("🟢", "Blood pressure is healthy.")

        elif feature == "Age":
            if value >= 60:
                return ("🔴", "Older age increases risk. Get screened regularly.")
            elif value >= 45:
                return ("🟡", "Age is moderately high. Stay active and check health annually.")
            else:
                return ("🟢", "Age-related risk is low.")

        elif feature == "Insulin":
            if value >= 250:
                return ("🔴", "High insulin levels detected. May indicate insulin resistance.")
            elif value >= 150:
                return ("🟡", "Slightly elevated insulin. Consider dietary control.")
            else:
                return ("🟢", "Insulin is within a healthy range.")

        elif feature == "Pregnancies":
            if value >= 6:
                return ("🟡", "Higher pregnancies may slightly increase diabetes risk.")
            else:
                return ("🟢", "Pregnancy count within typical range.")

        elif feature == "DiabetesPedigreeFunction":
            if value >= 1.0:
                return ("🔴", "High genetic risk. Family history plays a major role.")
            elif value >= 0.5:
                return ("🟡", "Moderate genetic risk. Be mindful of lifestyle.")
            else:
                return ("🟢", "Low family history-related risk.")

        elif feature == "SkinThickness":
            if value >= 40:
                return ("🟡", "Higher skin thickness may suggest body fat.")
            else:
                return ("🟢", "Skin thickness within normal range.")

        return ("", "")

    features = {
        "Glucose": Glucose,
        "BMI": BMI,
        "BloodPressure": BloodPressure,
        "Age": Age,
        "Insulin": Insulin,
        "Pregnancies": Pregnancies,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "SkinThickness": SkinThickness
    }

    for feat, val in features.items():
        color, message = risk_tip(feat, val)
        if message:
            st.markdown(f"{color} **{feat}:** {message}")

    # --- SHAP Explanation ---
    st.subheader("🔎 SHAP Explanation")
    shap_values = explainer(scaled_data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.waterfall_plot(shap_values[0], show=False)
    st.pyplot(bbox_inches="tight")

    # --- Classification Report ---
    st.subheader("📋 Classification Report")
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred)
    st.text(report)

# --- Feature Distribution ---
st.markdown("---")
st.subheader("📌 Feature Distribution")
selected_feat = st.selectbox("Choose a feature to visualize:", df.columns[:-1])
if selected_feat:
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feat], kde=True, ax=ax, color="teal")
    ax.set_title(f"Distribution of {selected_feat}", fontsize=14)
    ax.set_xlabel(selected_feat, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    st.pyplot(fig)

st.caption("Built with ❤️ using Streamlit, XGBoost, and SHAP")
