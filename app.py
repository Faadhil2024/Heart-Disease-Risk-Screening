import joblib
import pandas as pd
import streamlit as st

model = joblib.load("model.joblib")
THRESHOLD = 0.3

st.title("Heart Disease Risk Screening Demo")
st.caption("Educational demo built on the UCI Cleveland data. Not a diagnostic tool.")

SEX = {"Female": 0, "Male": 1}
CP = {"Typical angina": 1, "Atypical angina": 2,
      "Non-anginal pain": 3, "Asymptomatic": 4}
YESNO = {"No": 0, "Yes": 1}
ECG = {"Normal": 0, "ST-T wave abnormality": 1, "LV hypertrophy": 2}
SLOPE = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
THAL = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}

c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", 20, 90, 55)
    sex = SEX[st.selectbox("Sex", list(SEX))]
    cp = CP[st.selectbox("Chest pain type", list(CP))]
    trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
    fbs = YESNO[st.selectbox("Fasting blood sugar > 120 mg/dl", list(YESNO))]
    restecg = ECG[st.selectbox("Resting ECG", list(ECG))]
with c2:
    thalach = st.number_input("Max heart rate achieved", 60, 220, 150)
    exang = YESNO[st.selectbox("Exercise-induced angina", list(YESNO))]
    oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
    slope = SLOPE[st.selectbox("ST slope", list(SLOPE))]
    ca = st.selectbox("Major vessels colored by fluoroscopy", [0, 1, 2, 3])
    thal = THAL[st.selectbox("Thallium stress test", list(THAL))]

if st.button("Predict"):
    row = pd.DataFrame([{
        "age": age, "trestbps": trestbps, "chol": chol, "thalach": thalach,
        "oldpeak": oldpeak, "ca": ca, "sex": sex, "cp": cp, "fbs": fbs,
        "restecg": restecg, "exang": exang, "slope": slope, "thal": thal,
    }]).astype(float)

    prob = model.predict_proba(row)[0, 1]
    st.metric("Estimated risk score", f"{prob:.0%}")

    if prob >= THRESHOLD:
        st.error("Flagged: higher risk. Recommend further clinical evaluation.")
    else:
        st.success("Not flagged at the screening threshold.")

    st.caption(f"Screening threshold is {THRESHOLD:.0%}, chosen to favour catching "
               "sick patients over avoiding false alarms. This is a demo, not medical advice.")