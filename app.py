import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


with open("gs_logistic_regression_model_1.pkl", "rb") as file:
    model = pickle.load(file)


st.title("❤️ Heart Disease Prediction App")

st.markdown(
    """
    This application predicts whether a patient is likely to have **heart disease**
    based on clinical parameters.

    **Model Used:** Logistic Regression  
    **Accuracy:** ~89%
    """
)

st.divider()


st.subheader("🩺 Enter Patient Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type",
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic",
            ],
        )
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120
        )
        chol = st.number_input(
            "Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=240
        )

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox(
            "Resting ECG Results", ["Normal", "ST-T abnormality", "LV hypertrophy"]
        )
        thalach = st.number_input(
            "Maximum Heart Rate Achieved", min_value=70, max_value=210, value=150
        )
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input(
            "ST Depression (Oldpeak)", min_value=0.0, max_value=6.5, value=1.0
        )

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"]
    )
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    submit = st.form_submit_button("🔍 Predict")


if submit:
    
    sex = 1 if sex == "Male" else 0

    cp_mapping = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3,
    }
    cp = cp_mapping[cp]

    fbs = 1 if fbs == "Yes" else 0

    restecg_mapping = {
        "Normal": 0,
        "ST-T abnormality": 1,
        "LV hypertrophy": 2,
    }
    restecg = restecg_mapping[restecg]

    exang = 1 if exang == "Yes" else 0

    slope_mapping = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2,
    }
    slope = slope_mapping[slope]

    thal_mapping = {
        "Normal": 2,
        "Fixed Defect": 3,
        "Reversible Defect": 1,
    }
    thal = thal_mapping[thal]

    
    input_data = np.array(
        [
            [
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
            ]
        ]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.error(
            f"⚠️ **High Risk of Heart Disease**\n\n"
            f"Prediction Confidence: **{probability*100:.2f}%**"
        )
    else:
        st.success(
            f"✅ **Low Risk of Heart Disease**\n\n"
            f"Prediction Confidence: **{(1-probability)*100:.2f}%**"
        )


st.divider()
st.caption(
    "📌 Educational project | Logistic Regression model | Not a medical diagnosis"
)
