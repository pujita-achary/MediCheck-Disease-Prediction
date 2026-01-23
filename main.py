import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set Page Configuration
st.set_page_config(page_title="MediCheck AI", layout="wide", page_icon="🧑‍⚕️")

# Load the saved models
try:
    diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
    heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
    parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))
except:
    st.error("Models not found. Please run 'train_model.py' first.")

# Sidebar Navigation
with st.sidebar:
    selected = option_menu("MediCheck AI System",
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# --- DIABETES PAGE ---
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction using SVM')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    with col2:
        Age = st.text_input('Age of the Person')

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        prediction = diabetes_model.predict([user_input])
        if prediction[0] == 1:
            st.error('The person is diabetic')
        else:
            st.success('The person is not diabetic')

# --- HEART DISEASE PAGE ---
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using Logistic Regression')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (1=Male, 0=Female)')
    with col3:
        cp = st.text_input('Chest Pain types (0-3)')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results (0,1,2)')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1=Yes, 0=No)')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment (0-2)')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy (0-3)')
    with col1:
        thal = st.text_input('Thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        prediction = heart_disease_model.predict([user_input])
        if prediction[0] == 1:
            st.error('The person has heart disease')
        else:
            st.success('The person does not have heart disease')

# --- PARKINSONS PAGE ---
if selected == "Parkinsons Prediction":
    st.title("🧠 Parkinsons Prediction using Random Forest")
    st.write("Enter the values from the voice recording data:")
    
    # Parkinsons has many inputs (22), creating 5 columns for better layout
    cols = st.columns(5)
    # List of input feature names as per dataset
    feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 
                     'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 
                     'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 
                     'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    
    user_inputs = []
    for i, feature in enumerate(feature_names):
        with cols[i % 5]:
            val = st.text_input(feature, "0")
            user_inputs.append(val)

    if st.button("Parkinson's Test Result"):
        user_input = [float(x) for x in user_inputs]
        prediction = parkinsons_model.predict([user_input])
        if prediction[0] == 1:
            st.error("The person has Parkinson's disease")
        else:
            st.success("The person does not have Parkinson's disease")