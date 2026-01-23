import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

# --- 1. Train Diabetes Model (SVM) ---
try:
    diabetes_data = pd.read_csv('diabetes.csv')
    X = diabetes_data.drop(columns='Outcome', axis=1)
    Y = diabetes_data['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    # Using SVM (Support Vector Machine)
    model_diabetes = SVC(kernel='linear')
    model_diabetes.fit(X_train, Y_train)
    
    pickle.dump(model_diabetes, open('diabetes_model.sav', 'wb'))
    print("✅ Diabetes Model Trained & Saved!")
except Exception as e:
    print(f"❌ Error training Diabetes model: {e}")

# --- 2. Train Heart Disease Model (Logistic Regression) ---
try:
    heart_data = pd.read_csv('heart.csv')
    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    model_heart = LogisticRegression(max_iter=1000)
    model_heart.fit(X_train, Y_train)
    
    pickle.dump(model_heart, open('heart_disease_model.sav', 'wb'))
    print("✅ Heart Disease Model Trained & Saved!")
except Exception as e:
    print(f"❌ Error training Heart model: {e}")

# --- 3. Train Parkinson's Model (Random Forest) ---
try:
    parkinsons_data = pd.read_csv('parkinsons.csv')
    X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
    Y = parkinsons_data['status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    
    model_parkinsons = RandomForestClassifier()
    model_parkinsons.fit(X_train, Y_train)
    
    pickle.dump(model_parkinsons, open('parkinsons_model.sav', 'wb'))
    print("✅ Parkinson's Model Trained & Saved!")
except Exception as e:
    print(f"❌ Error training Parkinson's model: {e}")