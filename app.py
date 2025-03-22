import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 🔹 **Streamlit App Title**
st.title("🏋️ Personal Fitness Tracker")

# 🔹 **User Inputs for BMI Calculation**
st.sidebar.header("Enter Your Details")
weight = st.sidebar.number_input("Enter your weight (kg)", min_value=1.0, format="%.2f")
height = st.sidebar.number_input("Enter your height (m)", min_value=0.1, format="%.2f")
age = st.sidebar.slider("Select your age", 18, 60, 25)
activity_level = st.sidebar.selectbox("Select your activity level", ["Low", "Medium", "High"])

# 🔹 **BMI Calculation**
if weight > 0 and height > 0:
    bmi = weight / (height ** 2)
    st.write(f"📌 Your BMI is: **{bmi:.2f}**")

    # 🔹 **Histogram for BMI**
    fig, ax = plt.subplots()
    sns.histplot([bmi], kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.write("⚠️ Please enter valid weight and height.")

# 🔹 **Generate Sample Fitness Data**
st.subheader("📊 Generated Sample Data")

data = pd.DataFrame({
    "age": np.random.randint(18, 60, 100),
    "weight": np.random.randint(50, 100, 100),
    "height": np.random.uniform(1.5, 2.0, 100),
    "active": np.random.choice([0, 1], 100),
    "fitness_level": np.random.choice(["low", "medium", "high"], 100)
})

st.write("🔹 **Sample Data (First 5 Rows)**:", data.head())

# 🔹 **Convert Categorical Data to Numeric**
label_encoder = LabelEncoder()
data["fitness_level"] = label_encoder.fit_transform(data["fitness_level"])

# 🔹 **Split Data for Training**
X = data[["age", "weight", "height", "active"]]
y = data["fitness_level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 **Train Machine Learning Models**
models = {
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)

# 🔹 **User Input for Prediction**
st.subheader("📈 Predict Your Fitness Level")

if st.button("Predict"):
    user_data = pd.DataFrame({
        "age": [age],
        "weight": [weight],
        "height": [height],
        "active": [1 if activity_level == "High" else 0]
    })
    
    predictions = {name: model.predict(user_data)[0] for name, model in models.items()}
    
    st.write(f"**SVM Prediction:** {label_encoder.inverse_transform([predictions['SVM']])[0]}")
    st.write(f"**Logistic Regression Prediction:** {label_encoder.inverse_transform([predictions['Logistic Regression']])[0]}")
    st.write(f"**Random Forest Prediction:** {label_encoder.inverse_transform([predictions['Random Forest']])[0]}")

# 🔹 **Final Message**
st.write("✅ **Personal Fitness Tracker Ready!** 🚀")
