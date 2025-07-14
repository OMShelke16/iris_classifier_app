import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Classifier")
st.write("Enter the flower measurements:")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Predict
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"🌼 Predicted Species: {species[prediction]}")
