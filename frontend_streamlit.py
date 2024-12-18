import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model and preprocessor
model_filename = 'C:/Users/saiha/OneDrive/Desktop/space_traffic_app/trained_model.pkl'
with open(model_filename, 'rb') as f:
    # Do something with the file here, like loading the model

    model_data = pickle.load(f)

weights = model_data['weights']
preprocessor = model_data['preprocessor']

# Streamlit App
st.title("Space Traffic Density Prediction")

# User inputs
st.header("Input Parameters")
location = st.selectbox("Select Location:", ["Orbit LEO", "Orbit GEO", "Orbit MEO", "Other"])  # Update with your dataset's locations
object_type = st.selectbox("Select Object Type:", ["Satellite", "Debris", "Spacecraft", "Other"])  # Update with dataset's object types
peak_time = st.text_input("Enter Peak Time (e.g., 15:00):", value="15:00")

# Predict button
if st.button("Predict Traffic Density"):
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'Location': [location],
            'Object_Type': [object_type],
            'Peak_Time': [peak_time]
        })

        # Preprocess and predict
        transformed_input = preprocessor.transform(input_data).toarray()
        transformed_input = np.c_[np.ones((transformed_input.shape[0], 1)), transformed_input]
        prediction = np.dot(transformed_input, weights)
        
        # Display prediction
        st.success(f"Predicted Traffic Density: {prediction[0][0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")