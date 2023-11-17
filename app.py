import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to take user input and make predictions
def predict_diabetes(data):
    prediction = model.predict(data)
    return prediction

# Streamlit app
def main():
    st.title('Diabetes Predictor')

    # User input
    st.header('Enter Patient Information')
    pregnancies = st.slider('Pregnancies', 0, 20, 1)
    glucose = st.slider('Glucose', 0, 200, 100)
    blood_pressure = st.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.slider('Skin Thickness', 0, 99, 20)
    insulin = st.slider('Insulin', 0, 846, 79)
    bmi = st.slider('BMI', 0.0, 67.1, 25.0)
    pdf = st.slider('DiabetesPedigreeFunction',0.0,0.875,0.975)
    age = st.slider('Age', 21, 100, 25)

    # Make predictions
    if st.button('Predict'):
        # Prepare input data
        input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction' : pdf,
            'Age': age
        }

        # Convert input data to dataframe
        input_df = pd.DataFrame([input_data])

        # Make predictions
        prediction = predict_diabetes(input_df)
        st.write('Prediction:', prediction)

if __name__ == "__main__":
    main()
