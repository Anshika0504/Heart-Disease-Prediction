# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 08:47:14 2025

@author: Anshika Agarwal
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# function for prediction
def heart_prediction(input_data):
    input_array = np.asarray(input_data, dtype=float)
    input_reshaped = input_array.reshape(1, -1)
    prediction = loaded_model.predict(input_reshaped)

    if prediction[0] == 1:
        return "Heart disease detected"
    else:
        return "Healthy heart"


def main():
    st.title("Heart Disease Prediction Web App")

    # Taking input from user
    Age = st.number_input("Age", min_value=1, max_value=120, step=1)
    Sex = st.number_input("Sex (1=Male, 0=Female)", min_value=0, max_value=1)
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol")
    fbs = st.number_input("Fasting Blood Sugar")
    restecg = st.number_input("Resting ECG")
    thalach = st.number_input("Max heart rate achieved")
    exang = st.number_input("Exercise induced angina (1=yes, 0=no)", min_value=0, max_value=1)
    oldpeak = st.number_input("Oldpeak")
    slope = st.number_input("Slope (0-2)", min_value=0, max_value=2)
    ca = st.number_input("CA (0-4)", min_value=0, max_value=4)
    thal = st.number_input("Thal (1-3)", min_value=1, max_value=3)

    diagnosis = ""

    if st.button("Heart Disease Prediction"):
        diagnosis = heart_prediction([Age, Sex, cp, trestbps, chol, fbs, restecg,
                                      thalach, exang, oldpeak, slope, ca, thal])

    st.success(diagnosis)


if __name__ == '__main__':
    main()

