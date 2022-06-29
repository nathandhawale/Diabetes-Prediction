#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:03:34 2022

@author: nathandhawale
"""

import numpy as np
import pickle
import streamlit as st

#Loading the saved model
loaded_model = pickle.load(open('/Users/nathandhawale/Documents/GitHub/Diabetes-Prediction/trained_diabetes_model.sav',
                                'rb'))

#Creating a function for prediction
def diabetes_prediction(input_data):
    
    input_data = (10,168,74,0,0,38,0.537,34)

    #change to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is Non-Diabetic'
    else:
        return "The person is Diabetic"
    

def main():
    
    #setting title
    st.title("Diabetes Prediction Web App")
    
    #getting input data from the user
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("Body Mass Index")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")
    
    
    #code for Prediction
    
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, 
                                         Glucose, 
                                         BloodPressure, 
                                         SkinThickness,
                                         Insulin,
                                         BMI,
                                         DiabetesPedigreeFunction,
                                         Age])
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    