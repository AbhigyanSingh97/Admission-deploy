import numpy as np
import pickle
import pandas as pd

import streamlit as st


pickle_in = open("model.pkl","rb")
classifier = pickle.load(pickle_in)


def welcome():
    return "Welcome! Go ahead and predict your chance of admition."

def predict_admission(GRE,TOEFL,CGPA):
    prediction = classifier.predict([[GRE,TOEFL,CGPA]])
    return prediction * 100


def main():
    st.title("Abhigyan's model to predict your chance of admission")
    gre = st.text_input("GRE","Type Here")
    toefl = st.text_input("TOEFL","Type Here")
    CGPA = st.text_input("CGPA","Type Here")
    result = []
    if st.button("Predict"):
        result = predict_admission(gre,toefl,CGPA)
    st.success("Your Chance of Admission is {} %.".format(result))
    if st.button("WHAT IS THIS?"):
        st.text("First ever deployment")
        st.text("Happy Learning!!")

if __name__=='__main__':
    main()