import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_model():
    with open('football_model2.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_nationality = data["le_nationality"]
le_position = data["le_position"]

def show_predict_page():
    st.title("Football Player Market Value Prediction")

    st.write("""### We need some information to predict the market value""")

    nationalities = (
        'Argentina', 'Portugal', 'Brazil', 'Poland', 'Uruguay', 'Germany',
       'France', 'Norway', 'Belgium', 'Spain', 'Croatia',
       'England', 'Senegal', 'Netherlands', 'Wales', 'Sweden', 'Italy',
       'Korea Republic', 'Chile', 'Nigeria', 'Denmark', 'Colombia',
       'Scotland', 'Switzerland', 'Austria', 'Serbia', 'Turkey', 'Ghana',
       'Mexico', 'Japan', 'Russia', 'United States',
       'Republic of Ireland', 'Venezuela', 'Paraguay', 'China PR',
       'Australia', 'Romania', 'Saudi Arabia', 'Other'
    )
    
    positions = (
        'RW', 'CF', 'LW', 'ST', 'GK', 'CAM', 'CM', 'CB', 'CDM', 'RM', 'LM',
       'LB', 'RB', 'RWB', 'LWB'
    )

    nationality = st.selectbox("Nationality", nationalities)
    position = st.selectbox("Primary Position", positions)

    age = st.slider("Age", 18, 40, 25)
    pace = st.slider("Pace", 0, 100, 60)
    shooting = st.slider("Shooting", 0, 100, 60)  
    passing = st.slider("Passing", 0, 100, 60)
    dribbling = st.slider("Dribbling", 0, 100, 60)
    defending = st.slider("Defending", 0, 100, 60)
    physic = st.slider("Physical", 0, 100, 60)
    goalkeeping = st.slider("Goalkeeping", 0, 100, 60)
    overall = st.slider("Overall", 0, 100, 60)

    ok = st.button("Predict Market Value")
    if ok:
        X = np.array([[nationality, position, age, pace, shooting, passing, dribbling, defending, physic, goalkeeping, overall]])
        
        # Apply the label encoders to the categorical values
        X[:, 0] = le_nationality.transform(X[:, 0])  # Transform nationality
        X[:, 1] = le_position.transform(X[:, 1])  # Transform primary position
        X = X.astype(float)

        market_value = regressor.predict(X)
        st.subheader(f"The predicted market value is €{market_value[0]:,.2f}")
