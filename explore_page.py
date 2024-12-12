import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the football dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/football.csv", encoding='ISO-8859-1')
    df = df[["Age", "Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physic", "Goalkeeping", "Overall", "Value_euro", "Nationality", "Primary_Position"]]
    return df

df = load_data()

def show_explore_page():
    st.title("Explore Football Player Stats")

    # Show a sample of the data
    st.write(df.head())

    # # Distribution of Market Values
    # st.write("### Distribution of Market Values")
    # df['Value_euro'].hist(bins=50, figsize=(10, 6))
    # st.pyplot()

    # Compare Mean Market Value by Nationality
    st.write("### Mean Market Value by Nationality")
    nationality_avg_value = df.groupby("Nationality")["Value_euro"].mean().sort_values(ascending=False)
    st.bar_chart(nationality_avg_value)

    # Compare Mean Market Value by Primary Position
    st.write("### Mean Market Value by Primary Position")
    position_avg_value = df.groupby("Primary_Position")["Value_euro"].mean().sort_values(ascending=False)
    st.bar_chart(position_avg_value)

    # # Correlation between key attributes and Market Value
    # st.write("### Correlation between Attributes and Market Value")
    # corr_matrix = df.corr()
    # st.write(corr_matrix['Value_euro'].sort_values(ascending=False))
