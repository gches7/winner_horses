import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
horses = pd.read_csv("horse_winners_example.csv", sep=',')

# Define columns
cat_features = ['pedigree', 'breed', 'trainer']
num_features = ['favorable_proportions', 'training_frequency', 'age', 'has_disease',
                'vet_care', 'special_feed', 'supplementation']
features = num_features + cat_features

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(), cat_features)
])

x = horses[features]
y = horses['win_probability']

x_proc = preprocessor.fit_transform(x)

# Data for test/training
x_train, x_test, y_train, y_test = train_test_split(x_proc, y, test_size=0.4, random_state=42)

# Target to binary for classification
default_threshold = 0.65
y_train_bin = (y_train >= default_threshold).astype(int)

# Train model
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train_bin)

# Streamlit interface
st.title("Winner Horses Prediction 🐎")

# Slider for threshold
threshold = st.slider("Choose threshold to define a winner", min_value=0.0, max_value=1.0, value=0.65, step=0.01)

st.subheader("Enter horse information")

# Binary input (Yes/No)
def yes_no_input(label):
    return 1 if st.selectbox(label, ["Yes", "No"]) == "Yes" else 0

# Boolean inputs
favorable_proportions = yes_no_input("Favorable proportions?")
disease_history = yes_no_input("Disease history?")
under_vet_care = yes_no_input("Under vet care?")
special_feed = yes_no_input("Special feed?")
supplementation = yes_no_input("Receives supplementation?")

# Numeric inputs
training_frequency = st.number_input("Training frequency (times/week)", min_value=0.0)
age = st.number_input("Horse age", min_value=0.0)

# Dropdown categorical inputs
pedigree = st.selectbox("Pedigree", sorted(horses['pedigree'].unique()))
breed = st.selectbox("Breed", sorted(horses['breed'].unique()))
trainer = st.selectbox("Trainer", sorted(horses['trainer'].unique()))

# Predict button
if st.button("Predict winner"):
    input_dict = {
        'favorable_proportions': favorable_proportions,
        'training_frequency': training_frequency,
        'age': age,
        'has_disease': disease_history,
        'vet_care': under_vet_care,
        'special_feed': special_feed,
        'supplementation': supplementation,
        'pedigree': pedigree,
        'breed': breed,
        'trainer': trainer
    }

    input_df = pd.DataFrame([input_dict])
    input_proc = preprocessor.transform(input_df)
    prob = log_reg.predict_proba(input_proc)[0][1]
    winner = "Yes 🏆" if prob >= threshold else "No ❌"

    st.markdown(f"### Winner probability: **{prob:.2%}**")
    st.markdown(f"### Winner horse? **{winner}**")
