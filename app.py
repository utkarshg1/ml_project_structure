import streamlit as st
from src.predict import IrisPredictor

predictor = IrisPredictor()

# Set streamlit app
st.set_page_config("Iris Project")
st.title("Iris Project")
st.subheader("by Utkarsh Gaikwad")

# Take inputs from user
sep_len = st.number_input("Sepal Length : ", min_value=0.00, step=0.01)
sep_wid = st.number_input("Sepal Width : ", min_value=0.00, step=0.01)
pet_len = st.number_input("Petal Length : ", min_value=0.00, step=0.01)
pet_wid = st.number_input("Petal Width : ", min_value=0.00, step=0.01)

# create a predict button
button = st.button("Predict", type="primary")

# If button pressed
if button:
    xnew = predictor.to_dataframe(sep_len, sep_wid, pet_len, pet_wid)
    preds = predictor.predict(xnew)
    probs = predictor.predict_proba(xnew)
    st.subheader(f"Prediction : {preds}")
    st.subheader(f"Probabilities : ")
    st.dataframe(probs)
    st.bar_chart(probs.T)
