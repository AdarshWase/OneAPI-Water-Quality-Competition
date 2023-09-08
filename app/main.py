import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
)
# Title of the app
st.title("Drinkable Water Predictor")

# Subheader for uploading the data
st.subheader("Upload Data")

# Option to upload a data csv file
data = st.file_uploader("Upload Data CSV File", type=["csv"])

st.subheader('Upload Data Manually')

col1, col2, col3, col4 = st.columns(4)

# Add input fields to the first column
with col1:
    ph = st.number_input("pH")
    iron = st.number_input("Iron")
    nitrate = st.number_input("Nitrate")
    chloride = st.number_input("Chloride")
    lead = st.number_input("Lead")
    zinc = st.number_input("Zinc")

with col2:
    color = st.number_input("Color")
    turbidity = st.number_input("Turbidity")
    fluoride = st.number_input("Fluoride")
    copper = st.number_input("Copper")
    odor = st.number_input("Odor")

# Add input fields to the second column
with col3:
    sulfate = st.number_input("Sulfate")
    conductivity = st.number_input("Conductivity")
    chlorine = st.number_input("Chlorine")
    manganese = st.number_input("Manganese")
    total_dissolved_solids = st.number_input("Total Dissolved Solids")

with col4:    
    source = st.number_input("Source")
    water_temperature = st.number_input("Water Temperature")
    air_temperature = st.number_input("Air Temperature")
    iron_bin = st.number_input("Iron Bin")
    nitrate_bin = st.number_input("Nitrate Bin")
    copper_bin = st.number_input("Copper Bin")

# Button to predict the drinkability of water
if st.button("Predict"):
    # If data is uploaded
    if data is not None:
        # Check if it is a single data point or batched data
        data_df = pd.read_csv(data)
        if len(data_df) == 1:
            prediction = predict(data)
            st.write("The water is drinkable" if prediction[0] == 1 else "The water is not drinkable")
        else:
            # Predict the drinkability of water for the batched data
            predictions = predict(data)
            # Return the predictions as a csv file
            st.download_button("Download Predictions", data=predictions.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
    # If data is input manually
    else:
        # Predict the drinkability of water for the input data
        prediction = predict_manual_input(ph, iron, nitrate, chloride, lead, zinc, color, turbidity, fluoride, copper, odor, sulfate, conductivity, chlorine, manganese, total_dissolved_solids, source, water_temperature, air_temperature, iron_bin, nitrate_bin, copper_bin)
        st.write("The water is drinkable" if prediction == 1 else "The water is not drinkable")