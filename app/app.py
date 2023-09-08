import joblib
import random
import numpy as np
import pandas as pd
import streamlit as st
from preprocessing import fill_missing_with_mean, fill_color_mapping, fill_source_with_mode
from preprocessing import delete_non_important_columns, create_new_columns, scale_features

st.set_page_config(page_title = 'Water Quality Prediction', page_icon = '💧', layout = 'wide')

model = joblib.load('model/model.joblib')
pipeline = joblib.load('model/preprocessing_pipeline.joblib')
sample = pd.read_csv('data/sample.csv')

st.title('Water Quality Prediction')

# data = st.file_uploader("Upload Data CSV File", type=["csv"])

# if data is not None:
#     df = pd.read_csv(data)

#     cleaned_df = pipeline.transform(df)
#     prediction = model.predict(cleaned_df)

#     st.write('prediction - ', [prediction])


number = random.randint(0, 4)
col1, col2, col3 = st.columns(3)

# Add input fields to the first column
with col1:
    pH = st.number_input("pH", min_value = 0.0, max_value = 14.0, value = sample.iloc[number]['pH'].round(2), step = 0.5)
    Iron = st.number_input("Iron", min_value = 0.0, max_value = 20.0, value = sample.iloc[number]['Iron'].round(2), step = 1.0)
    Nitrate = st.number_input("Nitrate", min_value = 0.0, max_value = 100.0, value = sample.iloc[number]['Nitrate'].round(2), step = 5.0)
    Chloride = st.number_input("Chloride", min_value = 0.0, max_value = 1500.0, value = sample.iloc[number]['Chloride'].round(2), step = 50.0)
    Lead = st.number_input("Lead", min_value = 0.0, max_value = 5.0, value = sample.iloc[number]['Lead'].round(2), step = 0.2)
    Zinc = st.number_input("Zinc", min_value = 0.0, max_value = 30.0, value = sample.iloc[number]['Zinc'].round(2), step = 1.0)
    color_options = ['Colorless', 'Faint Yellow', 'Light Yellow', 'Near Colorless', 'Yellow']
    Color = st.selectbox('Water Color', color_options)

with col2:
    Turbidity = st.number_input("Turbidity", min_value = 0.0, max_value = 25.0, value = sample.iloc[number]['Turbidity'].round(2), step = 2.0)
    Fluoride = st.number_input("Fluoride", min_value = 0.0, max_value = 15.0, value = sample.iloc[number]['Fluoride'].round(2), step = 0.5)
    Copper = st.number_input("Copper", min_value = 0.0, max_value = 15.0, value = sample.iloc[number]['Copper'].round(2), step = 0.5)
    Odor = st.number_input("Odor", min_value = 0.0, max_value = 5.0, value = sample.iloc[number]['Odor'].round(2), step = 0.5)
    Sulfate = st.number_input("Sulfate", min_value = 0.0, max_value = 1500.0, value = sample.iloc[number]['Sulfate'].round(2), step = 20.0)
    Conductivity = st.number_input("Conductivity", min_value = 0.0, max_value = 2000.0, value = sample.iloc[number]['Conductivity'].round(2), step = 100.0)

with col3:   
    Chlorine = st.number_input("Chlorine", min_value = 0.0, max_value = 15.0, value = sample.iloc[number]['Chlorine'].round(2), step = 1.2)
    Manganese = st.number_input("Manganese", min_value = 0.0, max_value = 25.0, value = sample.iloc[number]['Manganese'].round(2), step = 1.5)
    total_dissolved_solids = st.number_input("Total Dissolved Solids", min_value = 0.0, max_value = 600.0, value = sample.iloc[number]['Total Dissolved Solids'].round(2), step = 25.0)
    source_options = ['Lake', 'River', 'Ground', 'Spring', 'Stream', 'Aquifer', 'Reservoir', 'Well']
    Source = st.selectbox('Water Source', source_options)
    water_temperature = st.number_input("Water Temperature", min_value = 0.0, max_value = 300.0, value = sample.iloc[number]['Water Temperature'].round(2), step = 15.0)
    air_temperature = st.number_input("Air Temperature", min_value = 0.0, max_value = 150.0, value = sample.iloc[number]['Air Temperature'].round(2), step = 15.0)

# Create a pandas dataframe from the user input data
data = {'pH': [pH], 'Iron': [Iron], 'Nitrate': [Nitrate], 'Chloride': [Chloride], 'Lead': [Lead], 'Zinc': [Zinc], 'Color': [Color], 'Turbidity': [Turbidity], 'Fluoride': [Fluoride], 'Copper': [Copper], 'Odor': [Odor], 'Sulfate': [Sulfate], 'Conductivity': [Conductivity], 'Chlorine': [Chlorine], 'Manganese': [Manganese], 'Total Dissolved Solids': [total_dissolved_solids], 'Source': [Source], 'Water Temperature': [water_temperature], 'Air Temperature': [air_temperature]}
df_user = pd.DataFrame(data)

cleaned_df_user = pipeline.transform(df_user)
prediction_user = model.predict(cleaned_df_user)

st.write('prediction - ', [prediction_user])