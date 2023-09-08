import joblib
import pandas as pd
import streamlit as st
from preprocessing import fill_missing_with_mean, fill_color_mapping, fill_source_with_mode
from preprocessing import delete_non_important_columns, create_new_columns, scale_features

model = joblib.load('model/model.joblib')
pipeline = joblib.load('model/preprocessing_pipeline.joblib')

st.title('Car Price Prediction')

data = st.file_uploader("Upload Data CSV File", type=["csv"])

if data is not None:
    df = pd.read_csv(data)

    cleaned_df = pipeline.transform(df)
    prediction = model.predict(cleaned_df)

    st.write('prediction - ', [prediction])