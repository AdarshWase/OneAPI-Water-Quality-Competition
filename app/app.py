import joblib
import dvc.api
import pandas as pd
import streamlit as st
from preprocessing import fill_missing_with_mean, fill_color_mapping, fill_source_with_mode
from preprocessing import delete_non_important_columns, create_new_columns, scale_features

model_path = dvc.api.get_url('model.joblib.dvc')

with dvc.api.open(model_path, mode='rb') as f:
    model = joblib.load(f)

st.title('Car Price Prediction')
