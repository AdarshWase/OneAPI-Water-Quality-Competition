import joblib
import random
import numpy as np
import pandas as pd
import streamlit as st
from preprocessing import fill_missing_with_mean, fill_color_mapping, fill_source_with_mode
from preprocessing import delete_non_important_columns, create_new_columns, scale_features

model = joblib.load('model/model.joblib')
pipeline = joblib.load('model/preprocessing_pipeline.joblib')
sample = pd.read_csv('data/sample.csv')

from markdownlit import mdlit
from streamlit_lottie import st_lottie

st.set_page_config(page_title = 'Water Quality Prediction', page_icon = '💧', layout = 'wide')

# Title
mdlit('<p style="font-size: 53px; text-align: center;"><b>Water Quality Prediction with OneAPI</b></p>')


# First Block
st.markdown("<hr>", unsafe_allow_html=True)
left, right = st.columns([1, 3])

with left:
    st_lottie("https://lottie.host/9227603b-9350-4b4f-a1ce-a6d1acee81d1/IP5X1a4d4K.json", height = 270, loop = True, quality = 'high')

with right:
    about_1 = "Freshwater, comprising a mere [red]3% of the Earth's total water volume[/red], stands as one of our most crucial yet limited natural resources. It intricately weaves into the fabric of our daily lives, serving as the foundation for [orange]drinking[/orange], [blue]recreation[/blue], [red]agriculture[/red], and [violet]various industrial processes[/violet]. Beyond human consumption, it drives the production of food, electricity, and numerous everyday products. The significance of a secure and hygienic water supply extends beyond human well-being, reaching out to the survival of surrounding ecosystems. These delicate balances are increasingly threatened by factors such as [blue]droughts, pollution[/blue], and the relentless rise in temperatures."
    about_2 = "This project is a vital step towards safeguarding this invaluable resource. By harnessing a comprehensive dataset encompassing parameters like pH levels, metal concentrations, and various other quality indicators, participants will assess the [red]suitability of water for consumption[/red]. The [green]aim[/green] is to provide accessible and actionable predictions about the [blue]water quality[/blue], ensuring that communities have the knowledge they need to make informed decisions about their water sources."
    about_3 = "[red]Join us[/red] in this endeavor to preserve and protect our freshwater reservoirs, as we work towards a [green]sustainable future[/green] for generations to come."
    mdlit(about_1)
    mdlit(about_2)
    mdlit(about_3)

st.markdown("<hr>", unsafe_allow_html=True)

# Second Block

mdlit('<p style="font-size: 33px; text-align: center;"><b>Predict the Water Quality</b></p>')
st.write(" ")

tab1, tab2 = st.tabs(['Upload Data', 'Enter Data Manually'])

with tab1:
    mdlit('<p style="font-size: 23px; text-align: center;"><b>Upload your data here</b></p>')
    data = st.file_uploader("Upload Data CSV File for prediction", type=["csv"])

    predict_button_1 = st.button('Make Prediction')

    if predict_button_1:
        if data is not None:
            df = pd.read_csv(data)

            cleaned_df = pipeline.transform(df)
            prediction = model.predict(cleaned_df)

            new_df_batched = df.insert(loc = 0, column = 'Predictions', value = prediction)
            styled_df = df.style.apply(lambda x: ['background-color: #ADD8E6' if i == 0 else '' for i in range(len(x))], axis=1)

            p_col, d_col = st.columns([1, 2])
            with p_col:
                st.write('Array of Predictions - ', [prediction])

            with d_col:
                st.dataframe(styled_df)

with tab2:
    mdlit('<p style="font-size: 23px; text-align: center;"><b>Fill your data here</b></p>')

    number = random.randint(0, 4)
    
    # Add input fields to the first column
    with st.form(key='form1'):
        col1, col2, col3 = st.columns(3)
        with col1:
            pH = st.number_input("pH", min_value = 0.0, max_value = 14.0, step = 0.5)
            Iron = st.number_input("Iron", min_value = 0.0, max_value = 20.0, step = 1.0)
            Nitrate = st.number_input("Nitrate", min_value = 0.0, max_value = 100.0, step = 5.0)
            Chloride = st.number_input("Chloride", min_value = 0.0, max_value = 1500.0, step = 50.0)
            Lead = st.number_input("Lead", min_value = 0.0, max_value = 5.0, step = 0.2)
            Zinc = st.number_input("Zinc", min_value = 0.0, max_value = 30.0, step = 1.0)
            color_options = ['Colorless', 'Faint Yellow', 'Light Yellow', 'Near Colorless', 'Yellow']
            Color = st.selectbox('Water Color', color_options)

        with col2:
            Turbidity = st.number_input("Turbidity", min_value = 0.0, max_value = 25.0, step = 2.0)
            Fluoride = st.number_input("Fluoride", min_value = 0.0, max_value = 15.0, step = 0.5)
            Copper = st.number_input("Copper", min_value = 0.0, max_value = 15.0, step = 0.5)
            Odor = st.number_input("Odor", min_value = 0.0, max_value = 5.0, step = 0.5)
            Sulfate = st.number_input("Sulfate", min_value = 0.0, max_value = 1500.0, step = 20.0)
            Conductivity = st.number_input("Conductivity", min_value = 0.0, max_value = 2000.0, step = 100.0)

        with col3:   
            Chlorine = st.number_input("Chlorine", min_value = 0.0, max_value = 15.0, step = 1.2)
            Manganese = st.number_input("Manganese", min_value = 0.0, max_value = 25.0, step = 1.5)
            total_dissolved_solids = st.number_input("Total Dissolved Solids", min_value = 0.0, max_value = 600.0, step = 25.0)
            source_options = ['Lake', 'River', 'Ground', 'Spring', 'Stream', 'Aquifer', 'Reservoir', 'Well']
            Source = st.selectbox('Water Source', source_options)
            water_temperature = st.number_input("Water Temperature", min_value = 0.0, max_value = 300.0, step = 15.0)
            air_temperature = st.number_input("Air Temperature", min_value = 0.0, max_value = 150.0, step = 15.0)

        predict_button_2 = st.form_submit_button('Make Prediction')

        if predict_button_2:
            data = {'pH': [pH], 'Iron': [Iron], 'Nitrate': [Nitrate], 'Chloride': [Chloride], 'Lead': [Lead], 'Zinc': [Zinc], 'Color': [Color], 'Turbidity': [Turbidity], 'Fluoride': [Fluoride], 'Copper': [Copper], 'Odor': [Odor], 'Sulfate': [Sulfate], 'Conductivity': [Conductivity], 'Chlorine': [Chlorine], 'Manganese': [Manganese], 'Total Dissolved Solids': [total_dissolved_solids], 'Source': [Source], 'Water Temperature': [water_temperature], 'Air Temperature': [air_temperature]}
            df_user = pd.DataFrame(data)

            cleaned_df_user = pipeline.transform(df_user)
            prediction_user = model.predict(cleaned_df_user)

            st.write('Prediction - ', [prediction_user])

st.markdown("<hr>", unsafe_allow_html=True)

# Third Block

mdlit('<p style="font-size: 33px; text-align: center;"><b>Tech Stack and Strategies</b></p>')
text1 = "Our project harnesses a powerful suite of technologies tailored to ensure accuracy, efficiency, and reliability in every facet of water quality analysis. Leveraging the Intel Optimized Modin and Pandas libraries, we enable swift and seamless data processing, providing a robust foundation for subsequent analytical tasks. Complementing this, the Intel AI Analytics Toolkit empowers us with a comprehensive set of tools for in-depth data exploration and insightful visualizations."
text2 = "To track experiments and model versions with precision, we rely on the GitHub MLflow integration and Data Version Control (DVC), ensuring transparency and reproducibility in our research endeavors. Fine-tuning our models for optimal performance is facilitated by the Intel Optimized XGBoost and Optuna, optimizing hyperparameters to achieve superior accuracy. Additionally, we employ the Intel Optimized scikit-learn library to enhance the efficiency of our machine learning algorithms."
text3 = "For seamless deployment, we utilize GitHub Actions in tandem with Streamlit, streamlining the transition from development to production. To further enhance model efficiency during inference, we've converted our XGBoost model into a daal4py model, capitalizing on its accelerated performance."
mdlit(text1)
mdlit(text2)
mdlit(text3)

one, two = st.columns([1, 2])
with one:
    st.image("image/mind.png")

with two:
    mdlit('<p style="font-size: 23px; text-align: center;"><b>Strategies</b></p>')

    features = """<ul>
    <li><b>Analysis</b>: The analysis phase was conducted on the Intel Optimized DevCloud Python platform. This environment facilitated an in-depth exploration of the dataset, revealing a notable prevalence of outliers and missing values. Furthermore, I employed powerful visualization tools like Seaborn and Matplotlib to gain comprehensive insights into the data's characteristics.'</li>
    <li><b>Feature Engineering</b>: In this crucial phase, I meticulously curated the dataset to enhance its predictive power. Here are the key steps: Missing Values: Addressing missing data is paramount. Employing a combination of mean and median imputations, alongside count mapping, I meticulously handled most of the gaps in the dataset. Outliers: Recognizing their potential value, I refrained from outright removal of outliers. Instead, I harnessed their presence to construct new features, bolstering the robustness of predictions. Feature Creations: Leveraging a binning approach, I ingeniously derived new features from existing ones and their associated outliers, introducing a layer of depth to the dataset.</li>
    <li><b>Sentiment analysis model</b>: A robust validation strategy is crucial for accurately assessing model performance. To this end, I adopted a multifaceted approach. By generating multiple validation datasets, I ensured a realistic evaluation of my model's predictive capabilities. This comprehensive validation framework provides a reliable indicator of the model's performance across various scenarios.</li>
    </ul> 
    """
    mdlit(features)