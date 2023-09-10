# Water Quality Prediction Project

## Overview

This project aims to predict water quality, a critical aspect of environmental and public health. Leveraging a comprehensive dataset and advanced machine learning techniques, we strive to provide valuable insights into the suitability of water for consumption.

## Table of Contents

- [Tech Stack](#tech-stack)
- [Features](#features)
- [Usage](#usage)
- [Feature Engineering](#feature-engineering)
- [Analysis](#analysis)
- [Validation](#validation)
- [Contributing](#contributing)
- [License](#license)

## Tech Stack

- Intel Optimized Modin and Pandas for efficient data processing
- Intel AI Analytics Toolkit for in-depth data analysis
- GitHub MLflow and DVC for experiment tracking and version control
- Intel Optimized XGBoost, Optuna, and scikit-learn for machine learning tasks
- GitHub Actions and Streamlit for deployment
- daal4py for accelerated model inference

## Features

- Comprehensive feature engineering including handling missing values, utilizing outliers, and creating new features through binning.
- Analysis performed on Intel Optimized DevCloud Python platform, employing Seaborn and Matplotlib for visualization.

## Usage

1. Clone the repository:
```
https://github.com/AdarshWase/OneAPI-Water-Quality-Competition.git
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Run the application:
```
streamlit run app/main.py
```


## Feature Engineering

- **Missing Values:** Handled using mean, median imputations, and count mapping.
- **Outliers:** Retained and utilized to create new predictive features.
- **Feature Creations:** Employed binning method to generate new features.

## Analysis

Conducted on Intel Optimized DevCloud Python environment, identifying prevalent outliers and missing values. Utilized Seaborn and Matplotlib for extensive visualization.

## Validation

Implemented a multi-faceted validation strategy, creating multiple validation datasets to provide robust performance indicators.

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.