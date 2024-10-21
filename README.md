# Mortgage Rate Prediction Model

This project aims to predict 5-year fixed mortgage rates for the next two years using historical data on chartered bank interest rates and Canada bond yields.

## Project Overview

The model uses various time series and machine learning techniques to forecast mortgage rates based on historical trends and related economic indicators. It incorporates data from chartered bank interest rates, Canada bond yields, and inflation rates to make predictions.

## Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Multiple forecasting models:
  - SARIMA (Seasonal AutoRegressive Integrated Moving Average)
  - Random Forest
  - Linear Regression
  - Gradient Boosting
  - Support Vector Regression (SVR)
  - LSTM (Long Short-Term Memory) Neural Networks
  - Prophet
- Model evaluation and comparison
- Visualization of results

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:   ```
   git clone https://github.com/yourusername/mortgage-rate-prediction.git
   cd mortgage-rate-prediction   ```

2. Create a virtual environment and install dependencies:   ```
   make setup   ```

## Usage

To run the prediction model:

python predict_mortgage_rates.py

## Data Sources

The project uses the following data sources:
- Chartered bank interest rates (`chartered_bank_interest.csv`)
- Canada bond yields (`canada_bond_yield.csv`)
- Inflation data (fetched from FRED)

## Model Evaluation

The project evaluates each model using the following metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²)

## Visualizations

The project generates various visualizations, including:
- Correlation matrix
- Historical trends
- Feature importance plots
- Residual analysis plots
- Forecast plots for each model

All visualizations are saved in the `images/` directory.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

[Add your chosen license here]
