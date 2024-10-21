# predict_mortgage_rates.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from fbprophet import Prophet
from pandas_datareader import data as pdr
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

def fetch_inflation_data(start_date, end_date):
    """
    Fetch inflation data from FRED.

    Parameters:
    - start_date (str): Start date for fetching data.
    - end_date (str): End date for fetching data.

    Returns:
    - pd.DataFrame: DataFrame containing inflation data.
    """
    try:
        # Fetch inflation data (CPIAUCSL) from FRED
        inflation_df = pdr.get_data_fred('CPIAUCSL', start=start_date, end=end_date)
        inflation_df.rename(columns={'CPIAUCSL': 'inflation'}, inplace=True)
        inflation_df.index.name = 'date'
        inflation_df.reset_index(inplace=True)
        print("Inflation Data Head:")
        print(inflation_df.head())
        return inflation_df
    except Exception as e:
        print(f"Error fetching inflation data: {e}")
        return pd.DataFrame()

def load_and_clean_data(chartered_path, bond_yield_path, inflation_df):
    """
    Load and clean the Chartered Bank Interest Rates, Canada Bond Yield, and Inflation data.

    Parameters:
    - chartered_path (str): Path to 'chartered_bank_interest.csv'.
    - bond_yield_path (str): Path to 'canada_bond_yield.csv'.
    - inflation_df (pd.DataFrame): DataFrame containing inflation data.

    Returns:
    - pd.DataFrame: Merged and cleaned DataFrame.
    """
    # Load the Chartered Bank Interest Rates data
    chartered_df = pd.read_csv(chartered_path, parse_dates=['date'])
    chartered_df.columns = chartered_df.columns.str.strip()
    print("\nChartered Bank Interest Rates Columns:")
    print(chartered_df.columns.tolist())

    # Load the Canada Bond Yield data
    bond_yield_df = pd.read_csv(bond_yield_path, parse_dates=['date'])
    bond_yield_df.columns = bond_yield_df.columns.str.strip()
    print("\nCanada Bond Yield Columns:")
    print(bond_yield_df.columns.tolist())

    # Verify if 'V80691335' exists in chartered_df
    if 'V80691335' not in chartered_df.columns:
        raise KeyError("'V80691335' column not found in 'chartered_bank_interest.csv'. Please verify the CSV file.")

    # Fill missing values using forward fill
    chartered_df.fillna(method='ffill', inplace=True)
    bond_yield_df.fillna(method='ffill', inplace=True)
    inflation_df.fillna(method='ffill', inplace=True)

    # Focus on the 5-year conventional mortgage rate (V80691335)
    mortgage_df = chartered_df[['date', 'V80691335']].rename(columns={'V80691335': 'mortgage_rate'})
    print("\nMortgage DataFrame Head:")
    print(mortgage_df.head())

    # Merge with bond yield data on date
    merged_df = pd.merge(mortgage_df, bond_yield_df, on='date', how='left')
    print("\nMerged DataFrame Head (Post-Merge):")
    print(merged_df.head())

    # Merge with inflation data on date
    merged_df = pd.merge(merged_df, inflation_df, on='date', how='left')
    print("\nMerged DataFrame with Inflation Head:")
    print(merged_df.head())

    # Rename bond yield columns for clarity
    merged_df.rename(columns={'value': 'bond_yield_value', 'market': 'bond_yield_market'}, inplace=True)

    # Fill any remaining missing values after merge
    merged_df.fillna(method='ffill', inplace=True)

    # Drop any rows that still contain NaN values
    merged_df.dropna(inplace=True)

    # Set date as index
    merged_df.set_index('date', inplace=True)

    # Final DataFrame check
    print("\nFinal Merged DataFrame Head:")
    print(merged_df.head())

    return merged_df

def plot_correlation_matrix(df):
    """
    Plot the correlation matrix of the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame to plot the correlation matrix.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    plt.title('Correlation Matrix')
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.savefig('images/correlation_matrix.png')
    plt.show()

def plot_historical_trends(df):
    """
    Plot historical trends of mortgage rates, bond yields, and inflation.

    Parameters:
    - df (pd.DataFrame): DataFrame containing historical data.

    Returns:
    - None
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['mortgage_rate'], label='5-Year Mortgage Rate')
    plt.plot(df.index, df['bond_yield_value'], label='Bond Yield Value')
    plt.plot(df.index, df['inflation'], label='Inflation', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Rate (%)')
    plt.title('Historical Trends of Mortgage Rates, Bond Yields, and Inflation')
    plt.legend()
    plt.savefig('images/historical_trends.png')
    plt.show()

def train_sarima_model(train_data):
    """
    Train a SARIMA model on the training data.

    Parameters:
    - train_data (pd.DataFrame): Training DataFrame.

    Returns:
    - SARIMAXResults: Fitted SARIMA model.
    """
    print("\nTraining SARIMA Model...")
    model = SARIMAX(
        train_data['mortgage_rate'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        exog=train_data[['bond_yield_value', 'bond_yield_market', 'inflation']]
    )
    model_fit = model.fit(disp=False)
    print("\nSARIMA Model Summary:")
    print(model_fit.summary())

    # Plot diagnostics and save
    fig = model_fit.plot_diagnostics(figsize=(15, 12))
    plt.savefig('images/sarima_diagnostics.png')
    plt.show()

    return model_fit

def train_random_forest(train_data, test_data):
    """
    Train a Random Forest model on the training data.

    Parameters:
    - train_data (pd.DataFrame): Training DataFrame.
    - test_data (pd.DataFrame): Testing DataFrame.

    Returns:
    - np.ndarray: Predicted values.
    """
    print("\nTraining Random Forest Model...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train_data[['bond_yield_value', 'bond_yield_market', 'inflation']], train_data['mortgage_rate'])
    predictions = rf.predict(test_data[['bond_yield_value', 'bond_yield_market', 'inflation']])
    return predictions

def train_linear_regression(train_data, test_data):
    """
    Train a Linear Regression model on the training data.

    Parameters:
    - train_data (pd.DataFrame): Training DataFrame.
    - test_data (pd.DataFrame): Testing DataFrame.

    Returns:
    - np.ndarray: Predicted values.
    """
    print("\nTraining Linear Regression Model...")
    lr = LinearRegression()
    lr.fit(train_data[['bond_yield_value', 'bond_yield_market', 'inflation']], train_data['mortgage_rate'])
    predictions = lr.predict(test_data[['bond_yield_value', 'bond_yield_market', 'inflation']])
    return predictions

def train_gradient_boosting(train_data, test_data):
    """
    Train a Gradient Boosting model on the training data.

    Parameters:
    - train_data (pd.DataFrame): Training DataFrame.
    - test_data (pd.DataFrame): Testing DataFrame.

    Returns:
    - np.ndarray: Predicted values.
    """
    print("\nTraining Gradient Boosting Model...")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(train_data[['bond_yield_value', 'bond_yield_market', 'inflation']], train_data['mortgage_rate'])
    predictions = gb.predict(test_data[['bond_yield_value', 'bond_yield_market', 'inflation']])
    return predictions

def train_svr(train_data, test_data):
    """
    Train a Support Vector Regression model on the training data.

    Parameters:
    - train_data (pd.DataFrame): Training DataFrame.
    - test_data (pd.DataFrame): Testing DataFrame.

    Returns:
    - np.ndarray: Predicted values.
    """
    print("\nTraining Support Vector Regression Model...")
    svr = SVR(kernel='rbf')
    svr.fit(train_data[['bond_yield_value', 'bond_yield_market', 'inflation']], train_data['mortgage_rate'])
    predictions = svr.predict(test_data[['bond_yield_value', 'bond_yield_market', 'inflation']])
    return predictions

def train_lstm(train_data, test_data):
    """
    Train an LSTM model on the training data.

    Parameters:
    - train_data (pd.DataFrame): Training DataFrame.
    - test_data (pd.DataFrame): Testing DataFrame.

    Returns:
    - np.ndarray: Predicted values.
    """
    print("\nTraining LSTM Model...")

    # Feature scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data[['mortgage_rate', 'bond_yield_value', 'bond_yield_market', 'inflation']])
    test_scaled = scaler.transform(test_data[['mortgage_rate', 'bond_yield_value', 'bond_yield_market', 'inflation']])

    # Create sequences
    X_train, y_train = train_scaled[:, 1:], train_scaled[:, 0]
    X_test, y_test = test_scaled[:, 1:], test_scaled[:, 0]

    # Reshape input to be [samples, time_steps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fit the model
    model.fit(X_train, y_train, epochs=50, batch_size=72, verbose=0, shuffle=False)
    print("LSTM Model Trained.")

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate((predictions, X_test[:, :, 1:]), axis=2))[:, 0]

    return predictions

def train_prophet(df):
    """
    Train a Prophet model on the data.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.

    Returns:
    - pd.DataFrame: Forecasted values.
    """
    print("\nTraining Prophet Model...")
    prophet_df = df.reset_index()[['date', 'mortgage_rate']].rename(columns={'date': 'ds', 'mortgage_rate': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    print("Prophet Model Trained.")

    # Create future dataframe
    future = model.make_future_dataframe(periods=24, freq='M')
    forecast = model.predict(future)
    print("Prophet Forecast Completed.")
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def evaluate_model(test_data, predictions, model_name):
    """
    Evaluate the model's performance using MAE, RMSE, MAPE, and R-squared.

    Parameters:
    - test_data (pd.DataFrame): Testing DataFrame.
    - predictions (pd.Series or np.ndarray): Predicted mortgage rates.
    - model_name (str): Name of the model being evaluated.

    Returns:
    - None
    """
    mae = mean_absolute_error(test_data['mortgage_rate'], predictions)
    rmse = np.sqrt(mean_squared_error(test_data['mortgage_rate'], predictions))
    mape = np.mean(np.abs((test_data['mortgage_rate'] - predictions) / test_data['mortgage_rate'])) * 100
    r2 = r2_score(test_data['mortgage_rate'], predictions)
    print(f"\n{model_name} Model Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R-squared: {r2:.4f}")

def plot_feature_importance(model, feature_names, model_name):
    """
    Plot and save the feature importance for tree-based models.

    Parameters:
    - model: Trained model.
    - feature_names (list): List of feature names.
    - model_name (str): Name of the model.

    Returns:
    - None
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(8, 6))
        plt.title(f'Feature Importances - {model_name}')
        sns.barplot(x=[feature_names[i] for i in indices], y=importances[indices], palette='viridis')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'images/{model_name.lower()}_feature_importance.png')
        plt.show()
    else:
        print(f"{model_name} does not have feature importances to plot.")

def plot_residuals(model_fit, model_name):
    """
    Plot and save residuals of the SARIMA model.

    Parameters:
    - model_fit (SARIMAXResults): Fitted SARIMA model.
    - model_name (str): Name of the model.

    Returns:
    - None
    """
    residuals = model_fit.resid
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, kde=True, color='skyblue')
    plt.title(f'Residuals Distribution - {model_name}')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig(f'images/{model_name.lower()}_residuals_distribution.png')
    plt.show()

    # Residual plot
    plt.figure(figsize=(12, 6))
    plt.plot(residuals)
    plt.title(f'Residuals Over Time - {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.savefig(f'images/{model_name.lower()}_residuals_over_time.png')
    plt.show()

def plot_feature_importance_rf_gb(model, feature_names, model_name):
    """
    Plot and save feature importance for Random Forest and Gradient Boosting models.

    Parameters:
    - model: Trained model.
    - feature_names (list): List of feature names.
    - model_name (str): Name of the model.

    Returns:
    - None
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(8, 6))
        plt.title(f'Feature Importances - {model_name}')
        sns.barplot(x=[feature_names[i] for i in indices], y=importances[indices], palette='viridis')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'images/{model_name.lower()}_feature_importance.png')
        plt.show()
    else:
        print(f"{model_name} does not have feature importances to plot.")

def main():
    # Ensure the images directory exists
    os.makedirs('images', exist_ok=True)

    # File paths
    chartered_path = 'chartered_bank_interest.csv'
    bond_yield_path = 'canada_bond_yield.csv'

    # Fetch inflation data
    inflation_df = fetch_inflation_data(start_date='1967-01-01', end_date='2023-12-31')

    # Load and clean data
    try:
        merged_df = load_and_clean_data(chartered_path, bond_yield_path, inflation_df)
        print("\nData loaded and cleaned successfully.")
    except KeyError as e:
        print(f"\nError: {e}")
        print("Please check the column names in your CSV files.")
        return
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the CSV files are in the correct directory.")
        return

    # Verify merged DataFrame
    if merged_df.empty:
        print("\nMerged DataFrame is empty. Please check the CSV files for data.")
        return

    # Plot correlation matrix
    print("\nPlotting Correlation Matrix...")
    plot_correlation_matrix(merged_df)

    # Plot historical trends
    print("\nPlotting Historical Trends...")
    plot_historical_trends(merged_df)

    # Split the data into training and testing sets
    train = merged_df.iloc[:-24]  # Use all data except the last 24 months for training
    test = merged_df.iloc[-24:]   # Last 24 months for testing

    # Check if there is enough data
    if len(train) < 24:
        print("\nNot enough data for training. Please provide more historical data.")
        return

    # List to store feature names
    feature_names = ['bond_yield_value', 'bond_yield_market', 'inflation']

    # Train SARIMA model
    sarima_fit = train_sarima_model(train)

    # Plot residuals for SARIMA
    plot_residuals(sarima_fit, "SARIMA")

    # Forecast on the test set using SARIMA
    sarima_forecast = sarima_fit.get_forecast(
        steps=24,
        exog=test[feature_names]
    )
    sarima_predictions = sarima_forecast.predicted_mean
    sarima_conf_int = sarima_forecast.conf_int()

    # Plot SARIMA forecast vs actual
    plot_forecast(train, test, sarima_predictions, sarima_conf_int, "SARIMA")

    # Evaluate SARIMA model
    evaluate_model(test, sarima_predictions, "SARIMA")

    # Train and evaluate Random Forest model
    rf_predictions = train_random_forest(train, test)
    evaluate_model(test, rf_predictions, "Random Forest")
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['mortgage_rate'], label='Training')
    plt.plot(test.index, test['mortgage_rate'], label='Actual', color='blue')
    plt.plot(test.index, rf_predictions, label='Random Forest Forecast', color='green')
    plt.xlabel('Date')
    plt.ylabel('5-Year Fixed Mortgage Rate (%)')
    plt.title('Mortgage Rate Forecast using Random Forest')
    plt.legend()
    plt.savefig('images/random_forest_forecast.png')
    plt.show()

    # Feature Importance for Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train[feature_names], train['mortgage_rate'])
    plot_feature_importance_rf_gb(rf, feature_names, "Random Forest")

    # Train and evaluate Linear Regression model
    lr_predictions = train_linear_regression(train, test)
    evaluate_model(test, lr_predictions, "Linear Regression")
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['mortgage_rate'], label='Training')
    plt.plot(test.index, test['mortgage_rate'], label='Actual', color='blue')
    plt.plot(test.index, lr_predictions, label='Linear Regression Forecast', color='purple')
    plt.xlabel('Date')
    plt.ylabel('5-Year Fixed Mortgage Rate (%)')
    plt.title('Mortgage Rate Forecast using Linear Regression')
    plt.legend()
    plt.savefig('images/linear_regression_forecast.png')
    plt.show()

    # Train and evaluate Gradient Boosting model
    gb_predictions = train_gradient_boosting(train, test)
    evaluate_model(test, gb_predictions, "Gradient Boosting")
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['mortgage_rate'], label='Training')
    plt.plot(test.index, test['mortgage_rate'], label='Actual', color='blue')
    plt.plot(test.index, gb_predictions, label='Gradient Boosting Forecast', color='orange')
    plt.xlabel('Date')
    plt.ylabel('5-Year Fixed Mortgage Rate (%)')
    plt.title('Mortgage Rate Forecast using Gradient Boosting')
    plt.legend()
    plt.savefig('images/gradient_boosting_forecast.png')
    plt.show()

    # Feature Importance for Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(train[feature_names], train['mortgage_rate'])
    plot_feature_importance_rf_gb(gb, feature_names, "Gradient Boosting")

    # Train and evaluate Support Vector Regression model
    svr_predictions = train_svr(train, test)
    evaluate_model(test, svr_predictions, "Support Vector Regression")
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['mortgage_rate'], label='Training')
    plt.plot(test.index, test['mortgage_rate'], label='Actual', color='blue')
    plt.plot(test.index, svr_predictions, label='Support Vector Regression Forecast', color='brown')
    plt.xlabel('Date')
    plt.ylabel('5-Year Fixed Mortgage Rate (%)')
    plt.title('Mortgage Rate Forecast using Support Vector Regression')
    plt.legend()
    plt.savefig('images/svr_forecast.png')
    plt.show()

    # Train and evaluate LSTM model
    lstm_predictions = train_lstm(train, test)
    evaluate_model(test, lstm_predictions, "LSTM")
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['mortgage_rate'], label='Training')
    plt.plot(test.index, test['mortgage_rate'], label='Actual', color='blue')
    plt.plot(test.index, lstm_predictions, label='LSTM Forecast', color='cyan')
    plt.xlabel('Date')
    plt.ylabel('5-Year Fixed Mortgage Rate (%)')
    plt.title('Mortgage Rate Forecast using LSTM')
    plt.legend()
    plt.savefig('images/lstm_forecast.png')
    plt.show()

    # Train and evaluate Prophet model
    prophet_forecast = train_prophet(merged_df)
    prophet_test = prophet_forecast.set_index('ds').loc[test.index]
    evaluate_model(test, prophet_test['yhat'], "Prophet")
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['mortgage_rate'], label='Training')
    plt.plot(test.index, test['mortgage_rate'], label='Actual', color='blue')
    plt.plot(prophet_test.index, prophet_test['yhat'], label='Prophet Forecast', color='magenta')
    plt.xlabel('Date')
    plt.ylabel('5-Year Fixed Mortgage Rate (%)')
    plt.title('Mortgage Rate Forecast using Prophet')
    plt.legend()
    plt.savefig('images/prophet_forecast.png')
    plt.show()

    # Forecast future rates using SARIMA
    forecast_future_rates(sarima_fit, merged_df, steps=24)

if __name__ == "__main__":
    main()
