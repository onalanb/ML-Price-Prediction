# Import necessary libraries
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

#######################
### DATA COLLECTION ###
#######################

# Set your Alpha Vantage API key
api_key = "your_api_key"

# Choose a company and collect historical stock price data
symbol = "AAPL"  # Example: Apple
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')

# Save data to a DataFrame
df = data.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'})

##########################
### DATA PREPROCESSING ###
##########################

# Handle missing values and remove outliers
df.dropna(inplace=True)
# Add feature engineering (e.g., moving averages)
df['ma_50'] = df['close'].rolling(window=50).mean()
df['ma_200'] = df['close'].rolling(window=200).mean()

#######################
### MODEL SELECTION ###
#######################

# Choose a machine learning algorithm (e.g., Linear Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Prepare features and target variable
X = df[['open', 'high', 'low', 'volume', 'ma_50', 'ma_200']]
y = df['close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

##################
### EVALUATION ###
##################

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

##################
### PREDICTION ###
##################

# Use the trained model to predict future stock prices (assuming new data)
future_data = ...  # Load new data for prediction
future_predictions = model.predict(future_data)

#####################
### VISUALIZATION ###
#####################

# Visualize predictions alongside actual prices
plt.figure(figsize=(10, 6))
plt.plot(df.index[-len(y_test):], y_test, label='Actual Prices')
plt.plot(df.index[-len(y_test):], y_pred, label='Predicted Prices')
plt.title('Model Prediction vs Actual Prices')
plt.legend()
plt.show()