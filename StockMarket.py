import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib

# Change the Matplotlib backend
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Dictionary of datasets
dataSetSelection = {
    "Amazon": "E:/DATASETS/amazon_stock_data.csv",
    "Google": "E:/DATASETS/google_stock_data.csv",
    "Tesla": "E:/DATASETS/tesla_stock_data.csv",
    "TCS": "E:/DATASETS/tcs_stock_data.csv",
    "Apple": "E:/DATASETS/apple_stock_data.csv",
}

# User input for stock selection
selected_stock = st.selectbox("Select a Stock", list(dataSetSelection.keys()))

# Load the stock data
stock_data = pd.read_csv(
    dataSetSelection[selected_stock], parse_dates=["Date"], infer_datetime_format=True
)

# Checking if the dataset contains null values
if stock_data.isnull().values.any():
    print(stock_data.isnull().sum())
    stock_data = stock_data.dropna()
    stock_data = stock_data.reset_index(drop=True)

print(stock_data.head())
print(stock_data.tail())
# Create the stock graph
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=stock_data["Date"], y=stock_data["Close"], mode="lines", name="Stock Price"
    )
)
fig.update_layout(
    title=f"{selected_stock} Stock Price",
    xaxis_title="Yearly Date",
    yaxis_title="Stock Price in Dollars[$]",
    xaxis=dict(tickformat="%Y", tickangle=45, tickfont=dict(size=10)),
)

# Display the title
st.title(f"{selected_stock} Stock Price")

# Display the stock graph using Streamlit
st.plotly_chart(fig)

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data["Close"].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]


# Prepare the training data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


seq_length = 60
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Reshape X_test
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

# Perform feature selection
feature_selector = SelectFromModel(
    RandomForestRegressor(n_estimators=125, random_state=60)
)
feature_selector.fit(X_train, y_train)
X_train = feature_selector.transform(X_train)
X_test = feature_selector.transform(X_test)

# Define the parameter grid
param_grid = {
    "n_estimators": [125],
    "max_depth": [25, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

# Create the random forest regressor
rf_regressor = RandomForestRegressor(random_state=60)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_rf_regressor = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions on the testing data
predictions = best_rf_regressor.predict(X_test)

# Scale the predictions back to their original range
scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y_test = y_test * scale_factor

# Calculate the R2 score (prediction accuracy)
r2_score_value = r2_score(y_test, predictions)

# Visualize the predicted vs. actual prices
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=stock_data["Date"][-len(y_test) :], y=y_test, mode="lines", name="Actual"
    )
)
fig.add_trace(
    go.Scatter(
        x=stock_data["Date"][-len(predictions) :],
        y=predictions,
        mode="lines",
        name="Predicted",
        line=dict(color="red"),
    )
)
fig.update_layout(
    title=f"{selected_stock} Stock Price Prediction",
    xaxis_title="Yearly Date",
    yaxis_title="Stock Price in Dollars[$]",
    xaxis=dict(tickformat="%b %Y", tickangle=45, tickfont=dict(size=10)),
)

# Display the title and R2 score
st.title(f"{selected_stock} Stock Price Prediction")
st.write(f"R2 Score: {r2_score_value}")

# Display the plot using Streamlit
st.plotly_chart(fig)

# Number of days to predict (30 days)
future_periods = 30

# Initialize a dictionary to store future predictions for the selected stock
future_predictions_dict = {}


def create_future_sequences(data, seq_length, future_periods):
    X_future = []

    # Append historical sequences
    for i in range(len(data) - seq_length, len(data)):
        X_future.append(data[i - seq_length : i, 0])

    # Create sequences for the future periods
    for _ in range(future_periods):
        X_future.append(np.zeros((seq_length,)))  # Pad with zeros

    return np.array(X_future)


# Preprocess the data for future predictions
seq_length = 60
X_train, y_train = create_sequences(scaled_data, seq_length)
X_train = feature_selector.transform(X_train)

# Create sequences for the future using the modified function
future_dates = pd.date_range(
    start=stock_data["Date"].max() + pd.Timedelta(days=1),
    periods=future_periods,
    freq="D",
)


X_future = create_future_sequences(scaled_data, seq_length, future_periods)
X_future_reshaped = X_future.reshape(-1, X_future.shape[-1])
X_future_selected = feature_selector.transform(X_future_reshaped)

# Train a Random Forest Regressor on the training data
rf_regressor.fit(X_train, y_train)

# Make predictions for the future dates
future_predictions = rf_regressor.predict(X_future_selected)

# Scale the future predictions back to their original range
future_predictions = future_predictions * scale_factor

# Store future predictions in the dictionary
future_predictions_dict[selected_stock] = {
    "Date": future_dates,
    "Predicted_Close": future_predictions,
}

# Print future predictions for the selected stock
print(f"Future Predictions for {selected_stock}:")
print("Date        Predicted_Close")
for date, prediction in zip(future_dates, future_predictions):
    print(f"{date}   {prediction:.2f}")

# Plot future predictions for the selected stock
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    future_dates[:future_periods],  # Only plot the first 30 dates
    future_predictions[:future_periods],  # Only plot the first 30 predictions
    label="Predicted Close Price",
    color="blue",
)
ax.set_title(f"Future Predictions for {selected_stock}")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()
ax.grid(True)
ax.set_xticklabels(future_dates, rotation=45)
plt.tight_layout()

# Display the plot using st.pyplot()
st.pyplot(fig)
