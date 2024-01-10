# DOT_MINI_PROJ
Stock Price Prediction Web App: 
This Python application utilizes machine learning techniques to predict stock prices and visualizes historical and predicted stock data. It features data preprocessing, model training, and Streamlit for interactive user interface. Additionally, it provides future stock price predictions for user-selected stocks.
Certainly! Here's a structured README for your Stock Price Prediction project:

# Stock Price Prediction App

## Overview

This repository contains a Streamlit app that leverages a Random Forest Regressor to predict stock prices for various companies. Users can select a company, view historical stock price trends, and get future stock price predictions for the next 30 days. The application aims to provide valuable insights into stock market data.

## Table of Contents

- [Introduction](#introduction)
- [Analysis Section](#analysis-section)
- [Streamlit App Section](#streamlit-app-section)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [Contact](#contact)

## Introduction

The Stock Price Prediction App utilizes a Random Forest Regressor model to forecast stock prices based on historical price data. Users can choose from a list of available companies, visualize past stock price trends, and receive predictions for the next 30 days.

## Analysis Section

The project begins with data collection and preprocessing. Historical stock price data is loaded for the selected company, and it's scaled using MinMaxScaler to prepare it for model training. The data is then split into training and testing sets. A Random Forest Regressor is trained on the training data to predict stock prices.

## Streamlit App Section

The Streamlit app provides an intuitive interface for users to interact with the stock price prediction model. Key features of the app include:

Company Selection: Users can select a company from a predefined list, including Amazon, Google, Tesla, TCS, and Apple.

Stock Price Visualization: The app displays historical stock price trends for the selected company, helping users understand past performance.

Future Predictions:Users can view predictions for the company's stock prices for the next 30 days, allowing for informed decision-making.

R2 Score: The app calculates and displays the R2 score, indicating the accuracy of the stock price predictions.

## Usage

To run the Stock Price Prediction App, ensure you have the required Python libraries, including Streamlit, scikit-learn, and pandas, installed in your environment. You can run the app using the following command:

```bash
streamlit run stock_price_prediction.py
```

## Conclusion

The Stock Price Prediction App provides a valuable tool for investors and traders to analyze stock price trends and make informed decisions. By leveraging machine learning techniques, it offers insights into future stock price movements.

## Contact

For inquiries, suggestions, or feedback, please don't hesitate to reach out to the project's creator:
Name: Rajvardhan Patil
Email: raj2003patil@gmail.com 

Your feedback and contributions are highly Appreciated