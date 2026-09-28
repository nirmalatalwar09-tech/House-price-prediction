# House-price-prediction
house price prediction using Linear Regression ,python ,pandas, Numpy, matplotlib, scikit-learn
# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts the estimated price of a house based on its:

- Area (square feet)
- Number of bedrooms
- Number of floors
- Age of the house

The project uses **Linear Regression**, a supervised machine learning algorithm, to learn the relationship between house features and house prices.

The user can enter the house details through the Python program, and the trained model provides an estimated house price.

## 🎯 Objective

The main objective of this project is to demonstrate how machine learning can be used to predict house prices based on different property features.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## 🤖 Machine Learning Algorithm

### Linear Regression

Linear Regression is used to model the relationship between the input features and the house price.

**Input Features:**
- Area
- Bedrooms
- Floors
- Age

**Target:**
- House Price

## ⚙️ How It Works

1. Sample house data is created using Pandas.
2. The data is divided into training and testing sets.
3. A Linear Regression model is trained using the training data.
4. The model predicts prices for the test data.
5. Model performance is evaluated using:
   - Mean Squared Error (MSE)
   - R² Score
6. The user enters house details.
7. The trained model predicts the estimated house price.
8. A graph displays actual prices versus predicted prices.

## ▶️ How to Run

### 1. Install Python

Make sure Python is installed on your computer.

### 2. Install required libraries

Open Command Prompt and run:

```bash
pip install pandas numpy matplotlib scikit-learn
