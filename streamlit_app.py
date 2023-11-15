import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load data (adjust this part to load your specific dataset)
@st.cache
def load_data():
    # Replace with the code to load your actual data
    df = pd.read_csv('HouseData.csv')
    return df

df = load_data()

# Streamlit app title
st.title("Random Forest Regression Analysis")

# Display DataFrame in Streamlit
if st.checkbox('Show DataFrame'):
    st.write(df)

# Data Preprocessing (Include your preprocessing steps here)

# ... Your existing preprocessing code ...

# Split data into features and target variable
X = df_scaled
y = df['InvestmentValue_log']

# Split data into training and testing sets
test_size = st.slider('Test size', min_value=0.1, max_value=0.5, value=0.2, step=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Plot feature importances
importances = rf.feature_importances_
sorted_idx = importances.argsort()

fig, ax = plt.subplots()
ax.barh(X.columns[sorted_idx], importances[sorted_idx])
ax.set_xlabel("Random Forest Feature Importance")
st.pyplot(fig)

# Grid Search for Hyperparameter Tuning
if st.button('Run GridSearchCV'):
    param_grid = {
        # Define your parameter grid
    }

    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    st.write("Best parameters found: ", grid_search.best_params_)

    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")

