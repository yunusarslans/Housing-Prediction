# California Housing Price Prediction

## Project Overview

This project focuses on building a machine learning model to predict the median house value in California districts. The core task is a classic **regression problem** where we predict a continuous target variable (`MedHouseVal`) based on various demographic and geographical features. The project serves as a hands-on application of a full machine learning pipeline, from data exploration and preprocessing to model training and evaluation.

## Dataset

The dataset used is the well-known `California Housing` dataset, which is conveniently available within the `scikit-learn` library. It contains data from the 1990 California census and includes features such as median income, house age, population, and geographical coordinates.

**Features:**
- `MedInc`: Median income in the district (log-transformed)
- `HouseAge`: Median house age in the district
- `AveRooms`: Average number of rooms per household (log-transformed)
- `AveBedrms`: Average number of bedrooms per household (log-transformed)
- `Population`: District population (log-transformed)
- `AveOccup`: Average household occupancy (log-transformed)
- `Latitude`: Latitude of the district
- `Longitude`: Longitude of the district

**Target Variable:**
- `MedHouseVal`: Median house value (log-transformed)

## Methodology

The project follows a standard machine learning workflow:

1.  **Exploratory Data Analysis (EDA):** Initial analysis of the dataset to understand its structure, identify missing values, and visualize feature distributions and correlations.
2.  **Data Preprocessing:** Handled skewed feature distributions and the capped target variable by applying a logarithmic transformation (`np.log1p`). The data was then split into training (80%) and testing (20%) sets.
3.  **Modeling:** Two different regression models were trained and compared: a simple **Linear Regression** and a more complex **Random Forest Regressor**.
4.  **Hyperparameter Tuning:** The Random Forest model, which showed superior performance, was further optimized using `GridSearchCV` to find the best combination of hyperparameters.
5.  **Final Evaluation:** The final, optimized model was evaluated on the unseen test set to get a final, unbiased performance metric.

## Key Findings

### Model Performance

- **Final Model:** Random Forest Regressor (tuned)
- **Test RMSE:** **0.153**

The Random Forest model significantly outperformed the Linear Regression model. While the model exhibited some signs of overfitting (training RMSE was 0.055), its ability to generalize to new data, as measured by the test RMSE, is strong.

### Feature Importance

The analysis of feature importances from our final model revealed the primary drivers of house prices:

1.  `MedInc` (Median Income): **46.5%**
2.  `Latitude`: **12.8%**
3.  `Longitude`: **12.1%**
4.  `AveOccup` (Average Household Occupancy): **11.7%**
5.  `AveRooms` (Average Room Count): **9.0%**

This strongly suggests that a district's economic status and geographical location are the most influential factors in determining median house values in California.

## Technologies Used

- Python
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`



---
