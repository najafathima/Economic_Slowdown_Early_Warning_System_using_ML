![Banner](./assets/banner.jpeg)


#### Project Overview

The Economic Slowdown Early Warning System is a machine learning–based project designed to identify early signals of economic slowdown using key macroeconomic indicators.
The goal of this system is to assist policymakers, analysts, and researchers in understanding potential economic risks before a slowdown becomes severe.


.

#### Objectives

- Analyze historical macroeconomic indicators

- Capture trend and lag effects in economic data

- Predict the risk of economic slowdown in advance

- Convert predictions into an interpretable risk score instead of a simple yes/no output


#### Input Features

The model uses the following macroeconomic indicators as input:

- GDP Growth (%)

- Inflation Rate (%)

- Export Growth (%)

- Population Growth (%)

- Current Account Balance (% of GDP)

- Lagged economic indicators (previous years’ values)

- Lag features are used to capture temporal economic trends and delayed impacts.

#### Machine Learning Approach

- Data preprocessing (handling missing values, scaling, encoding)

- Lag feature engineering to model economic momentum

- Random Forest Classifier for robust, non-linear predictions

- Output provided as a probability-based slowdown risk score


#### Technologies Used

- Python

- Pandas, NumPy

- Scikit-learn

- Jupyter Notebook

- Streamlit (for user interface)

- Git & GitHub


#### Output

- The system outputs a slowdown risk score

- Higher scores indicate a greater likelihood of economic slowdown

- Designed to support early decision-making and preventive actions






