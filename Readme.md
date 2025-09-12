# USA House Price Prediction
`ML, Data Science, Regression, Python`
This project aims to predict house prices in the USA using various machine learning models. The dataset includes features such as the number of bedrooms, bathrooms, square footage, location, and other relevant attributes. The goal is to build a robust model that can accurately estimate house prices based on these features.

<img src="eval/predictions_distribution.png" alt="Illustration" width="100%">

## Structure of the Project
```bash

```

## Data
The dataset used in this project is sourced from [GitHub](https://github.com/miirshe/USA-Housing-Analysis-and-Prediction-Price/tree/main). It contains various features related to house properties in USA and their corresponding sale prices:

- `avg_income`: The average income of the area where the house is located.
- `house_age`: The age of the house in years.
- `nof_rooms`: The number of rooms in the house.
- `nof_bedrooms`: The number of bedrooms in the house.
- `population`: The population of the area.
- `address`: The address of the house.
- `price`: The sale price of the house (target variable).

## Methodology
The project follows a structured approach to data preprocessing, model training, and evaluation.

1. **Data Exploration**: Initial exploration of the dataset to understand the features and their distributions.
2. **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical
    features. Special attention is given to the `address` column, which is preprocessed to extract meaningful information.
3. **Model Selection**: Several regression models are tested, including:
    - Linear Regression
    - Bayesian Ridge Regression
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - K-Nearest Neighbors Regressor
    - XGBoost Regressor
4. **Hyperparameter Tuning**: Using techniques like Randomized Search Cross-Validation to find the best hyperparameters for each model.
5. **Model Evaluation**: Evaluating model performance using metrics such as Mean Squared Error (MSE) and R-squared ($R^2$).
6. **Final Predictions**: Generating predictions on the evaluation dataset and preparing a submission file.

## Results
The XGBoost model achieved the best performance with an $R^2$ of 0.872 on the validation set, followed closely by Bayesian Ridge and Linear Regression models. The K-Nearest Neighbors model had the lowest performance among the tested models.

| Model             | Test MSE | Test $R^2$ | Rank |
| ----------------- | -------: | -----------: | ---: |
| **XGBoost**           |  1.64e10 |    **0.872** |    **1** |
| Bayesian Ridge    |  1.79e10 |        0.860 |    2 |
| Linear Regression |  1.79e10 |        0.860 |    3 |
| Random Forest     |  1.85e10 |        0.855 |    4 |
| KNN               |  2.51e10 |        0.804 |    5 |

<img src="eval/actual_vs_predicted_xgb.png" alt="Illutration" width="100%">

The results improved a lot after cleaning the data more carefully and making use of the **address information**, which turned out to be very valuable. The best model was **XGBoost**, while simpler ones like KNN worked less well. This shows that **good data preparation is just as important as the choice of model**, and future improvements could make predictions even more accurate.


## How to Run the Project?
To run this project, make sure you have **Python** (>=3.11) installed on your system. Once ready, follow these steps:

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd Housing-Price-Prediction
   ```

2. **Install the dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project**

   * Option 1: Open the Jupyter notebook and execute the cells sequentially:

   * Option 2: Run the training and evaluation script directly:

     ```bash
     python train_eval.py
     ```


### **Note**
- *The project was developed and tested using Python 3.10 on Windows 11 with Jupyter Notebook and VSCode. Results may vary slightly with different Python versions or operating systems.*
- *To ensure reproducibility, random seeds were set for NumPy, scikit-learn, and XGBoost. However, due to the nature of some algorithms, results may still show minor variations between runs.*
- *The preprocessing pipeline is effective but not yet fully optimized. Further improvements in feature engineering, especially around the `address` column, could enhance results.*
- ***To go further:***
    * *Experiment with more advanced models like LightGBM or CatBoost.*
    * *Explore deep learning approaches using frameworks like TensorFlow or PyTorch.*
    * *Incorporate external data sources for richer feature sets, such as geographic or economic indicators.*
> Dataset provided by [USA Housing Dataset on GitHub](https://github.com/miirshe/USA-Housing-Analysis-and-Prediction-Price/tree/main).