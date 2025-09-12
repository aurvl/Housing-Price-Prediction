# ****************House Price Prediction using ML****************

# ========================================= 
## Libraries
# ========================================= 
import os
import re
import joblib
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import Markdown, display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor

SEED = 42
warnings.filterwarnings('ignore')
pd.options.display.max_rows = 100
pd.options.display.max_columns = 50

# ========================================= 
## Explanatory Data Analysis
# ========================================= 

### Load data and quick look
df = pd.read_csv('house_data.csv')
print(df.head())
print(df.info())
print(df.duplicated().sum())
print(df.describe())

### Data Visualization
plt.style.use('dark_background')
plt.figure(figsize=(15, 5))
sns.histplot(df['Price'], bins=50, kde=True, color='blue', edgecolor='gray')
plt.title('Distribution of house prices')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(17, 4))
sns.histplot(df['Avg. Area House Age'], bins=30, kde=True, color='purple', edgecolor='gray', ax=ax[0])
sns.histplot(df['Avg. Area Number of Rooms'], bins=30, kde=True, color='orange', edgecolor='gray', ax=ax[1])
ax[0].set_title('Distribution of Average Area House Age')
ax[0].set_ylabel('Frequency')
ax[1].set_title('Distribution of Average Area Number of Rooms')
ax[1].set_ylabel('Frequency')
plt.grid(False)
plt.show()

# Corrplot
colplot = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Area Population']
color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# 2 rows and 2 columns
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
for i in range(2):
    for j in range(2):
        sns.scatterplot(data=df, x=colplot[i*2 + j], y='Price', color=color[i*2 + j], ax=ax[i, j])
        ax[i, j].set_title(f'Price vs {colplot[i*2 + j]}')
        ax[i, j].set_xlabel('')

# We need only numerical features for correlation matrix
num = df.select_dtypes(include=[np.number])

plt.figure(figsize=(14, 6))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.show()

def extract_state(s: str) -> str:
    if pd.isna(s):
        return np.nan
    # 1) two-letter code immediately before a 5-digit ZIP (optional -4)
    m = re.search(r'\b([A-Z]{2})\s+\d{5}(?:-\d{4})?\b', s)
    if m:
        return m.group(1)
    # 2) fallback: comma + state
    m = re.search(r',\s*([A-Z]{2})\b', s)
    return m.group(1) if m else np.nan

adress_data = df[['Address', 'Price']].copy()
adress_data['state'] = adress_data['Address'].apply(extract_state)

print(adress_data['state'].unique())
print('\nNb. of states:', adress_data['state'].nunique())

adress_data['state'].value_counts().sort_values(ascending=False).plot(kind='bar', figsize=(20, 5), color='tab:blue', edgecolor='none')
plt.title('States distribution')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.xlabel('States')
plt.xticks(rotation=90)
plt.grid(axis='y', alpha=0.5)
plt.show()

valid_states = adress_data[~adress_data['state'].isin(['AP', 'AE', 'AA'])]['state'].unique().tolist()
# # Valid states:
# ['NE', 'CA', 'WI', 'KS', 'CO', 'TN', 'NM', 'PW', 'AR', 'HI', 'ME',
# 'IN', 'MI', 'DE', 'AZ', 'MA', 'MN', 'AL', 'NY', 'NV', 'VA', 'ID',
# 'OK', 'NH', 'MO', 'WV', 'WY', 'MH', 'UT', 'SD', 'CT', 'AK', 'WA',
# 'RI', 'NJ', 'KY', 'NC', 'IA', 'VT', 'FM', 'ND', 'LA', 'MP', 'OR',
# 'TX', 'DC', 'PR', 'MT', 'AS', 'OH', 'MS', 'IL', 'VI', 'GA', 'PA',
# 'MD', 'SC', 'GU', 'FL']

filtered = len(adress_data[adress_data['state'].isin(valid_states)])
print(f"Conserved data: {filtered}")

def plot_states(adress_data, metric='mean', var='state', lab='States', boxplot=True, rot=90):
    if metric == 'mean':
        price_b_state = adress_data.groupby(var)['Price'].mean().sort_values(ascending=False)
        metric_lab = 'Average'
    elif metric == 'std':
        price_b_state = adress_data.groupby(var)['Price'].std().sort_values(ascending=False)
        metric_lab = 'Standard Deviation of'
    elif metric == 'count':
        price_b_state = adress_data.groupby(var)['Price'].count().sort_values(ascending=False)
        metric_lab = 'Numb. of available'
    elif metric == 'median':
        price_b_state = adress_data.groupby(var)['Price'].median().sort_values(ascending=False)
        metric_lab = 'Median'
    else:
        raise ValueError("Metric must be 'mean', 'std', or 'count'")
    price_b_state.plot(kind='bar', figsize=(20, 5), cmap='plasma', edgecolor='none')
    plt.title(f'{metric_lab} House Price by {lab}')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlabel(lab)
    plt.xticks(rotation=rot)
    plt.grid(axis='y', alpha=0.5)
    plt.show()

    if boxplot:
        # plot boxplot by state with points dispertion on the box
        plt.figure(figsize=(15, 5))
        sns.boxplot(data=adress_data, x=var, y='Price', palette='Set3')
        sns.stripplot(data=adress_data, x=var, y='Price', color='white', alpha=0.5, jitter=0.2, size=2.5)
        plt.title(f'House price distribution by {lab}')
        plt.xticks(rotation=90)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.grid(alpha=0.3)
        plt.show()
    
plot_states(adress_data, metric='std')

territories = {'PR','GU','VI','AS','MP','FM','MH','PW'}
region_map = {
    # Northeast
    'CT':'Northeast','ME':'Northeast','MA':'Northeast','NH':'Northeast','RI':'Northeast','VT':'Northeast',
    'NJ':'Northeast','NY':'Northeast','PA':'Northeast',
    # Midwest
    'IL':'Midwest','IN':'Midwest','MI':'Midwest','OH':'Midwest','WI':'Midwest',
    'IA':'Midwest','KS':'Midwest','MN':'Midwest','MO':'Midwest','NE':'Midwest','ND':'Midwest','SD':'Midwest',
    # South
    'DE':'South','FL':'South','GA':'South','MD':'South','NC':'South','SC':'South','VA':'South','DC':'South',
    'WV':'South','AL':'South','KY':'South','MS':'South','TN':'South',
    'AR':'South','LA':'South','OK':'South','TX':'South',
    # West
    'AZ':'West','CO':'West','ID':'West','MT':'West','NV':'West','NM':'West','UT':'West','WY':'West',
    'AK':'West','CA':'West','HI':'West','OR':'West','WA':'West'
}

adress_data['region'] = adress_data['state'].map(region_map)
adress_data.loc[adress_data['state'].isin(territories), 'region'] = 'Territories'
plot_states(adress_data, var='region', metric='std', lab='Regions', boxplot=True, rot=0)
filtered_data = adress_data[adress_data['state'].isin(valid_states)].copy()
dft = df.merge(filtered_data[['state']], left_index=True, right_index=True)
dft = dft[dft['state'].isin(valid_states)].copy()
dft.reset_index(drop=True, inplace=True)
dft.drop(columns='state', inplace=True)
print(dft.head())

# ========================================= 
## Data Preprocessing
# ========================================= 

# Let's split the data into train and test
split_rate = 0.2 # 20% for test
df_ = dft.copy()
train, test = train_test_split(df_, test_size=split_rate, random_state=SEED) # Seed=42 for reproducibility
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Let's now create a preprocessing pipeline to prepare the data for modeling

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Learn per-column clipping bounds on the training set and reuse them at transform time.
    method: 'iqr' (Q1-1.5*IQR, Q3+1.5*IQR) or 'zscore' (mean±k*std) or 'quantile' (q_low, q_high).
    """
    def __init__(self, method: str = 'iqr', z_k: float = 3.0, q_low: float = 0.01, q_high: float = 0.99):
        self.method = method
        self.z_k = z_k
        self.q_low = q_low
        self.q_high = q_high
        self.bounds_ = None          # dict: col -> (lower, upper)
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y=None):
        if isinstance(X, pd.Series):
            X = X.to_frame()
        self.feature_names_in_ = list(X.columns)
        bounds = {}
        for col in self.feature_names_in_:
            s = pd.to_numeric(X[col], errors='coerce')
            if self.method == 'iqr':
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            elif self.method == 'zscore':
                m, sd = s.mean(), s.std(ddof=0)
                lo, hi = m - self.z_k*sd, m + self.z_k*sd
            elif self.method == 'quantile':
                lo, hi = s.quantile(self.q_low), s.quantile(self.q_high)
            else:
                raise ValueError("method must be 'iqr', 'zscore', or 'quantile'")
            bounds[col] = (lo, hi)
        self.bounds_ = bounds
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.Series):
            X = X.to_frame()
        Xc = X.copy()
        for col in self.feature_names_in_:
            if col in Xc:
                lo, hi = self.bounds_[col]
                Xc[col] = pd.to_numeric(Xc[col], errors='coerce').clip(lower=lo, upper=hi)
        return Xc
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # optional: expose the learned bounds
    def get_bounds(self) -> dict:
        return self.bounds_

def preprocess_data(data:pd.DataFrame, train:bool=True, target:str='price') -> pd.DataFrame:
    """
    This function preprocesses the data for modeling. 
    It handles missing values, encodes categorical variables, scales numerical features, and selects relevant features.

    Args:
    data : pd.DataFrame)
        The input dataframe to preprocess.
    train :bool, optional
        Whether the data is for training or testing. Default is True.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    data = data.copy()
    # 1. processing address to extract state regions
    if 'Address' in data.columns:
        data['state'] = data['Address'].apply(extract_state)
        data.drop(columns=['Address'], inplace=True)
    else:
        raise ValueError("The dataframe must contain an 'Address' column.")
    
    territories = {'PR','GU','VI','AS','MP','FM','MH','PW'}
    region_map = {
        # Northeast
        'CT':'Northeast','ME':'Northeast','MA':'Northeast','NH':'Northeast','RI':'Northeast','VT':'Northeast',
        'NJ':'Northeast','NY':'Northeast','PA':'Northeast',
        # Midwest
        'IL':'Midwest','IN':'Midwest','MI':'Midwest','OH':'Midwest','WI':'Midwest',
        'IA':'Midwest','KS':'Midwest','MN':'Midwest','MO':'Midwest','NE':'Midwest','ND':'Midwest','SD':'Midwest',
        # South
        'DE':'South','FL':'South','GA':'South','MD':'South','NC':'South','SC':'South','VA':'South','DC':'South',
        'WV':'South','AL':'South','KY':'South','MS':'South','TN':'South',
        'AR':'South','LA':'South','OK':'South','TX':'South',
        # West
        'AZ':'West','CO':'West','ID':'West','MT':'West','NV':'West','NM':'West','UT':'West','WY':'West',
        'AK':'West','CA':'West','HI':'West','OR':'West','WA':'West'
    }
    data['region'] = data['state'].map(region_map)
    data.loc[data['state'].isin(territories), 'region'] = 'Territories'
    data.drop(columns=['state'], inplace=True)
    
    # 2. Renaming columns
    verified_colnames = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'region']
    colnames = ['avg_income', 'house_age', 'nof_rooms', 'nof_bedrooms', 'population', 'price', 'region']
    
    # 3. ensure the good columns are present
    if train:
        assert all([col in data.columns for col in verified_colnames]), f"Some columns are missing in the training data.\nAvailable colomns: {data.columns.tolist()}"
        data.columns = colnames
    else:
        verified_colnames = [col for col in verified_colnames if col != 'Price']
        assert all([col in data.columns for col in verified_colnames]), f"Some columns are missing in the testing data.\nAvailable colomns: {data.columns.tolist()}"
        data.columns = [col for col in colnames if col != 'price']
    
    if train:
        X = data.drop(columns=[target])
        X_num = X.select_dtypes(include=[np.number])
        X_cat = X.select_dtypes(include=['object'])
        y = data[target]
        
        # 4. Handling missing values
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = pd.DataFrame(imputer.fit_transform(X_num), columns=X_num.columns)
        
        # 5. Dealing with outliers
        clipper = OutlierClipper(method='zscore', z_k=2.25)
        X_out = clipper.fit_transform(X_imputed)
        
        # 6. Encoding categorical variables
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        state_encoded = pd.DataFrame(encoder.fit_transform(X_cat), columns=encoder.get_feature_names_out(['region']))
        X = pd.concat([X_out.reset_index(drop=True), state_encoded.reset_index(drop=True)], axis=1)
        
        # 7. Scaling numerical features
        scaler = StandardScaler()
        num_cols = X_num.columns
        X[num_cols] = scaler.fit_transform(X[num_cols])
        data = X
        data[target] = y.values
        
        # Save all
        os.makedirs('models/transforms', exist_ok=True)
        processor = {'encoder': encoder, 'imputer': imputer, 'scaler': scaler, 'clipper': clipper}
        joblib.dump(processor, 'models/transforms/processor.pkl')
    else:
        # Load the processor
        processor = joblib.load('models/transforms/processor.pkl')
        encoder = processor['encoder']
        imputer = processor['imputer']
        scaler = processor['scaler']
        clipper = processor['clipper']
        
        X = data
        X_num = X.select_dtypes(include=[np.number])
        X_cat = X.select_dtypes(include=['object'])
        
        # 4. Handling missing values
        X_imputed = pd.DataFrame(imputer.transform(X_num), columns=X_num.columns)
        X = pd.concat([X_imputed.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
        
        # 5. Dealing with outliers
        X_out = clipper.transform(X_imputed)
        
        # 6. Encoding categorical variables
        state_encoded = pd.DataFrame(encoder.transform(X_cat), columns=encoder.get_feature_names_out(['region']))
        X = pd.concat([X_out.reset_index(drop=True), state_encoded.reset_index(drop=True)], axis=1)
        
        # 7. Scaling numerical features
        num_cols = X_num.columns
        X[num_cols] = scaler.transform(X[num_cols])
        data = X
        
    return data

train_p = preprocess_data(train, True)
X_train = train_p.drop(columns=['price'])
y_train = train_p['price'].rename('price')

X_test = test.drop(columns=['Price'])
y_test = test['Price'].rename('price')
X_test = preprocess_data(X_test, False)

print(f"Shapes: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
sns.pairplot(pd.concat([X_train.drop(columns=['region_Midwest', 'region_Northeast', 'region_South',
       'region_Territories', 'region_West']), y_train], axis=1), diag_kind='kde', plot_kws={'alpha':0.5, 'color':'tab:blue'})
plt.show()

# ========================================= 
## Modelisation
# ========================================= 

### ================Linear Regression================
lr = LinearRegression()
lr.fit(X_train, y_train)

scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("*******Linear Regression CV*******")
print(f"Mean MSE: {-scores.mean():.2f}\nStd: {scores.std():.2f}")

y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("LinearRegression Metrics")
print(f"Test MSE: {mse_lr:.2f}\nTest R2: {r2_lr:.4f}")

### ================Bayesian Ridge Regression================
brr = BayesianRidge()
brr.fit(X_train, y_train)

scores = cross_val_score(brr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("*******Bayesian Ridge Regression CV*******")
print(f"Mean MSE: {-scores.mean():.2f}\nStd: {scores.std():.2f}")

y_pred_brr = brr.predict(X_test)
mse_brr = mean_squared_error(y_test, y_pred_brr)
r2_brr = r2_score(y_test, y_pred_brr)
print("BayesianRidge Metrics")
print(f"Test MSE: {mse_brr:.2f}\nTest R2: {r2_brr:.4f}")

### ================Random Forest Regressor================
# # Finding best hyperparameters for the RandomForestRegressor model
# random_rf = RandomForestRegressor(random_state=SEED)

# param_dist = {
#     'n_estimators': [100, 200, 300, 400],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2', None]
# }
# rand_search_rf = RandomizedSearchCV(random_rf, param_distributions=param_dist, n_iter=20, cv=5, scoring='neg_mean_squared_error', random_state=SEED)
# rand_search_rf.fit(X_train, y_train)

# print(f"Best hyperparameters for RandomForestRegressor: {rand_search_rf.best_params_}")
# print(f"Best CV MSE: {-rand_search_rf.best_score_:.2f}")

# Training the model with the best hyperparameters
# rf_model = rand_search_rf.best_estimator_
best_params = {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': 20}

rf_model = RandomForestRegressor(**best_params, random_state=SEED)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("RandomForestRegressor Metrics")
print(f"Test MSE: {mse_rf:.2f}\nTest R2: {r2_rf:.4f}")

### ================Gradient Boosting Regressor================
# # Finding best hyperparameters
# random_xgb = XGBRegressor(objective='reg:squarederror', random_state=SEED)
# param_dist = {
#     'n_estimators': [100, 200, 300, 400],
#     'max_depth': [3, 5, 7, 10],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'reg_alpha': [0, 0.01, 0.1, 1],
#     'reg_lambda': [1, 1.5, 2, 3]
# }
# rand_search_xgb = RandomizedSearchCV(random_xgb, param_distributions=param_dist, n_iter=20, cv=5, scoring='neg_mean_squared_error', random_state=SEED)
# rand_search_xgb.fit(X_train, y_train)

# print(f"Best hyperparameters for XGBRegressor: {rand_search_xgb.best_params_}")
# print(f"Best CV MSE: {-rand_search_xgb.best_score_:.2f}")

# Training the model with the best hyperparameters
# xgb_model = rand_search_xgb.best_estimator_
best_params = {'subsample': 1.0, 'reg_lambda': 2, 'reg_alpha': 1, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 0.6}

xgb_model = XGBRegressor(objective='reg:squarederror', random_state=SEED, **best_params)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("XGBRegressor Metrics")
print(f"Test MSE: {mse_xgb:.2f}\nTest R2: {r2_xgb:.4f}")

### ================KNN================
# Finding best hyperparameters for the KNeighborsRegressor model
random_knn = KNeighborsRegressor()
param_dist = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
}
rand_search_knn = RandomizedSearchCV(random_knn, param_distributions=param_dist, n_iter=20, cv=5, scoring='neg_mean_squared_error', random_state=SEED)
rand_search_knn.fit(X_train, y_train)

print(f"Best hyperparameters for KNeighborsRegressor: {rand_search_knn.best_params_}")
print(f"Best CV MSE: {-rand_search_knn.best_score_:.2f}")

# Training the model with the best hyperparameters
knn_model = rand_search_knn.best_estimator_
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)
print("KNeighborsRegressor Metrics")
print(f"Test MSE: {mse_knn:.2f}\nTest R2: {r2_knn:.4f}")

# ========================================= 
## Performance comparison
# ========================================= 
display(Markdown(f"""
| Models on val            | Test MSE | Test R2 | Rank |
|-------------------|---------:|--------:|-----:|
| XGBoost           | {mse_xgb:.4f} | {r2_xgb:.3f} | 1 |
| Bayesian Ridge    | {mse_brr:.4f} | {r2_brr:.3f} | 2 |
| Linear Regression | {mse_lr:.4f} | {r2_lr:.3f} | 3 |
| Random Forest     | {mse_rf:.4f} | {r2_rf:.3f} | 4 |
| K-Nearest Neighbors | {mse_knn:.4f} | {r2_knn:.3f} | 5 |
"""))

# ========================================= 
## Final evaluation
# ========================================= 
eval_features = pd.read_csv('eval/eval.csv')
eval_labels = pd.read_csv('eval/labels.csv')

X_eval = preprocess_data(eval_features, False)
y_eval = eval_labels['Price'].rename('price')
print(eval_features.shape, eval_labels.shape, X_eval.shape, y_eval.shape)

# Predict with the 3 best models
y_pred_xgb_eval = xgb_model.predict(X_eval)
y_pred_brr_eval = brr.predict(X_eval)
y_pred_lr_eval = lr.predict(X_eval)

# Evaluate the models
mse_xgb_eval = mean_squared_error(y_eval, y_pred_xgb_eval)
r2_xgb_eval = r2_score(y_eval, y_pred_xgb_eval)
mse_brr_eval = mean_squared_error(y_eval, y_pred_brr_eval)
r2_brr_eval = r2_score(y_eval, y_pred_brr_eval)
mse_lr_eval = mean_squared_error(y_eval, y_pred_lr_eval)
r2_lr_eval = r2_score(y_eval, y_pred_lr_eval)

display(Markdown(f"""
| Models on eval           | Eval MSE | Eval R2 |
|-------------------|---------:|--------:|
| XGBoost           | {mse_xgb_eval:.4f} | {r2_xgb_eval:.3f} |
| Bayesian Ridge    | {mse_brr_eval:.4f} | {r2_brr_eval:.3f} |
| Linear Regression | {mse_lr_eval:.4f} | {r2_lr_eval:.3f} |
"""))

plt.figure(figsize=(12, 5))
plt.scatter(y_eval, y_pred_xgb_eval, color='tab:blue', alpha=0.6)
plt.plot([y_eval.min(), y_eval.max()], [y_eval.min(), y_eval.max()], 'k--', lw=1.5, color='red')
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices (XGBoost)')
plt.title('Actual vs Predicted house prices (XGBoost)')
plt.grid(axis='y', alpha=0.3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.show()

plt.figure(figsize=(12, 5))
plt.title('Distribution of actual vs Predicted prices')
sns.kdeplot(y_eval, label='Actual prices', color='tab:blue', fill=False, alpha=0.5)
sns.kdeplot(y_pred_xgb_eval, label='Predicted prices (XGBoost)', color='orange', fill=False, alpha=0.5)
sns.kdeplot(y_pred_brr_eval, label='Predicted prices (Bayesian Ridge)', color='green', fill=False, alpha=0.5)
sns.kdeplot(y_pred_lr_eval, label='Predicted prices (Linear Regression)', color='purple', fill=False, alpha=0.5)
plt.xlabel('price')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.show()

# XGB feature importance
xgb_importance = xgb_model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': xgb_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', color='tab:blue', edgecolor='none')
plt.title('XGBoost feature importance')
plt.xlabel('Importance score')
plt.grid(axis='x', alpha=0.3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.show()