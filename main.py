import pandas as pd
import numpy as np
import joblib
from statistics import mean
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to visualize feature importance
# Function to visualize feature importance
def plot_feature_importance(model, feature_names):
    # Filter out locality-related features
    filtered_indices = [i for i, feature in enumerate(feature_names) if not feature.startswith("Locality_")]
    filtered_importances = model.feature_importances_[filtered_indices]
    filtered_features = [feature_names[i] for i in filtered_indices]
    sorted_indices = np.argsort(filtered_importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh([filtered_features[i] for i in sorted_indices], filtered_importances[sorted_indices], color="skyblue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance (Excluding Locality)")
    plt.gca().invert_yaxis()
    plt.show()

# Function to visualize actual vs predicted values
def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.show()

# Function to plot residuals
def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

# Function to encode building condition
def encode_building_condition(df):
    hierarchy = ['To restore', 'To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']
    building_condition_column = df[['Building condition']]
    encoder = OrdinalEncoder(categories=[hierarchy])
    encoded_building_condition = encoder.fit_transform(building_condition_column)
    df['Encoded Building Condition'] = encoded_building_condition.ravel()
    df = df.drop(['Building condition'], axis=1)
    return df

# Function to one-hot encode locality
def one_hot_encode_locality(df):
    df['Locality Prefix'] = df['Locality data'].astype(str).str[:2]
    locality_encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_locality = locality_encoder.fit_transform(df[['Locality Prefix']])
    locality_categories = [f'Locality_{cat}' for cat in locality_encoder.categories_[0][1:]]
    encoded_locality_df = pd.DataFrame(encoded_locality, columns=locality_categories, index=df.index)
    df = pd.concat([df, encoded_locality_df], axis=1)
    df = df.drop(['Locality data', 'Locality Prefix'], axis=1)
    joblib.dump(locality_encoder, 'models/locality_encoder.joblib')
    return df

# Function to one-hot encode columns
def one_hot_encode_columns(df, columns_to_encode):
    for column in columns_to_encode:
        column_data = df[[column]]
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_data = encoder.fit_transform(column_data)
        categories = encoder.categories_[0][1:]
        encoded_df = pd.DataFrame(encoded_data, columns=categories, index=df.index)
        df = pd.concat([df, encoded_df], axis=1)
        df = df.drop(column, axis=1)
    return df

# Function to drop irrelevant columns
def drop_irrelevant_columns(df, columns_to_drop):
    clean_df = df.drop(columns_to_drop, axis=1)
    return clean_df

# Function to preprocess data
def preprocess_data(df):
    df = drop_irrelevant_columns(df, ['Property ID', 'Open fire', 'Unnamed: 0', 'Region',
                                      'Price per m²', 'Terrace surface m²', 'Garden area m²', 'Province'])
    df = encode_building_condition(df)
    df = one_hot_encode_locality(df)
    columns_to_encode = ['Property type']
    df_clean = one_hot_encode_columns(df, columns_to_encode)
    df_clean['Price'] = np.log(df_clean['Price'] + 1)
    pd.set_option('display.width', 0)  # Automatically adjusts to the console width
    pd.set_option('display.max_columns', None)
    print(df_clean.head())
    return df_clean

# Function to split dataset
def split_dataset(df):
    y = df['Price']
    X = df.drop(columns=['Price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)
    return X_train, X_test, y_train, y_test

# Integrate visualizations into your random_forest_model function
def random_forest_model(X_train, X_test, y_train, y_test, feature_names):
    regressor_forest = RandomForestRegressor(n_estimators=200, random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4],
        'bootstrap': [True]
    }
    grid_search = GridSearchCV(estimator=regressor_forest, param_grid=param_grid, cv=3,
                               scoring='neg_mean_squared_error', n_jobs=-1, error_score='raise')
    grid_search.fit(X_train, y_train)
    best_forest = grid_search.best_estimator_
    print(f"Random Forest Regression score after additional parameters: {round(best_forest.score(X_test, y_test), 2)}")
    y_pred = best_forest.predict(X_test)
    y_pred = np.exp(y_pred) - 1
    mse_forest = mean_squared_error(np.exp(y_test) - 1, y_pred)
    root_mse = np.sqrt(mse_forest)
    print("Random Forest Regression Mean Squared Error:", mse_forest.round(2))
    print(f'Random Forest Regression Root Mean Squared Error: {root_mse.round(2)}')
    joblib.dump(best_forest, 'models/trained_rf_model.joblib')

    # Visualizations
    plot_feature_importance(best_forest, feature_names)
    plot_actual_vs_predicted(np.exp(y_test) - 1, y_pred)
    plot_residuals(np.exp(y_test) - 1, y_pred)

    return best_forest

# Read dataset and preprocess
df = pd.read_csv("Data/cleaned_data_with_region_and_price_per_m2.csv")
df_clean = preprocess_data(df)

# Split dataset
X_train, X_test, y_train, y_test = split_dataset(df_clean)
print(f"X_TRAIN columns: {X_train.columns}")
# Update your call to random_forest_model to include feature_names
feature_names = X_train.columns
best_model = random_forest_model(X_train, X_test, y_train, y_test, feature_names)

