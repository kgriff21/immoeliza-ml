import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display

def encode_building_condition(df):
    '''
    Encodes building condition with OrdinalEncoder, giving hierarcy ranging from 0 - 5:
    ['To restore', 'To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']
    This function also drops the column originally titled 'Building Condition'.
    Parameters:
    df (DataFrame): The original or modified DataFrame
    '''

    hierarchy = ['To restore', 'To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']
    building_condition_column = df[['Building condition']]
    # Define and fit the OrdinalEncoder
    encoder = OrdinalEncoder(categories=[hierarchy])
    encoded_building_condition = encoder.fit_transform(building_condition_column)
    # Add the transformed data back to the DataFrame
    df['Encoded Building Condition'] = encoded_building_condition.ravel()
    df = df.drop(['Building condition'], axis=1)
    # Display the updated DataFrame
    print(display(df.head(50)))
    return df


def one_hot_encode_columns(df, columns_to_encode):
    """
    One-hot encode the specified columns in the DataFrame.
    Parameters:
    df (DataFrame): The original DataFrame
    columns_to_encode (list): List of columns to be one-hot encoded
    Returns:
    DataFrame: The DataFrame with the specified columns one-hot encoded and added.
    """
    for column in columns_to_encode:
        column_data = df[[column]]
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_data = encoder.fit_transform(column_data)  # fit_transform applies the encoder
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column]))
        df = pd.concat([df, encoded_df], axis=1)
        df = df.drop(column, axis=1)
    return df  # return outside the loop to process all columns before returning updated DataFrame

def split_dataset(df):
    '''
    Split the dataset into training and test sets.
    :param: dataframe
    :return: A tuple of variables for X_train, X_test, y_train, y_test that will be used for model training
    '''
    y = df['Price'] # Target variable
    X = df.drop(columns=['Price']) # All other column features in df after dropping the target
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=41, test_size=0.2) # Get 4 parameters back
    return X_train, X_test, y_train, y_test


def standardize_data_features(df):
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = split_dataset(df)
    X_train_standardized = scaler.fit_transform(X_train)
    X_test_standardized = scaler.transform(
        X_test)  # Uses same mean and standard deviation calculated from the training set to transform the test data

    y_train_standardized = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_standardized = scaler.transform(y_test.values.reshape(-1, 1))

    X_train_standardized = pd.DataFrame(X_train_standardized, columns=X_train.columns)
    X_test_standardized = pd.DataFrame(X_test_standardized, columns=X_test.columns)

    # Check mean and standard deviation of the training data after scaling
    print("Means of standardized features (training set):", X_train_standardized.mean(axis=0).values)
    print("Standard deviations of standardized features (training set):", X_train_standardized.std(axis=0).values)
    return X_train_standardized, X_test_standardized, y_train_standardized, y_test_standardized

def fit_train_linear_regression(X_train, X_test, y_train, y_test):
    # Fit linear regression model
    lr = LinearRegression()
    # Finds the best fit line on the training model (find m (slope) and c (coefficient, y-intercept)) on training parameters
    lr.fit(X_train, y_train)
    c = lr.intercept_ # View the fitted and calculated y-intercept
    m = lr.coef_ # View the fitted and calculated slope value
    # print(m, c)
    # Train model and predict scores on test dataset
    train_score = lr.score(X_train, y_train)
    test_score = lr.score(X_test, y_test)
    print(f"Coefficient of determination (R^2) for trained data: {train_score}")
    print(f"Coefficient of determination (R^2) for test data: {test_score}")
    y_pred = lr.predict(X_test) # Generate predictions from a fitted liner regression model on new unseen data.
    # print(y_test)
    # print(y_pred)
    lr.score(X_test, y_test)
    # Calculate Mean Squared Error and R^2 Score from sklearn.metrics import
    mse = mean_squared_error(y_test, y_pred)
    root_mse = np.sqrt(mse)
    print(f"Linear regression Root Mean Squared Error: {root_mse.round(2)}")
    r2 = r2_score(y_test, y_pred)
    print("Linear regression Mean Squared Error:", mse.round(2))
    print("Linear regression R^2 Score:", round(r2, 2))


df = pd.read_csv("Data/cleaned_data_with_region_and_price_per_m2.csv")

# Encode building condition
df = encode_building_condition(df)

# Drop unnecessary columns not needed for the model
df = df.drop(['Property ID', 'Open fire', 'Unnamed: 0', 'Locality data', 'Region', 'Price per m²', 'Terrace surface m²', 'Garden area m²'], axis=1)

# Call one_hot_encode_columns() function
columns_to_encode = ['Property type', 'Province']
df = one_hot_encode_columns(df, columns_to_encode) # update the df on each iteration

# Split the dataset: Call the function and assign the returned values to variables
X_train, X_test, y_train, y_test = split_dataset(df)

X_train_standardized, X_test_standardized, y_train_standardized, y_test_standardized = standardize_data_features(df)

# Commented out standardized train and test data as did not make an impact on score.
# fit_train_linear_regression(X_train_standardized, X_test_standardized, y_train_standardized, y_test_standardized)

fit_train_linear_regression(X_train, X_test, y_train, y_test)

# Random Forest regression model
regressor_forest = RandomForestRegressor(n_estimators=200, random_state=0)
regressor_forest.fit(X_train, y_train)
print(f'Random Forest Regression score before additional parameters: {round(regressor_forest.score(X_test, y_test), 2)}')
param_grid = {'n_estimators': [30, 50, 100],
              'max_features': [8, 12, 20],
              'max_depth':[5, 6, 12]
              }
grid_search = GridSearchCV(estimator=regressor_forest, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)
best_forest = grid_search.best_estimator_
print(f"Random Forest Regression score after additional parameters: {round(best_forest.score(X_test, y_test), 2)}")
y_pred = regressor_forest.predict(X_test)
# Calculate Mean Squared Error (MSE)
mse_forest = mean_squared_error(y_test, y_pred)
root_mse = np.sqrt(mse_forest)
print("Random Forest Regression Mean Squared Error:", mse_forest.round(2))
print(f'Random Forest Regression Root Mean Squared Error: {root_mse.round(2)}')
correlation_matrix = df.corr(method='pearson')
price_correlation = correlation_matrix['Price'].sort_values(ascending=False)
print(price_correlation)