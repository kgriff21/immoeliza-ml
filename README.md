# immoeliza-ml
Machine learning assignment for the ImmoEliza project

## Regression

    Repository: immo-eliza-ml
    Type: Consolidation
    Duration: 5 days
    Deadline: 05/11/2024 4:00 PM
    Show and tell: 05/11/2024 4:00 - 5:00 PM
    Team: solo

### Learning Objectives


    Be able to preprocess data for machine learning.
    Be able to apply a linear regression in a real-life context.
    Be able to explore machine learning models for regression.
    Be able to evaluate the performance of a model
    (Optional) Be able to apply hyperparameter tuning and cross-validation.

### The Mission
The real estate company Immo Eliza asked you to create a machine learning model to predict prices of real estate properties in Belgium.
After the scraping, cleaning and analyzing, you are ready to preprocess the data and finally build a performant machine learning model!

### Steps
#### Data preprocessing

My own scraped dataset was used from the analysis part, under cleaned_data_with_region_and_price_per_m2.csv
Notes about the data:

Approximately 10,000 property listing were scraped. The region, province and price per m^2 were added as additional columns during the data visualization. Key features were kept such as number of bedrooms, size, property type and province.

Each property has a unique identifier id.

The target variable is price.

The data was further processed to handle Nan and dropped irrelevant features such as 'Building conditition'
as an additional column was added 'Encoded Building Condition'. This was done with the LabelEncoder() class.
The columns for Terrace and Garden area were dropped as they included many Nan values.
Categorical features were turned into binary/numeric ('house'/'apartment', 'province') via the one_hot_encode_columns function.

### Requirements

Dependencies: List required Python libraries such as pandas, scikit-learn, numpy, matplotlib, and seaborn. You can install a requirements.txt file for easy installation.

### Methodology

#### Data splitting

The data was split into a training and test set, 80-20 respectively. These were assigned variables X_train, X_test, y_train, y_test.

#### Model selection

Linear regression was selected as the first model to give us a good baseline performance of our data. As the score was not very high, other models were explored such as the random forest model.
This improved the score slightly but was ideally not high enough. I tried to install the XGBoost library as others found this to give them the best score, but this was not compatible with my Macbook computer.

### Feature engineering

Standardizing is important in models like linear regression, logistic regression, SVM, k-nearest neighbors (KNN), and neural networks as they are sensitive to the scale of the input features. Scale means the 
range and distribution of values for different features (columns) in your dataset. e.g. Size in sq meters could range from 20-500, while number of rooms might range from 1 to 10. StandardScaler was used for the linear regression model. It
is defined as the function def standardize_data_features(df), taking the dataframe as a parameter.

### Model training
In this script, the linear regression and random forest models with the scikit-learn library were used.

A fit line was calculated based on the linear regression training model. This finds the m (slope) and c (coefficient, y-intercept) on training parameters.
The test data was then tested and the coefficient of determination (R^2) score calculated. The linear regression function .predict() was used on the X_test

### Evaluation

The Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and R^2 Score was used to assess model's performance.

Mean squared error takes the true values (y_test) and predicted values (y_pred), calcualtes the error for each prediction, squares it, then computes the mean.

RMSE the squared of MSE and is in the same units as the original target variable (euros).

Additional parameters were included for the Random Forest model such as:

n-estimators: This controls the number of trees in the random forest. More trees leads to better generalization (less overfitting) but increases computation time.

max_features: This parameter limits the maximum number of features that can be considered for splitting at each node in the decision trees. The model will try using e.g 8, 12, and 20 features for the splitting decision.
If this is too high, it can lead to overfitting, too low, underfitting. need to find balance for this parameter.

max_depth: This indicates the depths of the trees. Deeper trees capture more complex relationships, but they can overfit if too deep.

The scoring metric used is negative mean squared error (scoring='neg_mean_squared_error'), which means the grid search will choose the parameters that yield the lowest MSE (i.e., highest negative value). 
By making it negative, the grid search algorithm can treat it like a value it wants to maximize.

|                         | Linear Regression | Random Forest Regression                |
|-------------------------|-------------------|-----------------------------------------|
| Root Mean Squared Error | 137521.76         | 127759.18                               |
| Mean Equared Error      | 18912234150.1     | 16322408974.34                          |
| R^2 Score               | 0.49              | 0.56, after additional parameters 0.57  |

### How to run
To run the script, install the necessary libraries via requirement.txt and execute the main.py file.