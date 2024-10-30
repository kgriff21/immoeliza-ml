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

Approximately 10,000 property listing were scraped. The region, province and price per m^2 were added as additional columns during the data visualization

Each property has a unique identifier id

The target variable is price

The data was further processed to handle Nan and dropped irrelevant features such as 'Building conditition'
as an additional column was added 'Encoded Building Condition'. This was done with the LabelEncoder() class.
The columns for Terrace and Garden area were dropped as they included many Nan values.
Categorical features were turned into binary/numeric ('house'/'apartment', 'province') via the one_hot_encode_columns function.

### Requirements

Dependencies: List required Python libraries such as pandas, scikit-learn, numpy, matplotlib, and seaborn. You can include a requirements.txt file for easy installation.
