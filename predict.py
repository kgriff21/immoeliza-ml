import joblib
import numpy as np
import pandas as pd

# Load saved model(s)
rf_model = joblib.load("models/trained_rf_model.joblib")

# Create listing data with column names
feature_names = [
    'Number of bedrooms', 'Living area m²', 'Equipped kitchen', 'Furnished',
    'Swimming pool', 'Encoded Building Condition', 'Property type_house',
    'Province_Brussels', 'Province_East Flanders', 'Province_Flemish Brabant',
    'Province_Hainaut', 'Province_Limburg', 'Province_Liège',
    'Province_Luxembourg', 'Province_Namur', 'Province_Walloon Brabant',
    'Province_West Flanders'
]

new_listing = pd.DataFrame(
    [np.array([3, 102, 1, 0, 0, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])],
    columns=feature_names
)

# Predict the price of the new listing
predicted_price = rf_model.predict(new_listing).round(2)
print(f"Predicted price: {predicted_price}")