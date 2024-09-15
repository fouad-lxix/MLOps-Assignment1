import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
data = pd.read_csv('housing.csv')

# Preprocessing
X = data.drop(columns=['median_house_value'])
y = data['median_house_value']

# Handle missing values in X
imputer_X = SimpleImputer(strategy='mean')
X = imputer_X.fit_transform(X)

# Handle missing values in y
imputer_y = SimpleImputer(strategy='mean')
y = imputer_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')