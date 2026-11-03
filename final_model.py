
# Imports 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Read train data
df_train = pd.read_csv("DAT200-CA3/train.csv")

# Remove dubpicates
df_train = df_train.drop(['Id'], axis=1).drop_duplicates()

# Add a new feature
df_train['wine_fraction_density'] =  df_train['density'] - df_train['alcohol'] * 0.789 / 100

# Choose features
features = [
    'wine_fraction_density',
    'sulphates',
    'total sulfur dioxide',
    'volatile acidity',
    'chlorides'
]
X_train = df_train[features]
y_train = df_train['quality']

# Train model
classifier = RandomForestClassifier(
    n_estimators=300,
    max_depth=4,
    random_state=42,
    min_impurity_decrease=0.011,
    criterion='entropy',
    n_jobs=-1,
    bootstrap=True,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=10
)
classifier.fit(X_train, y_train)

# Read test data
df_test = pd.read_csv("DAT200-CA3/test.csv")
df_test['wine_fraction_density'] =  df_test['density'] - df_test['alcohol'] * 0.789 / 100

X = df_test[features]

# Make predictions
y_final = classifier.predict(X)

# Create submission file
y_test = pd.DataFrame(y_final, columns=['quality'])
y_test.index.name = "Id"
y_test[['quality']].to_csv("DAT200-CA3/submission.csv")
