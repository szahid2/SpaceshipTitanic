# Sobia Zahid

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("C:/Users/shyer/Downloads/train.csv")
test_df = pd.read_csv("C:/Users/shyer/Downloads/test.csv")


def p_data(df):
    # convert categorical variables to numerical
    df['CryoSleep'] = df['CryoSleep'].map({'Yes': 1, 'No': 0, 'Unknown': -1})
    df['VIP'] = df['VIP'].map({'Yes': 1, 'No': 0})
    # Extract information from Cabin
    df['Cabin_prefix'] = df['Cabin'].str.extract(r'([A-Za-z]+)')
    df['Cabin_prefix'] = df['Cabin_prefix'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'Unknown': 0})
    return df


train_df = p_data(train_df)
test_df = p_data(test_df)

features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_prefix']
X = train_df[features]
y = train_df['Transported']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['Cabin_prefix']

numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = SimpleImputer(strategy='most_frequent')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


model = RandomForestClassifier(n_estimators=100, random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

test_predictions = pipeline.predict(test_df[features])

submission_df = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Transported": test_predictions
})

submission_df.to_csv("spaceship_titanic_submission.csv", index=False)
