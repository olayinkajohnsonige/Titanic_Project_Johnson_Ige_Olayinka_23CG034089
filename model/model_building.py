import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Dataset
df = pd.read_csv('train.csv')

# 2. Feature Selection (Selecting 5 features)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
X = df[features]
y = df['Survived']

# 3. Preprocessing
# a. Handling missing values
X['Age'] = X['Age'].fillna(X['Age'].median())
X['Fare'] = X['Fare'].fillna(X['Fare'].median())

# b. Encoding categorical variables (Sex: male/female to 0/1)
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])

# c. Feature Scaling (Recommended for Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Implement Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, predictions))

# 7. Save Model and Scaler (You need the scaler to process new inputs!)
joblib.dump(model, 'titanic_survival_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model and preprocessing objects saved successfully.")