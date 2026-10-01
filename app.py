import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset (Kaggle: Cervical Cancer Risk Dataset)
df = pd.read_csv('cervical_cancer.csv')

# 2. Data Preprocessing
df = df.replace('?', pd.NA).dropna()
X = df.drop('Biopsy', axis=1)
y = df['Biopsy']
X = pd.get_dummies(X)

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# Save model for deployment
import joblib
joblib.dump(model, 'cervical_model.pkl')
print("Model saved as cervical_model.pkl")
