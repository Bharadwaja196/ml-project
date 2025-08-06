
import pandas as pd
import sys
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset_path = sys.argv[1]
model_output = sys.argv[2]
metrics_output = sys.argv[3]

df = pd.read_csv(dataset_path)
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

os.makedirs(os.path.dirname(model_output), exist_ok=True)
joblib.dump(model, model_output)

with open(metrics_output, "w") as f:
    f.write(str(accuracy))

print(f"Trained on: {dataset_path}, Accuracy: {accuracy}")
