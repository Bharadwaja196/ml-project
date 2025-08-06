
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Load dataset and model
df = pd.read_csv("data/iris_v2.csv")  # You can switch to iris_v1.csv if needed
model = joblib.load("models/production_model.pkl")

X = df.drop("species", axis=1)
y = df["species"]

# Train/test split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix for Production Model")
plt.show()
