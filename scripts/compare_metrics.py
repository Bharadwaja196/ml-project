
v1 = float(open("results/metrics_v1.txt").read())
v2 = float(open("results/metrics_v2.txt").read())

if v1 > v2:
    best = "v1"
    best_model = "models/model_v1.pkl"
else:
    best = "v2"
    best_model = "models/model_v2.pkl"

print(f"Best dataset: iris_{best}, Accuracy: {max(v1, v2)}")

import shutil
shutil.copy(best_model, "models/production_model.pkl")
print("Saved best model as models/production_model.pkl")
