import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from BinaryPSO import BinaryPSO
import os
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"
# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Encode labels if necessary
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

np.random.seed(42)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n_features = X.shape[1]

def fitness_function(particles):
    scores = []
    for particle in particles:
        num_features = np.count_nonzero(particle)
        if num_features == 0:
            scores.append(1.0)
            continue
        selected_features = X_train.iloc[:, np.array(particle).astype(bool)]
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(selected_features, y_train)
        predictions = clf.predict(X_test.iloc[:, np.array(particle).astype(bool)])
        acc = accuracy_score(y_test, predictions)
        penalty = 0.01 * num_features / n_features * (num_features > 20)
        scores.append(1 - acc + penalty)
    return np.array(scores)

options = {'c1': 2, 'c2': 2, 'w': 0.9, 'patience': 20}

# Run optimization
cost, pos, convergence_curve = BinaryPSO(
    n_particles=25,
    dimensions=n_features,
    options=options,
    fitness_function=fitness_function,
    iters=50
)

# Results
selected_features = X.columns[[True if x else False for x in pos]]
print(f"\nSelected features: {selected_features.tolist()}")
print(f"Best Raw position: {pos}")
print(f"Number of selected features: {np.count_nonzero(pos)}")
print(f"Best score: {1 - cost:.4f}")
print(f"Best accuracy (inferred from score): {1 - cost + 0.01*(np.count_nonzero(pos)/30)*(np.count_nonzero(pos) > 20):.4f}")
# Plot convergence curve
plt.figure(figsize=(10, 6))
plt.plot(convergence_curve, marker='o')
plt.title("Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()
