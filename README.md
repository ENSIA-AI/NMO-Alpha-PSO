
# Feature Selection for Classification using Particle Swarm Optimization (PSO)

## Overview
This project tackles the problem of feature selection for classification tasks using Binary Particle Swarm Optimization (BPSO). We aim to find the optimal subset of features from the Breast Cancer Wisconsin Diagnostic Dataset to maximize classification accuracy while minimizing model complexity.

## Approach

In this project, Particle Swarm Optimization (PSO) is used to optimize the selection of features from a given dataset. The goal is to find a subset of features that results in the highest classification accuracy. The optimization problem is formulated as a maximization task where the accuracy is the fitness function, and a binary vector represents the selected features.

### Problem Definition
The objective is to find the optimal subset of features that maximizes the classification accuracy. The fitness function is defined as:

```
Accuracy = (TP + TN) / (TN + TP + FP + FN)
```

Where:
- TP: True Positives
- TN: True Negatives
- FP: False Positives
- FN: False Negatives

The optimization problem is formulated as:
```
S(x) =  1 - f(x) + 0.01\cdot \frac{\text{num\_features(x)}}{\text{total\_features}} \cdot \mathbbm{1}_{num\_features(x) > 20}(x)
```

### PSO Overview

PSO is a computational method inspired by swarm intelligence. The algorithm works by updating candidate solutions iteratively to refine their quality according to a specified fitness function. Each particle represents a solution in the problem space and moves towards the optimal solution based on personal and collective experience.

### Binary PSO (BPSO)

BPSO adapts PSO to binary search spaces. In this case, each particle’s position is represented by a binary vector, where each element corresponds to whether a feature is selected or not.

## Implementation

### Programming Environment
- **Language**: Python
- **Libraries**: NumPy, scikit-learn

### Fitness Function
The fitness function includes:
1. The accuracy of the classifier on the test data.
2. A penalty for the number of selected features to avoid selecting too many features.

### Classifier Used
- **Random Forest** with 10 estimators using the Gini impurity criterion.

## Results and Analysis

### Performance Metrics
- **Best Accuracy Achieved**: 99.42%
- **Number of Iterations**: 45
- **Optimization Time**: 17.61 seconds
- **Early Stopping**: Triggered due to lack of improvement

### Convergence Behavior
The accuracy improved rapidly in the early iterations, and after reaching a plateau, the algorithm converged to a near-optimal subset of features.

![Convergence Curve](/latex_src/convergence_curve.png)

### Feature Selection Frequency
The distribution of feature selections across particles showed that PSO efficiently explored the feature space without favoring any particular feature excessively.

![Feature Selection Frequency](/latex_src/feature_selection_frequency.png)

### Comparison of Accuracy with Different Swarm Sizes and Iteration Counts

| Iterations | Swarm Size | 10    | 25    | 50    | 100   |
|-------------|------------|-------|-------|-------|-------|
| **10**      |           | 0.9766 | 0.9766 | 0.9766 | 0.9796 |
| **20**      |           | 0.9883 | 0.9825 | 0.9942 | 0.9883 |
| **50**      |           | 0.9883 | 0.9942 | 1.0000 | 1.0000 |
| **100**     |           | 0.9942 | 0.9883 | 0.9942 | 1.0000 |

### Optimization Time
The optimization process took 17.61 seconds, making PSO an efficient method for feature selection.

### Summary of Observations
- **Rapid Convergence**: PSO showed rapid accuracy improvements in the early iterations.
- **Impact of Swarm Size and Iterations**: Larger swarm sizes and more iterations consistently led to better performance.
- **Efficiency**: The optimization was efficient, completing in under 20 seconds with early stopping.

## Challenges

- **Sub-optimal Convergence**: PSO is prone to converging to local optima.
- **Computational Time**: Fitness function evaluations can be time-consuming.
- **Parameter Sensitivity**: PSO's performance is sensitive to parameter choices.

## Conclusion

BPSO was effective in selecting a near-optimal subset of features for classification. The method achieved high accuracy (up to 99.42%) with a reasonable computational cost. However, challenges such as convergence to local optima and parameter sensitivity remain, which could be addressed in future work.

## References

- Blum, M., et al. "NP-complete problems for networks of processors", 1992.
- Kohavi, R., and John, G.H. "Wrappers for Feature Subset Selection", 1997.
- Pereira, G. "Particle Swarm Optimization", 2011.
- Cheng, S., et al. "A quarter century of particle swarm optimization", 2018.
- Sengupta, S., et al. "Particle swarm optimization: a survey of historical and recent developments with hybridization perspectives", 2018.
- Zain, I.F.M., and Shin, S.Y. "Distributed localization for wireless sensor networks using binary particle swarm optimization", 2014.

---

## Appendix

### Code Snippets

#### Velocity and Position Update

```python
velocities[i][j] = (w * velocities[i][j] +
                    c1 * r1 * (pbest[i][j] - swarm[i][j]) +
                    c2 * r2 * (gbest[j] - swarm[i][j]))
swarm[i][j] = 1 if random.random() < sigmoid(velocities[i][j]) else 0
```

#### Minimization Objective Function

```python
acc = accuracy_score(y_test, predictions)
penalty = 0.001 * curr_features / all_features * (curr_features > 17)
score = 1 - acc + penalty
```

### Full Python Code Repository
[GitHub Repository Link](https://github.com/ENSIA-AI/NMO-Alpha-PSO)
