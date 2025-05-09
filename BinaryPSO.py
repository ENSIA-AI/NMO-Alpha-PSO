import math
import random
import time
import matplotlib.pyplot as plt  # Import matplotlib for histogram plotting
random.seed(42)

def BinaryPSO(n_particles, dimensions, options, fitness_function, iters=30):
    patience = options.get('patience', 10)  # Default patience to 10 if not provided
    swarm = [[1 if i == random.randint(0, dimensions - 1) else 0 for i in range(dimensions)] for _ in range(n_particles)]
    velocities = [[random.random()*30] * dimensions for _ in range(n_particles)]
    personal_best = swarm.copy()
    personal_best_scores = [float('inf')] * n_particles
    global_best = [0] * dimensions
    global_best_score = float('inf')
    
    convergence_curve = []
    no_improvement_counter = 0  # Counter for patience

    feature_selection_count = [0] * dimensions  # Track feature selection frequency

    start_time = time.time()
    
    for iteration in range(iters):
        progress = (iteration + 1) / iters * 100
        print(f"\rProgress: [{'#' * int(progress // 2)}{' ' * (50 - int(progress // 2))}] {progress:.2f}%, gBest : {global_best_score:.2f}", end='')
        
        fitness_values = fitness_function(swarm)
        
        improved = False
        for i in range(n_particles):
            fitness = fitness_values[i]
            if fitness < personal_best_scores[i]:
                personal_best[i] = swarm[i]
                personal_best_scores[i] = fitness
            if fitness < global_best_score:
                global_best = swarm[i]
                global_best_score = fitness
                improved = True

        # Save best accuracy (1 - score) for this iteration
        convergence_curve.append(1 - global_best_score)

        if improved:
            no_improvement_counter = 0  # Reset counter if improvement occurs
        else:
            no_improvement_counter += 1  # Increment counter if no improvement

        # Early stopping if no improvement for 'patience' iterations
        if no_improvement_counter >= patience:
            print("\nEarly stopping due to no improvement.")
            break

        for i in range(n_particles):
            for j in range(dimensions):
                r1 = random.random()
                r2 = random.random()
                velocities[i][j] = (options['w'] * velocities[i][j] +
                                    options['c1'] * r1 * (personal_best[i][j] - swarm[i][j]) +
                                    options['c2'] * r2 * (global_best[j] - swarm[i][j]))
                if random.random() < 1 / (1 + math.exp(-velocities[i][j])):
                    swarm[i][j] = 1
                else:
                    swarm[i][j] = 0

                # Update feature selection count
                if swarm[i][j] == 1:
                    feature_selection_count[j] += 1

    duration = time.time() - start_time
    print(f"\nOptimization completed in {duration:.2f} seconds.")

    # Plot frequency histogram for features
    plt.bar(range(dimensions), feature_selection_count)
    plt.xlabel('Feature Index')
    plt.ylabel('Selection Frequency')
    plt.title('Feature Selection Frequency Histogram')
    plt.show()

    return global_best_score, global_best, convergence_curve
