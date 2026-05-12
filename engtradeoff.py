import numpy as np

# ======================================================
# AHP + Entropy Hybrid Trade-Off
# ======================================================

# ------------------------------------------------------
# Random Index values
# ------------------------------------------------------

RI_VALUES = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49
}

# ======================================================
# AHP FUNCTION
# ======================================================

def ahp_weights(matrix, criteria_names=None):

    matrix = np.array(matrix, dtype=float)

    n = matrix.shape[0]

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    max_index = np.argmax(eigenvalues.real)

    lambda_max = eigenvalues[max_index].real

    principal_eigenvector = eigenvectors[:, max_index].real

    weights = np.abs(principal_eigenvector)
    weights = weights / np.sum(weights)

    CI = (lambda_max - n) / (n - 1)

    RI = RI_VALUES[n]

    CR = CI / RI if RI != 0 else 0

    print("\n=== AHP RESULTS ===")

    for c, w in zip(criteria_names, weights):
        print(f"{c:<20}: {w:.4f}")

    print(f"\nConsistency Ratio: {CR:.4f}")

    return {
        "weights": weights,
        "CR": CR
    }

# ======================================================
# ENTROPY WEIGHTING
# ======================================================

def entropy_weights(matrix):

    X = np.array(matrix, dtype=float)

    m, n = X.shape

    # Sum normalization
    P = X / np.sum(X, axis=0)

    # Entropy
    k = 1 / np.log(m)

    E = np.zeros(n)

    for j in range(n):

        entropy_sum = 0

        for i in range(m):

            p = P[i, j]

            if p > 0:
                entropy_sum += p * np.log(p)

        E[j] = -k * entropy_sum

    # Divergence
    D = 1 - E

    # Entropy weights
    W = D / np.sum(D)

    # Variability
    CV = np.std(P, axis=0) / np.mean(P, axis=0)

    print("\n=== ENTROPY RESULTS ===")

    for i, w in enumerate(W):
        print(f"Criterion {i+1}: {w:.4f}")

    return {
        "weights": W,
        "variability": CV
    }

# ======================================================
# HYBRID FINAL WEIGHTS
# ======================================================

def final_weights(results_ahp, results_entropy):

    n = len(results_ahp['weights'])

    final_w = np.zeros(n)

    for i in range(n):

        cv = results_entropy['variability'][i]

        # Dynamic beta selection
        if cv < 0.1:
            beta = 1.0
        elif cv <= 0.3:
            beta = 0.8
        else:
            beta = 0.6

        final_w[i] = (
            beta * results_ahp['weights'][i]
            + (1 - beta) * results_entropy['weights'][i]
        )

    final_w = final_w / np.sum(final_w)

    print("\n=== FINAL HYBRID WEIGHTS ===")

    for i, w in enumerate(final_w):
        print(f"Criterion {i+1}: {w:.4f}")

    return final_w

# ======================================================
# CRITERIA
# ======================================================

criteria = [
    "Safety",
    "Reusability",
    "Complexity",
    "Volume/Mass",
    "Efficiency",
    "Maturity"
]

# ======================================================
# AHP PAIRWISE MATRIX
# (replace with your preferred comparisons)
# ======================================================

matrix_ahp = np.array([

    [1,   1,   3,   4,   4,   5],
    [1,   1,   3,   4,   4,   5],
    [1/3, 1/3, 1,   2,   2,   3],
    [1/4, 1/4, 1/2, 1,   1,   2],
    [1/4, 1/4, 1/2, 1,   1,   2],
    [1/5, 1/5, 1/3, 1/2, 1/2, 1]

])

# ======================================================
# DECISION MATRIX
# Rows = configurations
# Cols = criteria
# ======================================================

decision_matrix = np.array([

    # Safety Reuse Complexity Vol/Mass Efficiency Maturity

    [1, 1, 4, 5, 5, 4],  # Ext. Rocket-Ram-Glide
    [1, 2, 5, 5, 5, 5],  # Ext. Plane-Ram-Glide
    [5, 3, 3, 1, 3, 3],  # Rocket-Ram-Turbo
    [5, 5, 3, 2, 4, 4],  # Turbo-Ram-Turbo
    [3, 5, 2, 4, 3, 2],  # Turbo-Ram
    [3, 3, 1, 3, 3, 1]   # Turbo-Ram-Rocket

])

# ======================================================
# RUN WEIGHTING
# ======================================================

results_ahp = ahp_weights(matrix_ahp, criteria)

results_entropy = entropy_weights(decision_matrix)

weights = final_weights(results_ahp, results_entropy)

# ======================================================
# FINAL SCORES
# ======================================================

scores = decision_matrix @ weights

configurations = [
    "Ext. Rocket-Ram-Glide",
    "Ext. Plane-Ram-Glide",
    "Rocket-Ram-Turbo",
    "Turbo-Ram-Turbo",
    "Turbo-Ram",
    "Turbo-Ram-Rocket"
]

print("\n=== FINAL TRADE-OFF SCORES ===\n")

for c, s in zip(configurations, scores):
    print(f"{c:<30}: {s:.3f}")

best = np.argmax(scores)

print(f"\nBest configuration: {configurations[best]}")