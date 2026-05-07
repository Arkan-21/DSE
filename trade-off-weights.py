import numpy as np
import math as m

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


def ahp_weights(matrix, criteria_names=None, verbose=True):
    """
    Compute AHP weights and consistency ratio.

    Parameters
    ----------
    matrix : np.ndarray
        Pairwise comparison matrix.

    criteria_names : list[str], optional
        Names of the criteria.

    verbose : bool
        Print results if True.

    Returns
    -------
    dict
        Dictionary containing:
        - weights
        - lambda_max
        - CI
        - CR
    """

    matrix = np.array(matrix, dtype=float)

    # -----------------------------
    # Basic validation
    # -----------------------------
    n, m = matrix.shape

    if n != m:
        raise ValueError("Matrix must be square.")

    if criteria_names is not None and len(criteria_names) != n:
        raise ValueError("Number of criteria names must match matrix size.")

    if n not in RI_VALUES:
        raise ValueError(
            f"No RI value available for n={n}. "
            f"Extend RI_VALUES dictionary."
        )

    # -----------------------------
    # Eigenvalue method
    # -----------------------------
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    max_index = np.argmax(eigenvalues.real)

    lambda_max = eigenvalues[max_index].real

    principal_eigenvector = eigenvectors[:, max_index].real

    weights = principal_eigenvector / np.sum(principal_eigenvector)

    # Ensure positive weights
    weights = np.abs(weights)

    # Normalize again after abs
    weights = weights / np.sum(weights)

    # -----------------------------
    # Consistency calculations
    # -----------------------------
    CI = (lambda_max - n) / (n - 1)

    RI = RI_VALUES[n]

    if RI == 0:
        CR = 0
    else:
        CR = CI / RI

    # -----------------------------
    # Printing
    # -----------------------------
    if verbose:

        print("\n=== AHP RESULTS ===\n")

        if criteria_names is None:
            criteria_names = [f"Criterion {i+1}" for i in range(n)]

        for c, w in zip(criteria_names, weights):
            print(f"{c:<20}: {w:.4f}")

        print("\n--- Consistency ---")
        print(f"Lambda max        : {lambda_max:.4f}")
        print(f"Consistency Index : {CI:.4f}")
        print(f"Consistency Ratio : {CR:.4f}")

        if CR < 0.10:
            print("Consistency status: GOOD")
        elif CR < 0.20:
            print("Consistency status: ACCEPTABLE")
        else:
            print("Consistency status: POOR (revise comparisons)")

    # -----------------------------
    # Return results
    # -----------------------------
    return {
        "weights": weights,
        "lambda_max": lambda_max,
        "CI": CI,
        "CR": CR
    }


def entropy_weights(matrix, verbose=True):
    """
    Compute Shannon entropy weights for a full decision matrix
    using sum normalization.

    Parameters
    ----------
    matrix : array-like
        Decision matrix:
        rows = alternatives
        cols = criteria
    """

    X = np.array(matrix, dtype=float)

    m, n = X.shape

    # -------------------------------------------------
    # Step 1 — Sum normalization
    # -------------------------------------------------
    P = np.zeros((m, n))

    for j in range(n):

        col_sum = np.sum(X[:, j])

        if col_sum == 0:
            P[:, j] = 0
        else:
            P[:, j] = X[:, j] / col_sum

    # -------------------------------------------------
    # Step 2 — Entropy
    # -------------------------------------------------
    k = 1 / np.log(m)

    E = np.zeros(n)

    for j in range(n):

        entropy_sum = 0

        for i in range(m):

            p = P[i, j]

            # Avoid log(0)
            if p > 0:
                entropy_sum += p * np.log(p)

        E[j] = -k * entropy_sum

    # -------------------------------------------------
    # Step 3 — Divergence
    # -------------------------------------------------
    D = 1 - E

    # -------------------------------------------------
    # Step 4 — Final weights
    # -------------------------------------------------
    W = D / np.sum(D)

    # -------------------------------------------------
    # Step 5 - Determine variability
    # -------------------------------------------------
    CV = np.zeros(n)
    for t in range(n):
    
        CV[t] = np.std(P[:, t]) / np.mean(P[:, t])

    # -------------------------------------------------
    # Output
    # -------------------------------------------------
    if verbose:

        print("\n=== ENTROPY WEIGHTING RESULTS ===\n")

        print("Probability matrix:")
        print(P)

        print("\nEntropy values:")
        print(E)

        print("\nDivergence values:")
        print(D)

        print("\nFinal weights:")
        print(W)

        print("\nVariability:")
        print(CV)

    return {
        "probability_matrix": P,
        "entropy": E,
        "divergence": D,
        "weights": W,
        "variability": CV
    }

def final_weights(results_ahp,results_entropy, verbose=True):
    final_weights = np.zeros(len(results_entropy['variability']))
    beta_list = np.zeros(len(results_entropy['variability']))
    for i in range(len(results_entropy['variability'])):
        if results_entropy['variability'][i] < 0.1:
            beta = 0.7
        elif 0.1 <= results_entropy['variability'][i] <= 0.3:
            beta = 0.5
        elif results_entropy['variability'][i] > 0.3:
            beta = 0.3

    
        final_weights[i] = beta*results_ahp['weights'][i] + (1-beta)*results_entropy['weights'][i]
        beta_list[i] = beta

    print(final_weights, beta_list)

    return final_weights

        
# ======================================================
# EXAMPLE USAGE
# ======================================================

matrix_ahp = [
    [1.0, 0.5, 2.0, 4.0, 10.0],
    [2.0, 1.0, 2.0, 2.0, 4.0],
    [0.5, 0.5, 1.0, 2.0, 3.0],
    [0.25, 0.5, 0.5, 1.0, 2.0],
    [0.1, 0.25, 1/3, 0.5, 1.0]
]

criteria = [
    "Safety",
    "Energy density",
    "Cooling",
    "Combustibility",
    "Sustainability"
]

results_ahp = ahp_weights(matrix_ahp, criteria,verbose=False)

matrix_entropy = [
    [102, 120, 9,12,60],
    [100, 100, 5,14,45],
    [104, 140, 8,10,60]
]

results_entropy = entropy_weights(matrix_entropy,verbose=False)

results_final = final_weights(results_ahp,results_entropy)

print(results_entropy['weights'],results_ahp['weights'])











