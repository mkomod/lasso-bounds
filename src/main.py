import itertools
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from datetime import datetime

from sklearn import linear_model


def get_longest_element(x):
    longest_el = x[0]
    for el in x:
        if len(el) > len(longest_el):
            longest_el = el
    return longest_el


def normalise_matrix(X):
    col_means = np.mean(X, axis=0)
    X = X - col_means

    col_norm = np.linalg.norm(X, axis=0)
    X = X / col_norm

    X = X * np.sqrt(X.shape[0])

    return X


def algorithm_one(n, p, s, sig, rho):
    mean = np.zeros(p)
    cov = np.diag(np.ones(p) - rho) + rho
    X = np.random.multivariate_normal(mean, cov, n)
    beta = np.concatenate((np.ones(s), np.zeros(p - s)))
    epsilon = np.random.normal(0, sig, n)
    Y = X.dot(beta) + epsilon

    X_norm = normalise_matrix(X)

    return (X, beta, X_norm, Y)


def algorithm_two(X, beta, eta):
    """
    Following algorithm 2 - takes matrix from algorithm 1, extends it using parameter eta.

    Also reshapes beta to fit new matrix
    """
    n, p = X.shape
    N_mean = np.zeros(p)
    X = X.copy()

    for j in range(0, p - 1):
        N = np.random.multivariate_normal(N_mean, np.eye(p), n)
        X = np.concatenate((X, X[:, :p] + eta * N), axis=1)

    beta = np.concatenate((beta, np.zeros(p ** 2 - p)))

    X_norm = normalise_matrix(X)

    assert X.shape[0] == n and X.shape[1] == p ** 2

    return (X, beta, X_norm)


def run_experiments(
    params, lambdas, Nexp, meta_exp_idx, tot_meta_exp, prediction_errors, min_lambdas
):
    n, p, s, sig, rho, eta = params
    for exp_num in range(Nexp):
        print(
            f"\r eta {meta_exp_idx+1} / {tot_meta_exp}| experiment {exp_num+1} / {Nexp}",
            end="",
            flush=True,
        )
        X, beta, X_norm, Y = algorithm_one(n, p, s, sig, rho)

        if eta:
            _, beta, X_norm = algorithm_two(X, beta, eta)

        for lam_idx, lam in enumerate(lambdas):
            beta_hat = fit_lasso(X_norm, Y, lam)
            prediction_errors[lam_idx, exp_num] = np.sum(
                np.power(np.dot(X_norm, beta_hat - beta), 2)
            )
        min_lambdas[exp_num] = lambdas[np.argmin(prediction_errors[:, exp_num])]
    return min_lambdas, prediction_errors


def fit_lasso(X, y, lam):
    """
    Fits LASSO with LARS
    """
    model = linear_model.LassoLars(alpha=lam, fit_intercept=False, normalize=True)
    model.fit(X, y)
    beta_hat = model.coef_
    return beta_hat


def main(ns=[20], ps=[40], ss=[4], sigs=[1], rhos=[0], etas=[0], Nexp=5):
    print("Starting")

    # Parse which param you are varying
    var_label = "N/A"
    title_str = "Control"
    if len(sys.argv) > 1:
        param_given = sys.argv[1]
        values_given = sys.argv[2:]
        if param_given == "ns":
            ns = [int(x) for x in values_given]
            var_label = "n"
            title_str = r"Parameters: p={}, s={}, $\sigma$={}, $\rho$={}"
        elif param_given == "ps":
            ps = [int(x) for x in values_given]
            var_label = "p"
            title_str = r"Parameters: n={}, s={}, $\sigma$={}, $\rho$={}"
        elif param_given == "ss":
            ss = [int(x) for x in values_given]
            var_label = "s"
            title_str = r"Parameters: n={}, p={}, $\sigma$={}, $\rho$={}"
        elif param_given == "sigs":
            sigs = [float(x) for x in values_given]
            var_label = "$\sigma$"
            title_str = r"Parameters: n={}, p={}, s={}, $\rho$={}"
        elif param_given == "rhos":
            rhos = [float(x) for x in values_given]
            var_label = "$\sigma$"
            title_str = r"Parameters: n={}, p={}, s={}, $\sigma$={}"
        elif param_given == "etas":
            etas = [float(x) for x in values_given]
            var_label = "$\eta$"
            title_str = r"Parameters: n={}, p={}, s={}, $\sigma$={}, $\rho$={}"
        else:
            raise (ValueError("Invalid param to vary"))

    # Set ranges to experiment over for params of interest
    lambdas = np.linspace(0.001, 0.7, 100)
    orig_params = [ns, ps, ss, sigs, rhos, etas]
    param_combos = list(itertools.product(*orig_params))

    # Allocate space for results
    no_meta_exp = len(param_combos)
    mean_prediction_errors = np.zeros((len(lambdas), no_meta_exp))
    mean_min_lambdas = np.zeros((no_meta_exp))

    # Reusable data stuctures for temporary working
    prediction_errors = np.zeros((len(lambdas), Nexp))
    min_lambdas = np.zeros((Nexp))

    # Run experiments
    for meta_exp_idx, params in enumerate(param_combos):
        run_experiments(
            params,
            lambdas,
            Nexp,
            meta_exp_idx,
            no_meta_exp,
            prediction_errors,
            min_lambdas,
        )
        mean_prediction_errors[:, meta_exp_idx] = np.mean(prediction_errors, axis=1)
        mean_min_lambdas[meta_exp_idx] = np.mean(min_lambdas)

    print("\nDone experiments")

    print("Plotting...")
    # Plot Results
    varied_param = get_longest_element(orig_params)
    stable_params = [y[0] for y in orig_params if y != varied_param]
    for meta_exp_idx, val in enumerate(varied_param):

        green = 1 - (no_meta_exp - meta_exp_idx) / no_meta_exp

        if meta_exp_idx == 0:
            label = f"Control"
            c = "b"
        else:
            label = f"{var_label}={val}"
            # Designed to move from red to yellow by starting at red and adding green
            c = (1, green, 0)

        plt.plot(
            lambdas,
            mean_prediction_errors[:, meta_exp_idx],
            label=label,
            color=c,
        )
        plt.axvline(mean_min_lambdas[meta_exp_idx], color=c)

    # plt.axvline(np.sqrt(2) * mean_min_lambdas[0], c="k", ls="--", alpha=0.5)

    title_str = (
        title_str.format(*stable_params)
        .replace(".", "_")
        .replace("\\", "")
        .replace("$", "")
        .replace(" ", "")
        .replace("Parameters:", "")
    )
    plt.title(title_str)
    plt.xlabel("$\lambda$")
    plt.ylabel("Prediction Error")
    plt.legend(loc="lower right")

    print("Saving...")
    # Save your stuff
    plt.savefig(f"figs/{title_str}")
    # Save results if you wanna mess around with plotting later
    pickle.dump(
        (mean_min_lambdas, mean_prediction_errors, params, Nexp, lambdas),
        open(f"results/{title_str}.p", "wb"),
    )

    # Lets see the money
    plt.show()


if __name__ == "__main__":
    main()