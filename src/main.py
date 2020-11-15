import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from datetime import datetime

from sklearn import linear_model


def normalise_matrix(X):
    col_means = np.mean(X, axis=0)
    X = X - col_means

    col_norm = np.linalg.norm(X, axis=0)
    X = X / col_norm

    X = X * np.sqrt(X.shape[0])

    return X


def generate_data(n, p, s, sig, rho, eta):
    """
    Generates data according to either algorithm 1 or algorithm 2 depending on if eta is supplied.

    Returns tuple of ('real' beta, X, Y)
    """

    mean = np.zeros(p)
    cov = np.diag(np.ones(p) - rho) + rho
    X = np.random.multivariate_normal(mean, cov, n)
    orig_cols = X.shape[1]
    if eta:
        for j in range(0, p - 1):
            N = np.random.multivariate_normal(mean, np.eye(p), n)
            X = np.concatenate((X, X[:, :orig_cols] + eta * N), axis=1)

    beta = np.concatenate((np.ones(s), np.zeros(X.shape[1] - s)))
    epsilon = np.random.normal(0, sig, n)
    Y = X.dot(beta) + epsilon

    X = normalise_matrix(X)

    return (beta, X, Y)


def fit_lasso(X, y, lam):
    """
    Fits LASSO with LARS
    """
    model = linear_model.LassoLars(alpha=lam, fit_intercept=False, normalize=True)
    model.fit(X, y)
    beta_hat = model.coef_
    return beta_hat


def main(n=20, p=40, s=4, sig=1, rho=0, eta=5, Nexp=3):
    print("Starting", end="")
    # Parse any args supplied if used as a CLI
    if len(sys.argv) > 1:
        try:
            (
                n,
                p,
                s,
                sig,
                rho,
            ) = [int(x) for x in sys.argv[1:-2]]
            eta = float(sys.argv[-2])
            Nexp = int(sys.argv[-1])
        except ValueError:
            raise (
                ValueError(
                    "You did not supply all experiment values\n Need n, p, s, sigma, rho, eta, number_of_experiments"
                )
            )
    else:
        n, p, s, sig, rho, eta, Nexp = (
            n,
            p,
            s,
            sig,
            rho,
            eta,
            Nexp,
        )

    # Set ranges to experiment over for params of interest
    etas = np.linspace(0, eta, 5)
    etas = np.round(etas, 1)
    lambdas = np.linspace(0.001, 0.7, 100)

    # Allocate space for results
    mean_prediction_errors = np.zeros((len(lambdas), len(etas)))
    min_lambdas = np.zeros([len(etas)])
    quantiles = np.zeros((len(lambdas), len(etas), 2))

    prediction_errors = np.zeros((len(lambdas), Nexp))

    tot_etas = len(etas)
    for eta_idx, eta in enumerate(etas):
        for exp_num in range(Nexp):
            print(
                f"\r eta {eta_idx+1} / {tot_etas}| experiment {exp_num+1} / {Nexp}",
                end="",
                flush=True,
            )

            beta, X, Y = generate_data(n, p, s, sig, rho, eta)

            for lam_idx, lam in enumerate(lambdas):
                beta_hat = fit_lasso(X, Y, lam)
                prediction_errors[lam_idx, exp_num] = np.sum(
                    np.power(np.dot(X, beta_hat - beta), 2)
                )

        quantiles[:, eta_idx, 0] = np.quantile(prediction_errors, 0.025, axis=1)
        quantiles[:, eta_idx, 1] = np.quantile(prediction_errors, 0.975, axis=1)
        mean_prediction_errors[:, eta_idx] = np.mean(prediction_errors, axis=1)
        min_lambdas[eta_idx] = lambdas[np.argmin(mean_prediction_errors[:, eta_idx])]

    # Plot Results
    for eta_idx, eta in enumerate(etas):
        blueness = (tot_etas - eta_idx) / tot_etas
        c = (1 - blueness, 0.5, blueness)
        plt.plot(
            lambdas,
            mean_prediction_errors[:, -eta_idx],
            label=f"$\eta$={eta}",
            color=c,
        )
        plt.fill_between(
            lambdas,
            quantiles[:, -eta_idx, 0],
            quantiles[:, -eta_idx, 1],
            color=c,
            alpha=0.5,
        )
        plt.axvline(min_lambdas[-eta_idx], color=c, alpha=0.5)

    # Save results if you wanna mess around with plotting later

    pickle.dump(
        (lambdas, mean_prediction_errors, etas, n, p, s, sig, rho, eta, Nexp),
        open("numexp{Nexp}_n{n:}p{p}s{s}sig{sig}rho{rho}etarange_results.p", "wb"),
    )

    plt.title(
        r"Parameters: n={}, p={}, s={}, $\sigma$={}, $\rho$={}".format(
            n, p, s, sig, rho
        )
    )
    plt.axvline(np.sqrt(2) * min_lambdas[0], c="k", ls="--", alpha=0.5)
    plt.xlabel("$\lambda$")
    plt.ylabel("Prediction Error")
    plt.legend(loc="lower right")
    plt.savefig(f"figs/numexp{Nexp}_n{n:}p{p}s{s}sig{sig}rho{rho}etarange")
    plt.show()

    print("\nDone")


if __name__ == "__main__":
    main()