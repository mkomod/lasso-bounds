import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

def generate_data(n, p, s, sig, rho):
    """
    Generates data according to algorithm 1 from
    """
    beta_0 = np.concatenate((np.ones(s), np.zeros(p-s)))

    mean = np.zeros(p)
    cov = np.diag(np.ones(p) - rho) + rho
    X = np.random.multivariate_normal(mean, cov, n)
    # Note this is still not normalised

    epsilon = np.random.normal(0, sig, n)
    Y = X.dot(beta_0) + epsilon

    return {"X": X, "Y": Y, "beta_0": beta_0, 
            "n": n, "p": p, "s": s, "sig":sig, "rho": rho}


def fit_lasso(X, y, lam):
    """
    Fits LASSO with LARS
    """
    model = linear_model.LassoLars(alpha = lam, fit_intercept = False, 
            normalize = False)
    model.fit(X, y)
    beta_hat = model.coef_
    return beta_hat
    


lambdas = np.arange(0.001, 10.001, 0.1)
data = generate_data(20, 40, 4, 1, 0.5)
beta_0 = data.get("beta_0")
X = data.get("X")
Y = data.get("Y")

prediction_error = []
for lam in lambdas:
    beta_hat = fit_lasso(X, Y, lam)
    prediction_error += [np.sum(np.power(np.dot(X, beta_hat - beta_0), 2))]


plt.plot(prediction_error)
plt.show()

