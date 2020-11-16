import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

def normalise_matrix(X):
    col_means = np.mean(X, axis = 0)
    X = X - col_means

    col_norm = np.linalg.norm(X, axis = 0)
    X = X / col_norm

    X = X * np.sqrt(X.shape[0])

    return X

def generate_data(n, p, s, sig, rho, eta):
    """
    Generates data according to algorithm 1 from
    """
    beta_01 = np.concatenate((np.ones(s), np.zeros(p-s)))
    beta_02 = np.concatenate((np.ones(s), np.zeros(p**2-s)))

    mean = np.zeros(p)
    cov = np.diag(np.ones(p) - rho) + rho
    X = np.random.multivariate_normal(mean, cov, n)
    X2 = X.copy()

    #for j in range(0, p-1):
    #    N = np.random.multivariate_normal(mean, np.eye(p), n)
    #    X2 = np.concatenate((X2, X+eta*N), axis = 1)

    epsilon = np.random.normal(0, sig, n)
    Y = X.dot(beta_01) + epsilon

    X = normalise_matrix(X)
    #X2 = normalise_matrix(X2)

    return {"X": X, "X2": X2, "Y": Y, "beta_01": beta_01, "beta_02": beta_02,
            "n": n, "p": p, "s": s, "sig":sig, "rho": rho}


def fit_lasso(X, y, lam):
    """
    Fits LASSO with LARS
    """
    model = linear_model.LassoLars(alpha = lam, fit_intercept = False, 
            normalize = True)
    model.fit(X, y)
    beta_hat = model.coef_
    return beta_hat
    

n, p, s, sig, rho, eta = 20, 40, 10, 1, 0.99, 0

print()
print("params:", n, p, s, sig, rho, eta)
print()

Nexp = 1000
lambdas = np.linspace(0.001, 0.7, 100)
prediction_error = np.zeros((len(lambdas), Nexp))
prediction_error2 = np.zeros((len(lambdas), Nexp))

min_lambdas = np.zeros(Nexp)
min_lambdas2 = np.zeros(Nexp)
min_PEs = np.zeros(Nexp)

print("starting", end = "")

for exp_num in range(Nexp):
    data = generate_data(n, p, s, sig, rho, eta)
    beta_01 = data.get("beta_01")
    beta_02 = data.get("beta_02")
    X = data.get("X")
    X2 = data.get("X2")
    Y = data.get("Y")

    for i, lam in enumerate(lambdas):
        beta_hat = fit_lasso(X, Y, lam)
        #beta_hat2 = fit_lasso(X2, Y, lam)
        
        prediction_error[i, exp_num] = np.sum(np.power(np.dot(X, beta_hat - beta_01), 2))
        #prediction_error2[i, exp_num] = np.sum(np.power(np.dot(X2, beta_hat2 - beta_02), 2))
    
    print(f"\r experiment {exp_num+1} / {Nexp}", end = "", flush = True)

    # ignore this, this is for my experiment
    pe_min = np.min(prediction_error[:, exp_num])
    min_PEs[exp_num] = pe_min

    # find the minimum lambda for this particular experiment
    lambda_min = lambdas[np.argmin(prediction_error[:, exp_num])]
    #lambda_min2 = lambdas[np.argmin(prediction_error2[:, exp_num])]

    min_lambdas[exp_num] = lambda_min
    #min_lambdas2[exp_num] = lambda_min2

print("\nDone")

# mean prediction error 
mean_pde = np.mean(prediction_error, axis = 1)
mean_pde2 = np.mean(prediction_error2, axis = 1)

lambda_min_1 = np.mean(min_lambdas)
lambda_min_2 = np.mean(min_lambdas2)

#ignore this, it's for my experiment
print()
print(f"Average Minimum Lambda: {np.mean(min_lambdas)}")
print(f"Standard Deviation: {np.std(min_lambdas)}")
print()
print(f"Average Minimum Prediction Error: {np.mean(min_PEs)}")
print(f"Standard Deviation: {np.std(min_PEs)}")

plot_graphs = False

if plot_graphs == True:
    # ALGO 1
    plt.plot(lambdas, mean_pde, label = "Algo 1", color = "b")
    #plt.fill_between(lambdas, np.quantile(prediction_error, 0.025, axis = 1), np.quantile(prediction_error, 0.975, axis = 1),
    #color = "b", alpha = .2)
    plt.axvline(lambda_min_1, color = "b")

    # ALGO 2
    plt.plot(lambdas, mean_pde2, label = "Algo 2", color = "r")
    #plt.fill_between(lambdas, np.quantile(prediction_error2, 0.025, axis = 1), np.quantile(prediction_error2, 0.975, axis = 1),
    #color = "r", alpha = .2)
    plt.axvline(lambda_min_2, color = "r")

    plt.title(f"Parameters: {(n, p, s, sig, rho, eta)}")
    plt.axvline(np.sqrt(2)*lambda_min_1, c = "k", ls = "--")
    plt.xlabel("$\lambda$")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.show()