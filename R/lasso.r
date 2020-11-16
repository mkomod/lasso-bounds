library(mvtnorm)
library(glmnet)
library(boot)


#' Normalise X
normalise <- function(X) {
    norms <- apply(X, 2, function(x) norm(as.matrix(x), type="F"))
    X <- t(apply(X, 1, function(x) x / norms)) * sqrt(nrow(X))
    return(X)
}


#' Generates data according to algorithm 1 from [HL12]
gen_data <- function(n, p, s, sig, rho, eta, gen_X2=TRUE) {
    if (s > p) stop("s > p")
    if (rho < 0 | rho >= 1) stop("rho must be in [0, 1)")

    S <- diag(1 - rho, p) + rho
    mu <- rep(0, p)

    X <- mvtnorm::rmvnorm(n, mean=mu, S)

    beta.0 <- c(rep(1, s), rep(0, p-s))
    beta.0.2 <- c(beta.0, rep(0, p^2 - p))
    e <- rnorm(n, 0, sd=sig)
    Y <- X %*% beta.0 + e

    X.2 <- X
    if (gen_X2) {
	for (i in 1:(p-1)) {
	    X.2 <- cbind(X.2, X + eta*rmvnorm(n, mu, diag(1, p)))
	}
	X.2 <- normalise(X.2)
    }
    X <- normalise(X)

    return(list(Y=Y, X=X, X.2=X.2, beta=beta.0, beta.2=beta.0.2,
		rho=rho, sig=sig, p=p, n=n, s=s, eta=eta))
}


N_exp <- 0.5e2
N.lambda <- 100
lambdas <- seq(0.0001, 0.701, length.out=N.lambda)


# Control -----------------------------------------------------------------
prediction.error.control <- array(dim=c(N_exp, length(lambdas)))
for (i in 1:N_exp) {
    d <- gen_data(n=20, p=40, s=4, sig=1, rho=0, eta=etas[eta.index], gen_X2=F)
    l <- glmnet::glmnet(d$X, d$Y, alpha=1, standardize=FALSE, 
	    family="gaussian", thresh=1e-9, maxit=1e6, lambda=lambdas,
	    intercept=FALSE)
    for (j in seq_along(lambdas)) {
	prediction.error.control[i, j] <- 
	    sum((d$X.2 %*% (l$beta[ , j] - d$beta))^2)
    }
    cat(".")
}

mean.prediction.error.control <- rev(apply(prediction.error.control, 2, mean))
plot(lambdas, mean.prediction.error.control, type="l", 
     ylim=c(0, 50), col="red")


# Plot control v other experiment -----------------------------------------
plot.control.against <- function(against) {
    plot(lambdas, mean.prediction.error.control, type="l", 
    ylim=c(0, 50), col="red")
    for (i in 1:nrow(against)) {
	lines(lambdas, rev(against[i, ]), col=i + 2)
    }

}


# Experiment with varying etas --------------------------------------------
etas <- c(0.001, 0.01, 0.1, 1)
prediction.error.etas <- array(dim=c(N_exp, length(lambdas), length(etas)))
for (eta.index in 1:length(etas)) {
    for (i in 1:N_exp) {
	d <- gen_data(n=20, p=40, s=4, sig=1, rho=0, eta=etas[eta.index])
	l <- glmnet::glmnet(d$X.2, d$Y, alpha=1, standardize=FALSE,
		family="gaussian", thresh=1e-9, maxit=1e6, lambda=lambdas,
		intercept=FALSE)
	for (j in seq_along(lambdas)) {
	    prediction.error.etas[i, j, eta.index] <- 
		sum((d$X.2 %*% (l$beta[ , j] - d$beta.2))^2)
	}
    cat(".")
    }
}

mean.prediction.error.etas <- t(sapply(seq_along(etas), function(eta) {
    apply(prediction.error.etas[ , , eta], 2, mean)
}))

plot.control.against(mean.prediction.error.etas)


# Experiment with varying rhos --------------------------------------------
rhos <- c(0.001, 0.01, 0.1, 0.2)
prediction.error.rhos <- array(dim=c(N_exp, length(lambdas), length(rhos)))
for (rho in seq_along(rhos)) {
    for (i in 1:N_exp) {
	d <- gen_data(n=20, p=40, s=4, sig=1, rho=rhos[rho], eta=0, gen_X2=F)
	l <- glmnet::glmnet(d$X.2, d$Y, alpha=1, standardize=FALSE,
		family="gaussian", thresh=1e-9, maxit=1e6, lambda=lambdas,
		intercept=FALSE)
	
	for (j in seq_along(lambdas)) {
	    prediction.error.rhos[i, j, rho] <- 
		sum((d$X.2 %*% (l$beta[ , j] - d$beta))^2)
	}
    }
    cat(".")
}
mean.prediction.error.rhos <- t(sapply(seq_along(rhos), function(rho) {
    apply(prediction.error.rhos[ , , rho], 2, mean)
}))

plot.control.against(mean.prediction.error.rhos)

