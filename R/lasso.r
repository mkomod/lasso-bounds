library(mvtnorm)
library(glmnet)


#' Generates data according to algorithm 1 from [HL12]
gen_data <- function(n, p, s, sig, rho) {
    if (s > p) stop("s > p")
    if (rho < 0 | rho >= 1) stop("rho must be in [0, 1)")

    S <- diag(1 - rho, p) + rho
    X <- mvtnorm::rmvnorm(n, mean=rep(0, p), S)
    norms <- apply(X, 2, function(x) norm(as.matrix(x), type="F"))
    X <- t(apply(X, 1, function(x) x / norms)) * sqrt(n)

    e <- rnorm(n, 0, sd=sig)
    beta.0 <- c(rep(1, s), rep(0, p-s))
    Y <- X %*% beta.0 + e

    return(list(Y=Y, X=X, beta=beta.0, rho=rho, sig=sig, p=p, n=n, s=s))
}


N_exp <- 0.5e3
N.lambda <- 100
lambdas <- seq(0.0001, 10.001, length.out=N.lambda)
prediction.error <- matrix(nrow=N_exp, ncol=length(lambdas))

for (i in 1:N_exp) {
    d <- gen_data(n=20, p=40, s=4, sig=1, rho=0)
    l <- glmnet::glmnet(d$X, d$Y, 
	    alpha=1,                          # for LASSO
	    # nlambda=100,
	    standardize=FALSE,
	    family="gaussian",
	    thresh=1e-9,
	    maxit=1e6,
	    lambda=lambdas,
	    intercept=FALSE)
    
    for (j in seq_along(lambdas)) {
	prediction.error[i, j] <- sum((d$X %*% (l$beta[ , j] - d$beta))^2)
    }
}

plot(lambdas, rev(apply(prediction.error, 2, mean)), type="l", ylim=c(0, 100))
lines(lambdas, rev(apply(prediction.error, 2, function(p) quantile(p, 0.025))))
lines(lambdas, rev(apply(prediction.error, 2, function(p) quantile(p, 0.975))))

