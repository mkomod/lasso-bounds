library(mvtnorm)
library(glmnet)


#' Generates data according to algorithm 1 from [HL12]
gen_data <- function(n, p, s, sig, rho) {
    if (s > p) stop("s > p")
    if (rho < 0 | rho >= 1) stop("rho must be in [0, 1)")

    S <- diag(1 - rho, p) + rho
    X <- mvtnorm::rmvnorm(n, mean=rep(0, p), S)
    X <- scale(X, center=FALSE)      # TODO: fix X'X_jj != n?

    e <- rnorm(n)
    beta.0 <- c(rep(1, s), rep(0, p-s))
    Y <- X %*% beta.0 + sig * e

    return(list(Y=Y, X=X, beta=beta.0, rho=rho, sig=sig, p=p, n=n, s=s))
}


N.lambda <- 100
lambdas <- seq(0.0001, 10.0001, length.out=N.lambda)
PE <- matrix(nrow=100, ncol=length(lambdas))

for (j in 1:N.lambda) {
    d <- gen_data(20, 40, 4, 1, 0)
    l <- glmnet::glmnet(d$X, d$Y, 
	    alpha=1,                          # for LASSO
	    # nlambda=100,
	    standardize=F,
	    thresh=1e-9,
	    maxit=1e6,
	    lambda=lambdas,
	    intercept=FALSE)
    
    for (i in seq_along(lambdas)) {
	PE[j, i] <- sum((d$X %*% (l$beta[ , i] - d$beta))^2)
    }
}


plot(rev(apply(PE, 2, mean)))

