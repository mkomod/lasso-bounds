
# Normalising of matrix
X <- mvtnorm::rmvnorm(100, rep(0, 5), diag(1, 5))
n <- apply(X, 2, function(x) norm(as.matrix(x), type="F"))
X.scl <- apply(X, 1, function(x) x / n)
X.scl <- t(X.scl)
X.scl <- X.scl * sqrt(100)
(t(X.scl) %*% X.scl) 

