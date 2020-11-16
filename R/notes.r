# Random


p <- 1:100
n <- seq(1, 1e2)
z <- outer(n, p, function(p, n) log(p) / n)

plot(log(p) / 9 , type="l", col="red")
for (i in 10:50) 
    lines(log(p) / i, col=rgb(1, 0, 0, 1.15 - i/ 50))


# Rank and noise
p1 <- sample(1:1000)
p2 <- sample(1:1000)
P <- matrix(c(p1, p2), nrow=1000)
for (i in 1:998) {
    a <- sample(-5:5, 1)
    b <- sample(-5:5, 1)
    P <- cbind(P, a * p1 + b*p2)
}

xi <- rnorm(1000)
xi.2 <- (diag(1, 1000) - P) %*% xi


hist(xi)
hist(xi.2)

