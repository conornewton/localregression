#' @export
mse <- function(ys1, ys2) {
    mean((ys1 - ys2)^2)
}

#' Cross validation for the local regression
#' @param y is the true outputs
#' @param x is the inputs
#' @param X is the inputs with a feature transform
#' @return the cross-validation error
#' @export
k_fold_cross_validation_lr <- function(y, x, X, L, k) {
    n <- length(y)
    ss <- floor(n / k) # split size
    ids <- sample(1:n, n) # Id's to sample the data randomly

    err <- 0
    for (i in 1:k) {
        test_ids <- ids[1:ss + (i - 1) * ss]

        y_test <- y[test_ids]
        y_train <- y[-test_ids]

        x_test <- x[test_ids, ]
        x_train <- x[-test_ids, ]

        X_test <- X[test_ids, ]
        X_train <- X[-test_ids, ]

        predLocalRcpp_iter <- lm_local_reg_iter(y_train, x_test, X_test, x_train, X_train, L)

        err <- err + mse(y_test, predLocalRcpp_iter)
    }
    return(err / (k + 1))
}