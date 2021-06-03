// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

//' Evaluates the mutltivariate normal density function
//' @param X,
//' @param mu, the mean of the distribution
//' @param L, the lower cholesky matrix of the bandwidth matrix H
// [[Rcpp::export(name = "dmvnInt")]]
arma::vec dmvnInt(const arma::mat &X, const arma::rowvec &mu, const arma::mat &L)
{
    unsigned int d = X.n_cols;
    unsigned int m = X.n_rows;

    arma::vec D = L.diag();
    arma::vec out(m);
    arma::vec z(d);

    double acc;
    unsigned int icol, irow, ii;
    for (icol = 0; icol < m; icol++)
    {
        for (irow = 0; irow < d; irow++)
        {
            acc = 0.0;
            for (ii = 0; ii < irow; ii++)
                acc += z.at(ii) * L.at(irow, ii);
            z.at(irow) = (X.at(icol, irow) - mu.at(irow) - acc) / D.at(irow);
        }
        out.at(icol) = sum(square(z));
    }

    out = exp(-0.5 * out - ((d / 2.0) * log(2.0 * M_PI) + sum(log(D))));

    return out;
}

// [[Rcpp::export(name = "lm_qr_rcpp")]]
arma::vec lm_qr_rcpp(const arma::mat &X, const arma::vec &y)
{
    arma::mat Q;
    arma::mat R;
    //faster than arma::qr
    arma::qr_econ(Q, R, X); //Compute the QR-decomposition

    arma::vec betas = arma::solve(arma::trimatu(R), Q.t() * y);

    return betas;
}

// [[Rcpp::export(name = "lm_local_reg_pred")]]
double lm_local_reg_pred(const arma::vec &y, const arma::rowvec &x0, const arma::rowvec &X0, const arma::mat &x, const arma::mat &X, const arma::mat &L)
{

    arma::vec weights = arma::sqrt(dmvnInt(x, x0, L));

    arma::vec betas = lm_qr_rcpp(X.each_col() % weights, y % weights);

    return arma::as_scalar(X0 * betas); //element-wise multiplication
}

// [[Rcpp::export(name = "lm_local_reg_pred_multiple")]]
arma::vec lm_local_reg_pred_multiple(arma::vec &y, const arma::mat &x0, const arma::mat &X0, arma::mat &x, arma::mat &X, arma::mat &L)
{
    unsigned int n = X0.n_rows;
    arma::vec out(n);

    for (unsigned int i = 0; i < n; i++)
    {
        out[i] = lm_local_reg_pred(y, x0.row(i), X0.row(i), x, X, L);
    }
    return out;
}

//' Get the lower cholesky decomposition of  a matrix
//' @param H arma::mat, the matrix to perform cholesky decomposition on
//' @return Lower cholesky matrix of H
// [[Rcpp::export(name = "get_chol")]]
arma::mat get_chol(arma::mat &H)
{
    return arma::chol(H, "lower");
}