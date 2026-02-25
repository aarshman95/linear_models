import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy as sp
import statsmodels.api as sm
from typing import List, Mapping, Union


class OLS:
    def __init__(self, add_intercept: bool = False):
        self.add_intercept = add_intercept

    def _get_resids(self, y: np.array, y_hat: np.array = None, X: np.array = None):
        if y_hat is None:
            y_hat = self.predict(X)
        resids = y - y_hat

        return resids

    def fit(
        self,
        X: Union[np.array, pd.DataFrame, pd.Series],
        y: Union[np.array, pd.DataFrame, pd.Series],
        y_mean: float = None,
        intercept_included: bool = False,
    ):
        self.obs_names_fitted, self.X_names, self.y_name = get_names(X=X, y=y)
        X, y, _ = get_arrays(X=X, y=y)

        self.nobs_fitted = X.shape[0]
        if self.add_intercept and not intercept_included:
            X = np.concatenate([np.ones(shape=(self.nobs_fitted, 1)), X], axis=1)
            intercept_included = True
        self.dof_model = X.shape[1] - intercept_included
        self.dof_resid = self.nobs_fitted - self.dof_model - intercept_included

        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        self.XtX_inv_fitted = XtX_inv

        self.beta_hat = XtX_inv @ X.T @ y
        # Projection Matrix
        self.H = X @ XtX_inv @ X.T
        # Leverage values are equivalent to dy_hat/dy. Measures how much influence an observation has on its fitted value.
        self.leverage_values = np.diag(self.H)
        # Residual Maker
        self.M = np.eye(self.nobs_fitted) - self.H
        # Equivalent to self.H @ y
        self.y_hat_fitted = X @ self.beta_hat
        # Equivalent to self.M @ y
        self.resids_hat_fitted = y - self.y_hat_fitted
        # This is the unbiased estimate of residual variance
        self.error_var_fitted = (
            self.resids_hat_fitted.T @ self.resids_hat_fitted
        ).item() / self.dof_resid
        # The MLE estimate of residual variance is biased
        self.error_var_fitted_mle = (
            self.error_var_fitted * self.dof_resid / self.nobs_fitted
        )

        # Hypothesis test metrics for coefficients
        self.beta_cov = self.error_var_fitted * XtX_inv
        self.beta_var = np.diag(self.beta_cov)
        self.beta_hat_tstat = self.beta_hat.ravel() / np.power(self.beta_var, 0.5)
        self.beta_hat_pvals = 2 * (
            1 - sp.stats.t.cdf(np.abs(self.beta_hat_tstat), self.dof_resid)
        )

        # Studentized Residuals
        self.resids_cov = self.error_var_fitted * self.M
        resids_se = np.diag(self.resids_cov) ** (0.5)
        self.resids_hat_fitted_studentized = self.resids_hat_fitted / resids_se

        # Cooks Distance (TO-DO: Derive formula yourself)
        # Defined as sum of all changes in predictions when removing observation i
        term1 = self.resids_hat_fitted**2 / (
            (self.dof_model + intercept_included) * self.error_var_fitted
        )
        term2 = self.leverage_values / ((1 - self.leverage_values) ** 2)
        self.cooks_distance = term1 * term2

        y_mean_flag = 0
        if y_mean is None:
            y_mean = np.mean(y)
            y_mean_flag = 1
        self.y_mean = y_mean

        sse = np.sum(self.resids_hat_fitted**2).item()
        sst = np.sum((y - self.y_mean) ** 2).item()
        ssr = np.sum((self.y_hat_fitted - self.y_mean) ** 2).item()
        msr = ssr / self.dof_model
        mse = sse / self.dof_resid
        mst = sst / (self.nobs_fitted - y_mean_flag)
        self.r2 = ssr / sst
        self.r2adj = 1 - mse / mst
        # Can prove that this ratio follows an F-Distribution
        self.f_stat = msr / mse
        self.f_stat_pval = 1 - sp.stats.f.cdf(
            self.f_stat, self.dof_model, self.dof_resid
        )

        # For log-likelihood, we want to use the MLE variance
        self.log_likelihood = (-self.nobs_fitted / 2) * np.log(
            2 * np.pi * self.error_var_fitted_mle
        ) - np.sum(self.resids_hat_fitted**2) / (2 * self.error_var_fitted_mle)
        self.aic = 2 * (self.dof_model + intercept_included - self.log_likelihood)
        self.bic = -2 * self.log_likelihood + (
            self.dof_model + intercept_included
        ) * np.log(self.nobs_fitted)

        return

    def predict(
        self,
        X: Union[np.array, pd.DataFrame, pd.Series],
        intercept_included: bool = False,
    ):
        if self.add_intercept and not intercept_included:
            X = np.concatenate([np.ones(shape=(X.shape[0], 1)), X], axis=1)
        return X @ self.beta_hat

    def check_linearity(
        self,
        y: Union[np.array, pd.DataFrame, pd.Series],
        y_hat: Union[np.array, pd.DataFrame, pd.Series] = None,
        X: Union[np.array, pd.DataFrame, pd.Series] = None,
        show_fig: bool = True,
    ):
        obs_names, _, _ = get_names(X=X, y=y)
        X, y, y_hat = get_arrays(X=X, y=y, y_hat=y_hat)
        if y_hat is None:
            y_hat = self.predict(X=X)
        resids = self._get_resids(y=y, y_hat=y_hat, X=X)

        resids_yhat_summ = univariate_regression(X=y_hat, y=resids, add_intercept=False)
        y_yhat_summ = univariate_regression(X=y_hat, y=y, add_intercept=False)

        if show_fig:
            fig = plot_linearity_check(
                X=y_hat,
                y=resids,
                reg_summ=resids_yhat_summ,
                X_label=f"{self.y_name}_hat",
                y_label="Residuals",
                obs_labels=obs_names,
            )
            fig.show()

            fig = plot_linearity_check(
                X=y_hat,
                y=y,
                reg_summ=y_yhat_summ,
                X_label=f"{self.y_name}_hat",
                y_label=self.y_name,
                obs_labels=obs_names,
            )
            fig.show()

        resids_yhat_summ = pd.DataFrame(
            list(resids_yhat_summ.items()), columns=["stat", "Residuals"]
        ).set_index("stat")

        y_yhat_summ = pd.DataFrame(
            list(y_yhat_summ.items()), columns=["stat", self.y_name]
        ).set_index("stat")

        result = pd.concat([resids_yhat_summ, y_yhat_summ], axis=1)

        return result

    def check_normality(
        self,
        y: Union[np.array, pd.DataFrame, pd.Series],
        y_hat: Union[np.array, pd.DataFrame, pd.Series] = None,
        X: Union[np.array, pd.DataFrame, pd.Series] = None,
        nbins_hist: int = 50,
        show_fig: bool = True,
    ):
        obs_names, _, _ = get_names(X=X, y=y)
        X, y, y_hat = get_arrays(X=X, y=y, y_hat=y_hat)

        if y_hat is None:
            y_hat = self.predict(X)

        resids = y - y_hat
        resids = np.asarray(resids).ravel()

        jb_test_result = jb_test(resids)
        if show_fig:
            resids_std = resids / (self.error_var_fitted ** (0.5))
            qq_fig = plot_qq_normal(resids_std, obs_labels=obs_names)
            hist_fig = plot_hist(resids_std, nbins=nbins_hist)

            jb_stat, jb_pvalue, skew, kurt = (
                jb_test_result["jb_stat"],
                jb_test_result["jb_pvalue"],
                jb_test_result["skew"],
                jb_test_result["kurtosis"],
            )
            print("Jarque–Bera Test")
            print(f"JB statistic : {jb_stat:.4f}")
            print(f"p-value      : {jb_pvalue:.6f}")
            print(f"Skewness     : {skew:.4f}")
            print(f"Kurtosis     : {kurt:.4f}")

            qq_fig.show()
            hist_fig.show()

        return pd.Series(jb_test_result)

    def check_outliers(
        self,
        show_fig: bool = True,
        size_scale: float = 40,
    ):
        result = pd.DataFrame(
            [
                self.leverage_values,
                self.resids_hat_fitted_studentized,
                self.cooks_distance,
            ],
            index=["leverage", "resid_studentized", "cooks_distance"],
            columns=self.obs_names_fitted,
        ).T

        if show_fig:
            fig = plot_influence(
                resids_data=result,
                obs_labels=self.obs_names_fitted,
                size_scale=size_scale,
            )
            fig.show()

        return result

    def check_multicollinearity(
        self, X: Union[np.array, pd.DataFrame, pd.Series], show_fig: bool = True
    ):
        if type(X) == pd.DataFrame:
            X = X[self.X_names]

        obs_labels, _, _ = get_names(X=X)
        X, _, _ = get_arrays(X=X)
        corr_mat = np.corrcoef(X.T)
        corr_mat = pd.DataFrame(corr_mat, index=self.X_names, columns=self.X_names)

        # Condition Number is the max ratio of the relative error in the solution to the relative error of the input.
        # Consider the linear equation Ax=b for invertible A w/ solution x=A-1b. The error in x, dx, given an error
        #   in b, db, is then dx = A-1(db). We want max ( ||dx|| / ||x|| ) / ( ||db|| / ||b|| ). This is equivalent to
        #   the ratio of the max abs eigenvalue to the min abs eigenvalue of A.
        # For OLS, we let A be the covariance matrix of the features.
        eigvals = np.linalg.eigvals(X.T @ X) ** (0.5)
        condition_number = (max(abs(eigvals)) / min(abs(eigvals))).item()

        # Variance Inflation Factor for a predictor X_j tells us how much the variance of Beta_j is impacted by X_j's
        #   correlation to other features.
        # We have var(Beta_j) = sigma2 * (XtX)-1 = ( sigma2 / ( (n-1) * var(X_j) ) ) * VIF_j
        #   where VIF_j = 1 / (1 - r2), where r2 is calculated upon regressing X_j on the remaining features.
        X = pd.DataFrame(X, index=obs_labels, columns=self.X_names)
        vifs = {}
        for X_col in self.X_names:
            ols = OLS(add_intercept=True)
            ols.fit(X=X.drop(X_col, axis=1), y=X[X_col])
            vifs[X_col] = 1.0 / (1.0 - ols.r2)
        vifs = pd.Series(vifs, name="vif")

        if show_fig:
            fig = plot_corr_heatmap(corr_mat=corr_mat)
            fig.update_layout(
                title=f"Correlation Matrix, Condition Number = {condition_number:.2f}",
            )
            fig.show()

            print("Variance Inflation Factors:")
            print(vifs.sort_values(ascending=False))

        return corr_mat, condition_number, vifs

    def check_homoscedasticity(
        self,
        X: Union[np.array, pd.DataFrame, pd.Series],
        y: Union[np.array, pd.DataFrame, pd.Series] = None,
        y_hat: Union[np.array, pd.DataFrame, pd.Series] = None,
        intercept_included: bool = False,
        oos_data: bool = False,
        show_fig: bool = True,
    ):
        obs_names, _, _ = get_names(X=X, y=y)
        X, _, _ = get_arrays(X=X)
        if self.add_intercept and not intercept_included:
            X = np.concatenate([np.ones(shape=(self.nobs_fitted, 1)), X], axis=1)

        if oos_data:
            if y_hat is None:
                y_hat = self.predict(X, intercept_included=True)
            resids = y - y_hat
            # This is the out of sample prediction standard error. Slightly different than the in sample version.
            resids_se = self.error_var_fitted * (
                np.eye(X.shape[0]) + X @ self.XtX_inv_fitted @ X.T
            )
            resids_se = np.diag(resids_se) ** (0.5)
            resids_studentized = resids / resids_se
        else:
            obs_names = self.obs_names_fitted
            y_hat = self.y_hat_fitted
            resids = self.resids_hat_fitted
            resids_studentized = self.resids_hat_fitted_studentized

        resids_studentized_yhat_summ = univariate_regression(
            X=y_hat, y=resids_studentized, add_intercept=False
        )
        bp_test = self._bp_test(resids=resids, X=X, intercept_included=True)

        if show_fig:
            fig = plot_linearity_check(
                X=y_hat,
                y=resids,
                reg_summ=resids_studentized_yhat_summ,
                X_label=f"{self.y_name}_hat",
                y_label="Studentized Residual",
                obs_labels=obs_names,
            )
            fig.show()

            lm_stat, lm_pvalue, f_stat, f_pvalue = (
                bp_test["lm_stat"],
                bp_test["lm_pvalue"],
                bp_test["f_stat"],
                bp_test["f_pvalue"],
            )
            print("Breusch-Pagan Test")
            print(f"LM statistic : {lm_stat:.4f}")
            print(f"LM p-value      : {lm_pvalue:.6f}")
            print(f"F statistic     : {f_stat:.4f}")
            print(f"F p-value     : {f_pvalue:.4f}")

        resids_studentized_yhat_summ = pd.DataFrame(
            list(resids_studentized_yhat_summ.items()),
            columns=["stat", "Studentized Residual"],
        ).set_index("stat")

        return pd.Series(bp_test), resids_studentized_yhat_summ

    def _bp_test(self, resids, X, intercept_included: bool = False):
        X, _, _ = get_arrays(X=X)
        if self.add_intercept and not intercept_included:
            X = np.concatenate([np.ones(shape=(self.nobs_fitted, 1)), X], axis=1)

        # TO-DO: Derive and calculate test from scratch
        lm, lm_pval, fval, fpval = sm.stats.diagnostic.het_breuschpagan(
            resid=resids, exog_het=X
        )

        return {"lm_stat": lm, "lm_pvalue": lm_pval, "f_stat": fval, "f_pvalue": fpval}

    def check_autocorrelation(
        self,
        y: Union[np.array, pd.DataFrame, pd.Series],
        y_hat: Union[np.array, pd.DataFrame, pd.Series] = None,
        X: Union[np.array, pd.DataFrame, pd.Series] = None,
        lags: int = 20,
        significance_level: float = 0.05,
        show_fig: bool = True,
    ):
        obs_names, _, _ = get_names(X=X, y=y)
        X, y, y_hat = get_arrays(X=X, y=y, y_hat=y_hat)
        resids = self._get_resids(y=y, y_hat=y_hat, X=X)
        resids = pd.Series(resids, name="resid", index=obs_names)

        resids_lagged = []
        for i in range(1, lags + 1):
            resids_lag_i = resids.shift(i)
            resids_lag_i.name = f"resid_l{i}"
            resids_lagged.append(resids_lag_i)
        resids_lagged = pd.concat(resids_lagged, axis=1)

        resids_arr, resids_lagged_arr = resids.values, resids_lagged.values
        n_resids = len(resids_arr)
        acf_stats = {}
        for lag in range(1, lags + 1):
            ols = OLS(add_intercept=False)
            ols.fit(X=resids_lagged_arr[lag:, lag - 1], y=resids[lag:], y_mean=0.0)
            coef_se = ols.beta_var.item() ** (0.5)
            t_stat_lb, t_stat_ub = sp.stats.t.ppf(
                q=significance_level, df=n_resids - lag
            ), sp.stats.t.ppf(q=1.0 - significance_level, df=n_resids - lag)
            coef_lb, coef_ub = t_stat_lb * coef_se, t_stat_ub * coef_se
            acf_lag = {
                "coef": ols.beta_hat.item(),
                "tstat": ols.beta_hat_tstat.item(),
                "pvalue": ols.beta_hat_pvals.item(),
                "coef_lb": coef_lb,
                "coef_ub": coef_ub,
            }
            acf_stats[lag] = acf_lag
        acf_stats = pd.DataFrame(acf_stats).T
        acf_stats.index.name = "lag"

        pacf_stats = {}
        for lag in range(1, lags + 1):
            ols = OLS(add_intercept=False)
            ols.fit(X=resids_lagged_arr[lag:, :lag], y=resids[lag:], y_mean=0.0)
            coef_se = ols.beta_var[lag - 1].item() ** (0.5)
            t_stat_lb, t_stat_ub = sp.stats.t.ppf(
                q=significance_level, df=n_resids - lag
            ), sp.stats.t.ppf(q=1.0 - significance_level, df=n_resids - lag)
            coef_lb, coef_ub = t_stat_lb * coef_se, t_stat_ub * coef_se
            pacf_lag = {
                "coef": ols.beta_hat[lag - 1].item(),
                "tstat": ols.beta_hat_tstat[lag - 1].item(),
                "pvalue": ols.beta_hat_pvals[lag - 1].item(),
                "coef_lb": coef_lb,
                "coef_ub": coef_ub,
            }
            pacf_stats[lag] = pacf_lag
        pacf_stats = pd.DataFrame(pacf_stats).T
        pacf_stats.index.name = "lag"

        if show_fig:
            fig = plot_acf_with_bounds(acf_stats)
            fig.update_layout(
                title=f"Autocorrelation Function (ACF), Significance Level = {100*significance_level:.1f}%",
            )
            fig.show()

            fig = plot_acf_with_bounds(pacf_stats, name="PACF")
            fig.update_layout(
                title=f"Partial Autocorrelation Function (PACF), Significance Level = {100*significance_level:.1f}%"
            )
            fig.show()

        return acf_stats, pacf_stats


def get_names(
    X: Union[np.array, pd.DataFrame, pd.Series] = None,
    y: Union[np.array, pd.DataFrame, pd.Series] = None,
):
    obs_names, X_names, y_name = None, None, None
    if type(X) == pd.DataFrame:
        obs_names = [str(i) for i in X.index.values]
        X_names = X.columns
    elif type(X) == pd.Series:
        obs_names = [str(i) for i in X.index.values]
        X_names = [X.name if X.name is not None else "X"]
    elif X is not None:
        obs_names = [str(i) for i in np.arange(1, len(X) + 1)]
        X_names = ["X"]

    if type(y) == pd.DataFrame:
        y_name = y.columns[0]
        if obs_names is None:
            obs_names = [str(i) for i in y.index.values]
    elif type(y) == pd.Series:
        y_name = y.name if y.name is not None else "y"
        if obs_names is None:
            obs_names = [str(i) for i in y.index.values]
    elif y is not None:
        y_name = "y"
        if obs_names is None:
            obs_names = [str(i) for i in np.arange(1, len(y) + 1)]

    return obs_names, X_names, y_name


def get_arrays(
    X: Union[np.array, pd.DataFrame, pd.Series] = None,
    y: Union[np.array, pd.DataFrame, pd.Series] = None,
    y_hat: Union[np.array, pd.DataFrame, pd.Series] = None,
):
    if type(X) == pd.DataFrame or type(X) == pd.Series:
        X = X.values
    if type(y) == pd.DataFrame or type(y) == pd.Series:
        y = y.values
    if type(y_hat) == pd.DataFrame or type(y_hat) == pd.Series:
        y_hat = y_hat.values

    if X is not None:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
    if y is not None:
        y = y.ravel()
    if y_hat is not None:
        y_hat = y_hat.ravel()

    return X, y, y_hat


def univariate_regression(
    X: Union[np.array, pd.DataFrame, pd.Series],
    y: Union[np.array, pd.DataFrame, pd.Series],
    add_intercept: bool = False,
):
    ols = OLS(add_intercept=add_intercept)
    ols.fit(X=X, y=y)
    coef = ols.beta_hat[-1].item()
    tstat = ols.beta_hat_tstat[-1].item()
    pval = ols.beta_hat_pvals[-1].item()
    r2 = ols.r2

    reg_summ = {"coef": coef, "tstat": tstat, "pvalue": pval, "r2": r2}
    return reg_summ


def jb_test(resids):
    # ============================================================
    # 3️⃣ Jarque–Bera Test
    # TO-DO: Derive formula and distribution yourself
    # ============================================================

    n = len(resids)

    m2 = np.mean(resids**2)
    m3 = np.mean(resids**3)
    m4 = np.mean(resids**4)

    skew = m3 / (m2**1.5)
    kurt = m4 / (m2**2)

    jb_stat = n / 6 * (skew**2 + (kurt - 3) ** 2 / 4)
    jb_pval = 1 - sp.stats.chi2.cdf(jb_stat, df=2)

    return {
        "jb_stat": jb_stat,
        "jb_pvalue": jb_pval,
        "skew": skew,
        "kurtosis": kurt,
    }


def plot_linearity_check(
    X: Union[np.array, pd.DataFrame, pd.Series],
    y: Union[np.array, pd.DataFrame, pd.Series],
    reg_summ: Mapping[str, float],
    X_label: str = None,
    y_label: str = None,
    obs_labels: List[str] = None,
):
    coef, tstat, pval, r2 = (
        reg_summ["coef"],
        reg_summ["tstat"],
        reg_summ["pvalue"],
        reg_summ["r2"],
    )
    X_sorted = np.sort(X.ravel()).reshape(X.shape)
    fitted_line = coef * X_sorted

    title_text = (
        f"{y_label} vs {X_label} <br>"
        f"Coef = {coef:.4f} | "
        f"t-stat = {tstat:.2f} | "
        f"p-value = {pval:.3e} | "
        f"R² = {r2:.4f}"
    )

    # --- Create scatter plot ---
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=X.ravel(),
            y=y.ravel(),
            mode="markers",
            marker=dict(size=6, opacity=0.6),
            text=obs_labels,  # labels for each point
            hovertemplate="Label: %{text}<br>X: %{x}<br>Y: %{y}<extra></extra>",
            name="Residuals",
        )
    )

    # # # Add horizontal zero line (important diagnostic)
    # fig.add_hline(y=0, line_dash="dash", line_width=1)

    # Regression-implied line
    fig.add_trace(
        go.Scatter(
            x=X_sorted.ravel(),
            y=fitted_line.ravel(),
            mode="lines",
            line=dict(width=2),
            name=f"Fitted Line",
        )
    )

    fig.update_layout(
        title=title_text,
        xaxis_title=X_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=600,
        width=800,
    )

    return fig


def plot_qq_normal(resids, obs_labels: List[str] = None):
    # ============================================================
    # 1️⃣ Q-Q Plot against N(0, 1)
    # ============================================================

    # Sample quantiles
    resids_labels = [(resid, label) for resid, label in zip(resids, obs_labels)]
    resids_labels = sorted(resids_labels, key=lambda x: x[0])
    resids, obs_labels = np.array([i[0] for i in resids_labels]), [
        i[1] for i in resids_labels
    ]
    del resids_labels
    n = len(resids)

    # Theoretical quantiles from N(0, 1)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo_quantiles = sp.stats.norm.ppf(probs, loc=0, scale=1)

    qq_fig = go.Figure()

    qq_fig.add_trace(
        go.Scatter(
            x=theo_quantiles,
            y=resids,
            mode="markers",
            text=obs_labels,  # labels for each point
            hovertemplate="Label: %{text}<br>X: %{x}<br>Y: %{y}<extra></extra>",
            name="Residuals",
        )
    )

    # 45-degree reference line
    line_min = min(theo_quantiles.min(), resids.min())
    line_max = max(theo_quantiles.max(), resids.max())

    qq_fig.add_trace(
        go.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            name="45° Line",
        )
    )

    qq_fig.update_layout(
        title="Q–Q Plot vs Normal(0, 1)",
        xaxis_title="Theoretical Normal Quantiles",
        yaxis_title="Residuals Quantiles",
        template="plotly_white",
    )

    return qq_fig


def plot_hist(resids, nbins=50):
    # ============================================================
    # 2️⃣ Histogram + Theoretical Normal Density
    # ============================================================

    hist_fig = go.Figure()

    # Histogram (density-scaled)
    hist_fig.add_trace(
        go.Histogram(
            x=resids,
            histnorm="probability density",
            nbinsx=nbins,
            name="Residuals",
            opacity=0.6,
        )
    )

    # Overlay theoretical normal PDF
    x_grid = np.linspace(resids.min(), resids.max(), 400)
    pdf = sp.stats.norm.pdf(x_grid, loc=0, scale=1)

    hist_fig.add_trace(go.Scatter(x=x_grid, y=pdf, mode="lines", name="Normal PDF"))

    hist_fig.update_layout(
        title="Histogram with Normal(0, 1) Overlay",
        xaxis_title="Residual",
        yaxis_title="Density",
        template="plotly_white",
        barmode="overlay",
    )

    return hist_fig


def plot_influence(
    resids_data: pd.DataFrame, obs_labels: List[str], size_scale: float = 40.0
):
    leverage = resids_data["leverage"].values
    resid = resids_data["resid_studentized"].values
    cooks = resids_data["cooks_distance"].values

    # --- Scale Cook's distance for marker size ---
    # sqrt scaling is standard for bubble plots
    sizes = size_scale * np.sqrt(cooks / cooks.max())

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=leverage,
            y=resid,
            mode="markers",
            marker=dict(size=sizes, sizemode="diameter", opacity=0.6),
            text=obs_labels,
            hovertemplate=(
                "Label: %{text}<br>"
                "Leverage: %{x:.4f}<br>"
                "Studentized Residual: %{y:.4f}<br>"
                "Cook's Distance: %{customdata:.4f}<extra></extra>"
            ),
            customdata=cooks,
            name="Observations",
        )
    )

    # Reference line
    fig.add_hline(y=0, line_dash="dash")

    fig.update_layout(
        title="Influence Plot",
        xaxis_title="Leverage",
        yaxis_title="Studentized Residual",
        template="plotly_white",
        height=600,
        width=800,
    )

    return fig


def plot_corr_heatmap(corr_mat: pd.DataFrame):
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_mat.abs().values,  # color by magnitude
            x=corr_mat.columns,
            y=corr_mat.columns,
            colorscale=[[0, "green"], [1, "red"]],
            zmin=0,
            zmax=1,
            text=corr_mat.round(2).values,  # show real correlations
            texttemplate="%{text}",  # render text in cells
            textfont={"color": "black"},
            colorbar=dict(title="|Correlation|"),
        )
    )
    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        title=f"Correlation Matrix",
        width=700,
        height=700,
    )

    return fig


def plot_acf_with_bounds(acf_stats: pd.DataFrame, name: str = "ACF"):
    """
    Create an ACF plot with confidence bounds using plotly graph_objects.

    Parameters
    ----------
    acf_stats : pandas.DataFrame
        Must contain columns: ['coef', 'coef_lb', 'coef_ub']
        Index should represent lags.
    """

    lags = acf_stats.index.to_numpy()
    coef = acf_stats["coef"].to_numpy()
    lb = acf_stats["coef_lb"].to_numpy()
    ub = acf_stats["coef_ub"].to_numpy()

    fig = go.Figure()

    # Vertical stem lines
    for lag, value in zip(lags, coef):
        fig.add_trace(
            go.Scatter(
                x=[lag, lag],
                y=[0, value],
                mode="lines",
                line=dict(color="blue"),
                showlegend=False,
            )
        )

    # Markers
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=coef,
            mode="markers",
            marker=dict(color="blue", size=8),
            name=name,
        )
    )

    # Upper bound (dashed)
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=ub,
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Upper Bound",
        )
    )

    # Lower bound (dashed)
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=lb,
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Lower Bound",
        )
    )

    # Zero line
    fig.add_shape(
        type="line",
        x0=lags.min(),
        x1=lags.max(),
        y0=0,
        y1=0,
        line=dict(color="black", width=1),
        xref="x",
        yref="y",
    )

    fig.update_layout(
        title="Autocorrelation Function (ACF)",
        xaxis_title="Lag",
        yaxis_title=name,
        template="simple_white",
        width=800,
        height=500,
    )

    return fig
