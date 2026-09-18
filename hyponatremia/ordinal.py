"""Ordinal outcome tools: Brant test components and a partial proportional odds model."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize
from scipy.special import expit
from statsmodels.miscmodels.ordinal_model import OrderedModel


def brant_components(X, y, cutpoints):
    """Binary logistic fit at every cutpoint (Y >= k) with the joint sandwich covariance of Brant (1990).

    Returns the coefficient differences beta_k - beta_1 (stacked), their covariance and the
    number of parameters per binary model (intercept included).
    """
    A = sm.add_constant(X.values)
    p = A.shape[1]
    betas, ainv, scores = [], [], []
    for k in cutpoints:
        yk = (y >= k).astype(float)
        res = sm.Logit(yk, A).fit(disp=0, maxiter=500)
        pk = res.predict(A)
        betas.append(np.asarray(res.params))
        ainv.append(np.linalg.inv((A * (pk * (1 - pk))[:, None]).T @ A))
        scores.append(A * (yk - pk)[:, None])
    K = len(cutpoints)
    V = np.zeros((K * p, K * p))
    for a in range(K):
        for c in range(K):
            V[a * p:(a + 1) * p, c * p:(c + 1) * p] = ainv[a] @ (scores[a].T @ scores[c]) @ ainv[c]
    beta = np.concatenate(betas)
    C = np.zeros(((K - 1) * p, K * p))
    for k in range(1, K):
        C[(k - 1) * p:k * p, 0:p] = -np.eye(p)
        C[(k - 1) * p:k * p, k * p:(k + 1) * p] = np.eye(p)
    return C @ beta, C @ V @ C.T, p


def brant_index(names, cols, n_cut, p):
    """Positions of the difference components belonging to `cols` in the stacked Brant vector."""
    return [j * p + 1 + names.index(c) for j in range(n_cut - 1) for c in cols]


def fit_proportional_odds(y, Z):
    return OrderedModel(y, Z, distr="logit").fit(method="bfgs", maxiter=5000, disp=0, gtol=1e-6)


def fit_partial_proportional_odds(X, y, relax, constrained=False, gtol=1e-4):
    """Cumulative logit P(Y >= k | x) = expit(theta_k + x beta + x_R gamma_k), k = 1..K.

    Variables in `relax` get cutpoint-specific coefficients. In the constrained form
    gamma_k = g * s_k with s_k a linear score of the cutpoint (one extra parameter per
    relaxed variable); in the unconstrained form gamma_k is free for every cutpoint and the
    common coefficient of the relaxed variables is dropped. Maximum likelihood by L-BFGS-B
    with analytic gradient, started at the proportional odds solution, followed by Newton
    steps with a numerical Hessian. Validity requires optimizer success, max |gradient| <
    gtol on the standardized scale, a positive definite Hessian and no negative category
    probabilities.
    """
    names = list(X.columns)
    p = len(names)
    Xm = X.values
    n = len(Xm)
    K = int(y.max())
    R = X[relax].values if relax else np.zeros((n, 0))
    q = R.shape[1]
    mu, sd = Xm.mean(0), Xm.std(0)
    sd[sd == 0] = 1
    Z = (Xm - mu) / sd
    if q:
        mu_r, sd_r = R.mean(0), R.std(0)
        sd_r = np.where(sd_r == 0, 1, sd_r)
        ZR = (R - mu_r) / sd_r
    else:
        ZR = R
    po = fit_proportional_odds(y, pd.DataFrame(Z, columns=names))
    thresholds = po.model.transform_threshold_params(po.params.values[p:])[1:-1]
    theta0 = -np.asarray(thresholds)
    s_k = (np.arange(1, K + 1) - (K + 1) / 2) / K
    relaxed_idx = [names.index(c) for c in relax]
    beta_idx = [j for j in range(p) if (constrained or j not in relaxed_idx)]
    pb = len(beta_idx)
    Zb = Z[:, beta_idx]
    n_gamma = q if constrained else K * q
    x0 = np.concatenate([theta0, po.params.values[:p][beta_idx], np.zeros(n_gamma)])
    rows = np.arange(n)

    def unpack(par):
        theta, beta, g = par[:K], par[K:K + pb], par[K + pb:]
        gamma = s_k[:, None] * g[None, :] if (constrained and q) else g.reshape(K, q)
        return theta, beta, gamma

    def eta(par):
        theta, beta, gamma = unpack(par)
        E = theta[None, :] + (Zb @ beta)[:, None]
        if q:
            E = E + ZR @ gamma.T
        return E

    def probabilities(par):
        S = expit(eta(par))
        Sf = np.column_stack([np.ones(n), S, np.zeros(n)])
        return S, Sf[:, :-1] - Sf[:, 1:]

    def negll(par):
        _, P = probabilities(par)
        return -np.sum(np.log(np.maximum(P[rows, y], 1e-12)))

    def grad(par):
        S, P = probabilities(par)
        py = np.maximum(P[rows, y], 1e-12)
        dS = S * (1 - S)
        G = np.zeros((n, K))
        m1 = y >= 1
        G[rows[m1], y[m1] - 1] += dS[rows[m1], y[m1] - 1] / py[m1]
        m2 = y + 1 <= K
        G[rows[m2], y[m2]] -= dS[rows[m2], y[m2]] / py[m2]
        g_theta = G.sum(0)
        g_beta = Zb.T @ G.sum(1)
        if not q:
            g_gamma = np.zeros(0)
        elif constrained:
            g_gamma = ((G * s_k[None, :]).sum(1)[:, None] * ZR).sum(0)
        else:
            g_gamma = (G.T @ ZR).ravel()
        return -np.concatenate([g_theta, g_beta, g_gamma])

    def hessian(x):
        H = np.zeros((len(x), len(x)))
        h = 1e-5
        for j in range(len(x)):
            e = np.zeros(len(x))
            e[j] = h
            H[:, j] = (grad(x + e) - grad(x - e)) / (2 * h)
        return (H + H.T) / 2

    res = optimize.minimize(negll, x0, jac=grad, method="L-BFGS-B",
                            options={"maxiter": 20000, "maxfun": 200000, "gtol": 1e-7, "ftol": 1e-12})
    x = res.x.copy()
    H = hessian(x)
    newton_steps = 0
    for _ in range(8):
        g = grad(x)
        if np.max(np.abs(g)) < gtol:
            break
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        x_new = x - step
        if negll(x_new) <= negll(x) + 1e-9:
            x = x_new
            newton_steps += 1
            H = hessian(x)
        else:
            break
    gnorm = float(np.max(np.abs(grad(x))))
    eig = np.linalg.eigvalsh(H)
    pd_ok = bool(eig.min() > 0)
    V = np.linalg.inv(H) if pd_ok else np.linalg.pinv(H)
    sd_b = sd[beta_idx]
    _, beta_z, _ = unpack(x)
    beta = np.full(p, np.nan)
    beta[beta_idx] = beta_z / sd_b
    Vbb = V[K:K + pb, K:K + pb] / np.outer(sd_b, sd_b)
    Vb = np.full((p, p), np.nan)
    Vb[np.ix_(beta_idx, beta_idx)] = Vbb
    _, P = probabilities(x)
    n_negative = int((P < -1e-12).sum())
    valid = bool(res.success) and gnorm < gtol and pd_ok and np.isfinite(Vbb).all() and n_negative == 0
    return dict(beta=beta, cov=Vb, beta_idx=beta_idx, valid=valid, optimizer_success=bool(res.success), gnorm=gnorm, gtol=gtol,
                newton_steps=newton_steps, min_eigenvalue=float(eig.min()), hessian_pd=pd_ok, n_negative=n_negative, n_cells=int(P.size),
                loglik=-negll(x), loglik_po=float(po.llf), n_params=len(x), message=str(res.message)[:60])
