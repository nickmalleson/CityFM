import numpy as np

ALPHAS = np.logspace(-2, 5, 15)


def fold_ids(groups, n_folds, seed):
    # deal whole groups (neighbourhoods) into folds at random
    names = np.unique(groups)
    fold_of = dict(zip(names, np.random.default_rng(seed).permutation(len(names)) % n_folds))
    return np.array([fold_of[g] for g in groups])


def ridge_path(X_tr, y_tr, X_te, alphas):
    # ridge predictions on X_te for every alpha from one SVD; features standardised on the training rows
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1
    a, b = (X_tr - mu) / sd, (X_te - mu) / sd
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    uty = u.T @ (y_tr - y_tr.mean())
    bv = b @ vt.T
    return np.stack([y_tr.mean() + bv @ (s / (s ** 2 + alpha) * uty) for alpha in alphas])


def spatial_cv_predict(X, y, groups, n_folds=5, seed=0, alphas=ALPHAS):
    """Predict every row from a ridge regression fitted on the other folds of whole groups.

    The ridge strength is chosen by the same grouped cross-validation inside the training folds.
    (Leave-one-out, as RidgeCV does, leaves each row's neighbours in training, so it picks too little
    smoothing and the model then extrapolates badly to a held-out neighbourhood.)
    """
    X = np.asarray(X, dtype=np.float64)
    fold = fold_ids(groups, n_folds, seed)
    pred = np.zeros_like(y, dtype=np.float64)
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
        inner = fold_ids(g_tr, n_folds, seed + 1000)
        err = np.zeros(len(alphas))
        for j in range(n_folds):
            i_tr, i_te = inner != j, inner == j
            err += ((ridge_path(X_tr[i_tr], y_tr[i_tr], X_tr[i_te], alphas) - y_tr[i_te]) ** 2).sum(1)
        pred[te] = ridge_path(X_tr, y_tr, X[te], [alphas[err.argmin()]])[0]
    return pred


def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
