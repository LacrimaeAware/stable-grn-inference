"""Counterfactual factor atlas: discover reusable sub-features and test which are
class-DEFINING (core) versus transferable nuisance/shortcut factors.

The idea (from the Track B "latent factor atlas" line): an example is not just a class
label, it is a class PLUS a set of reusable sub-features that cut across classes (a digit
9 that is also red, thick, slanted). A sub-feature is *core* to a class only if:

  - NECESSITY: removing it from the class's examples breaks the class, and
  - SUFFICIENCY: adding it to a rival class's examples converts them.

If a feature can be moved across classes without changing identity, it is a transferable
nuisance/style factor, not part of the class definition. A *shortcut* is a nuisance factor
that happens to be spuriously correlated with a class in the training data (all 9s are red);
the counterfactual test is supposed to see through that correlation.

This module provides:
  - ``make_factor_atlas_data``: a controlled generator with planted core / nuisance /
    shortcut factors AND paired (with/without factor) examples, so we know the ground truth.
  - ``discover_factor_directions``: recover factor directions from paired intervention
    deltas without using factor labels (clustering), scored by ARI.
  - ``counterfactual_necessity_sufficiency``: the remove/add test, per (class, factor).
  - ``held_out_combination_accuracy``: the anti-overfitting payoff -- train on a biased
    class x factor distribution, test on UNSEEN combinations, comparing a raw classifier to
    one that projects out discovered nuisance directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FactorAtlasData:
    X: np.ndarray                 # n x dims
    classes: np.ndarray           # n,  class label in {0, 1}
    factor_on: np.ndarray         # n x n_factors, binary
    factor_kind: list[str]        # 'core' | 'nuisance' | 'shortcut' per factor
    directions: np.ndarray        # n_factors x dims, the planted factor directions
    pair_base_index: np.ndarray   # n, index of the matched factor-OFF example (for deltas)
    delta_factor_id: np.ndarray   # m, which factor each discovery-delta toggles
    deltas: np.ndarray            # m x dims, Phi(with) - Phi(without) for one factor at a time
    meta: dict = field(default_factory=dict)


def make_factor_atlas_data(*, n_per_class: int = 400, dims: int = 30, n_nuisance: int = 2,
                           core_strength: float = 3.0, factor_strength: float = 2.0,
                           noise: float = 0.6, shortcut_p: float = 0.9, seed: int = 0) -> FactorAtlasData:
    """Two-class data with planted factors. Class is defined by a 'core' direction. There is
    one 'shortcut' factor (separate direction, but on with prob ``shortcut_p`` for class 1 and
    1-p for class 0 -> spuriously correlated) and ``n_nuisance`` independent nuisance factors
    (on with prob 0.5, no class correlation). Also emits one-factor-at-a-time paired deltas."""
    rng = np.random.default_rng(seed)

    def _unit():
        v = rng.normal(size=dims)
        return v / np.linalg.norm(v)

    core_dir = _unit() * core_strength
    factor_kind = ["shortcut"] + ["nuisance"] * n_nuisance
    dirs = np.vstack([_unit() * factor_strength for _ in factor_kind])

    n = 2 * n_per_class
    classes = np.array([0] * n_per_class + [1] * n_per_class)
    base = np.outer(classes, core_dir)             # class signal
    factor_on = np.zeros((n, len(factor_kind)), dtype=int)
    for f, kind in enumerate(factor_kind):
        if kind == "shortcut":
            p = np.where(classes == 1, shortcut_p, 1 - shortcut_p)
            factor_on[:, f] = (rng.random(n) < p).astype(int)
        else:
            factor_on[:, f] = (rng.random(n) < 0.5).astype(int)
    X = base + factor_on @ dirs + rng.normal(0.0, noise, size=(n, dims))

    # paired deltas for discovery: for each factor, perturb a fresh OFF example to ON
    delta_rows, delta_id = [], []
    for f in range(len(factor_kind)):
        for _ in range(150):
            y = int(rng.random() < 0.5)
            off = y * core_dir + rng.normal(0.0, noise, size=dims)
            on = off + dirs[f]
            delta_rows.append(on - off)
            delta_id.append(f)
    return FactorAtlasData(
        X=X, classes=classes, factor_on=factor_on, factor_kind=factor_kind, directions=dirs,
        pair_base_index=np.arange(n), delta_factor_id=np.array(delta_id),
        deltas=np.vstack(delta_rows),
        meta={"core_direction": core_dir, "core_strength": core_strength},
    )


def discover_factor_directions(deltas: np.ndarray, true_factor_id: np.ndarray,
                               n_factors: int) -> dict:
    """Cluster one-factor-at-a-time deltas (no factor labels) and recover a direction per
    cluster. Returns ARI vs the true factor identity and the recovered unit directions."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    km = KMeans(n_clusters=n_factors, n_init=10, random_state=0).fit(deltas)
    labels = km.labels_
    ari = float(adjusted_rand_score(true_factor_id, labels))
    dirs = []
    for c in range(n_factors):
        d = deltas[labels == c].mean(axis=0)
        dirs.append(d / (np.linalg.norm(d) or 1.0))
    return {"ari": ari, "labels": labels, "directions": np.vstack(dirs)}


def _fit_classifier(X, y):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000).fit(X, y)


def counterfactual_necessity_sufficiency(data: FactorAtlasData, *, target_class: int = 1) -> list[dict]:
    """For each planted factor, the remove/add test against a classifier trained on the
    (biased) data. NECESSITY R = fraction of target-class examples carrying the factor that
    STILL classify as target after the factor direction is removed (high R = not necessary).
    SUFFICIENCY A = fraction of non-target examples that FLIP to target after the factor is
    added (high A = sufficient). A core factor: low R, high A. Nuisance/shortcut: high R, low A."""
    clf = _fit_classifier(data.X, data.classes)
    rows = []
    # include the core direction as factor index -1 (positive control)
    factors = [("core", data.meta["core_direction"], None)]
    for f, kind in enumerate(data.factor_kind):
        factors.append((kind, data.directions[f], f))
    for kind, direction, f in factors:
        if f is None:
            has = data.classes == target_class                       # core "on" == being class 1
        else:
            has = (data.factor_on[:, f] == 1)
        # necessity: target-class examples that carry it; remove and re-predict
        idx_nec = np.where((data.classes == target_class) & has)[0]
        if len(idx_nec):
            removed = data.X[idx_nec] - direction
            R = float(np.mean(clf.predict(removed) == target_class))
        else:
            R = float("nan")
        # sufficiency: non-target examples without it; add and re-predict
        idx_suf = np.where((data.classes != target_class) & (~has))[0]
        if len(idx_suf):
            added = data.X[idx_suf] + direction
            A = float(np.mean(clf.predict(added) == target_class))
        else:
            A = float("nan")
        rows.append({"factor": kind, "necessity_R": R, "sufficiency_A": A,
                     "core_score": (1 - R) * A if np.isfinite(R) and np.isfinite(A) else float("nan")})
    return rows


def project_out_directions(X: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Remove the span of ``directions`` (rows) from X (orthogonal projection out)."""
    if directions is None or len(directions) == 0:
        return X.copy()
    Q = np.linalg.qr(directions.T)[0]            # dims x r orthonormal basis of the span
    return X - (X @ Q) @ Q.T


def held_out_combination_accuracy(data: FactorAtlasData, *, nuisance_directions: np.ndarray,
                                  seed: int = 0) -> dict:
    """Anti-overfitting test. Train where the shortcut factor is glued to class 1 (its native
    correlation), then TEST on the OPPOSITE combinations: class-1 examples WITHOUT the shortcut
    and class-0 examples WITH it. Compare a raw classifier (overfits the shortcut) to one that
    projects out the discovered nuisance directions first. Higher held-out accuracy = the
    factored representation did not confuse the shortcut for class identity."""
    sc = 0  # shortcut factor index (always 0 in the generator)
    has_sc = data.factor_on[:, sc] == 1
    # "consistent" combos (train): class1+shortcut, class0+no-shortcut
    train_mask = ((data.classes == 1) & has_sc) | ((data.classes == 0) & ~has_sc)
    # "flipped" combos (held-out test): class1 without shortcut, class0 with shortcut
    test_mask = ((data.classes == 1) & ~has_sc) | ((data.classes == 0) & has_sc)
    Xtr, ytr = data.X[train_mask], data.classes[train_mask]
    Xte, yte = data.X[test_mask], data.classes[test_mask]

    raw = _fit_classifier(Xtr, ytr)
    raw_acc = float(np.mean(raw.predict(Xte) == yte))

    Xtr_f = project_out_directions(Xtr, nuisance_directions)
    Xte_f = project_out_directions(Xte, nuisance_directions)
    fac = _fit_classifier(Xtr_f, ytr)
    fac_acc = float(np.mean(fac.predict(Xte_f) == yte))
    return {"raw_heldout_acc": raw_acc, "factored_heldout_acc": fac_acc,
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())}
