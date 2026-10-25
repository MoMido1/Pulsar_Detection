"""Leakage-safe pulsar-candidate classification experiments.

Each CSV row contains numeric features followed by a binary label. Preprocessing is
fitted on each training fold; the held-out test set is used only for final evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC, SVC


class ClassConditionalGMM(BaseEstimator, ClassifierMixin):
    """Binary class-conditional GMM with log-likelihood-ratio decision scores."""

    def __init__(self, n_components=3, covariance_type="full", random_state=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        self.models_ = []
        for label in self.classes_:
            class_data = X[y == label]
            if class_data.shape[0] == 0:
                raise ValueError(f"No training samples for class {label}.")
            model = GaussianMixture(
                n_components=min(int(self.n_components), len(class_data)),
                covariance_type=self.covariance_type,
                reg_covar=1e-4,
                random_state=self.random_state,
                n_init=1,
            )
            model.fit(class_data)
            self.models_.append(model)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return self.models_[1].score_samples(X) - self.models_[0].score_samples(X)

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def load_dataset(path: Path):
    """Read comma-separated features with the binary label in the final column."""
    try:
        data = np.loadtxt(path, delimiter=",", dtype=float, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path} as comma-separated numeric data: {exc}") from exc
    if data.shape[1] < 2 or not np.all(np.isfinite(data)):
        raise ValueError(f"{path} must contain finite numeric features and a label column.")
    labels = data[:, -1]
    if not np.all(np.isin(np.unique(labels), [0.0, 1.0])):
        raise ValueError(f"{path} must use labels 0 and 1 in the final column.")
    y = labels.astype(int)
    if np.unique(y).size != 2:
        raise ValueError(f"{path} must contain examples from both classes.")
    return data[:, :-1], y


def build_pipeline(classifier, n_samples, use_pca=True):
    # The quantile transform and PCA are inside the pipeline, so each CV fold
    # fits them only on its training partition.
    steps = [("gaussianize", QuantileTransformer(
        n_quantiles=min(1000, n_samples),
        output_distribution="normal",
        random_state=0,
    ))]
    if use_pca:
        steps.append(("pca", PCA(n_components=4, svd_solver="full")))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def continuous_scores(model, X):
    """Return continuous scores where larger values favour class 1."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).reshape(-1)
    if hasattr(model, "predict_log_proba"):
        logp = model.predict_log_proba(X)
        classes = list(model.classes_)
        return logp[:, classes.index(1)] - logp[:, classes.index(0)]
    raise TypeError(f"{type(model).__name__} has no supported continuous score method.")


def error_rates(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    positives, negatives = y_true == 1, y_true == 0
    fnr = np.sum(positives & (y_pred == 0)) / max(np.sum(positives), 1)
    fpr = np.sum(negatives & (y_pred == 1)) / max(np.sum(negatives), 1)
    return float(fnr), float(fpr)


def normalized_dcf(y_true, scores, prior=0.5, threshold=0.0):
    y_pred = (np.asarray(scores) > threshold).astype(int)
    fnr, fpr = error_rates(y_true, y_pred)
    return float((prior * fnr + (1.0 - prior) * fpr) / min(prior, 1.0 - prior))


def min_dcf(y_true, scores, prior=0.5):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    thresholds = np.concatenate(([-np.inf], np.unique(scores), [np.inf]))
    return min(normalized_dcf(y_true, scores, prior, t) for t in thresholds)


def out_of_fold_scores(estimator, X, y, folds=5):
    """Every training sample is scored by a model that did not train on that sample."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    scores = np.full(len(y), np.nan, dtype=float)
    for train_idx, valid_idx in splitter.split(X, y):
        fold_model = clone(estimator)
        fold_model.fit(X[train_idx], y[train_idx])
        scores[valid_idx] = continuous_scores(fold_model, X[valid_idx])
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("Cross-validation failed to produce finite out-of-fold scores.")
    return scores


def evaluate(name, estimator, X_train, y_train, X_test, y_test, prior):
    # Train the calibrator on out-of-fold scores only; never use in-sample scores
    # or test labels for calibration.
    oof = out_of_fold_scores(estimator, X_train, y_train)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    calibrator.fit(oof.reshape(-1, 1), y_train)

    train_prior = float(np.mean(y_train))
    estimator.fit(X_train, y_train)
    test_raw = continuous_scores(estimator, X_test)
    # Logistic calibration estimates posterior log-odds under the training prior;
    # subtract that prior's log-odds to obtain an approximate log-likelihood ratio.
    calibrated_llr = calibrator.decision_function(test_raw.reshape(-1, 1)) - np.log(
        train_prior / (1.0 - train_prior)
    )
    threshold = np.log((1.0 - prior) / prior)
    predictions = (calibrated_llr > threshold).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions) * 100.0,
        "min_dcf_raw": min_dcf(y_test, test_raw, prior),
        "act_dcf_calibrated": normalized_dcf(y_test, calibrated_llr, prior, threshold),
        "min_dcf_calibrated": min_dcf(y_test, calibrated_llr, prior),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare pulsar classifiers with leakage-safe preprocessing.")
    parser.add_argument("--train", type=Path, default=Path("data/Train.txt"))
    parser.add_argument("--test", type=Path, default=Path("data/Test.txt"))
    parser.add_argument("--prior", type=float, default=0.5, help="Target positive-class prior for DCF.")
    args = parser.parse_args()
    if not 0.0 < args.prior < 1.0:
        parser.error("--prior must be strictly between 0 and 1.")

    X_train, y_train = load_dataset(args.train)
    X_test, y_test = load_dataset(args.test)
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and test files must contain the same number of features.")
    if X_train.shape[1] < 4:
        raise ValueError("PCA(4) requires at least four input features.")
    if min(np.bincount(y_train, minlength=2)) < 5:
        raise ValueError("Each training class needs at least five samples for 5-fold calibration.")
    if set(np.unique(y_test)) != {0, 1}:
        raise ValueError("The test set must contain both classes.")

    models = [
        ("Gaussian Naive Bayes", GaussianNB()),
        ("Logistic Regression", LogisticRegression(C=1.0, max_iter=3000)),
        ("Linear SVM", LinearSVC(C=1.0, dual=False, max_iter=10000, random_state=0)),
        ("RBF SVM", SVC(C=1.0, kernel="rbf", gamma="scale")),
        ("Class-conditional GMM (3 components)", ClassConditionalGMM(n_components=3)),
    ]
    print(f"Train: {len(y_train)} samples | Test: {len(y_test)} samples | Features: {X_train.shape[1]}")
    print(f"Training classes: 0={np.sum(y_train == 0)}, 1={np.sum(y_train == 1)}")
    print("Preprocessing: fold-fitted Gaussianization + PCA(4); calibration uses out-of-fold training scores.")
    print("The test set is held out until final evaluation.\n")

    results = []
    for name, classifier in models:
        pipeline = build_pipeline(classifier, len(y_train), use_pca=True)
        result = evaluate(name, pipeline, X_train, y_train, X_test, y_test, args.prior)
        results.append(result)
        print(
            f"{name}: accuracy={result['accuracy']:.2f}% | "
            f"minDCF(raw)={result['min_dcf_raw']:.4f} | "
            f"actDCF(calibrated)={result['act_dcf_calibrated']:.4f} | "
            f"minDCF(calibrated)={result['min_dcf_calibrated']:.4f}"
        )
    best = min(results, key=lambda row: row["min_dcf_calibrated"])
    print(f"\nLowest calibrated minDCF on this test split: {best['model']} ({best['min_dcf_calibrated']:.4f}).")
    print("This is one held-out split; treat the comparison as exploratory, not as an unbiased model-selection estimate.")


if __name__ == "__main__":
    main()
