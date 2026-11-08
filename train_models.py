"""Train and save the five reference classifiers (run on BITS Virtual Lab).

Education-only project: replace the methodology and reflect on choices before submitting.
Source: UCI Breast Cancer Wisconsin (Diagnostic) as distributed in scikit-learn.
Data split and preprocessing deliberately occur *before* test evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'model'
RESULTS_DIR = BASE_DIR / 'results'
SEED = 42
TEST_SIZE = 0.20
FEATURE_COLUMNS = None
MODEL_SPECS = {
    'Logistic Regression': ('logistic_regression.joblib', make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED))),
    'Decision Tree': ('decision_tree.joblib', DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED)),
    'K-Nearest Neighbors': ('knn.joblib', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
    'Gaussian Naive Bayes': ('gaussian_nb.joblib', GaussianNB()),
    'Support Vector Machine': ('svm_rbf.joblib', make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=SEED))),
}


def run_training() -> pd.DataFrame:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    # The scikit-learn source encodes 0=malignant, 1=benign.
    # For clear positive-class sensitivity, this project uses 1=malignant, 0=benign.
    outcome = pd.Series((dataset.target == 0).astype(int), name='diagnosis')
    assert features.shape == (569, 30)
    assert not features.isna().any().any()
    assert outcome.isin([0, 1]).all()

    X_train, X_test, y_train, y_test = train_test_split(
        features, outcome, test_size=TEST_SIZE, random_state=SEED, stratify=outcome,
    )
    test_frame = X_test.copy()
    test_frame['diagnosis'] = y_test.to_numpy()
    test_frame.to_csv(BASE_DIR / 'test_data.csv', index=False)

    rows = []
    for label, (filename, classifier) in MODEL_SPECS.items():
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)
        probability = classifier.predict_proba(X_test)[:, list(classifier.classes_).index(1)]
        tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
        rows.append({
            'ML Model Name': label,
            'Accuracy': accuracy_score(y_test, predicted),
            'AUC': roc_auc_score(y_test, probability),
            'Precision': precision_score(y_test, predicted, zero_division=0),
            'Recall': recall_score(y_test, predicted, zero_division=0),
            'F1': f1_score(y_test, predicted, zero_division=0),
            'MCC': matthews_corrcoef(y_test, predicted),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        })
        joblib.dump(classifier, MODEL_DIR / filename)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False, float_format='%.9f')
    details = {
        'dataset': 'UCI Breast Cancer Wisconsin (Diagnostic) via scikit-learn',
        'source_url': 'https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic',
        'instance_count': len(features), 'feature_count': features.shape[1],
        'train_rows': len(X_train), 'test_rows': len(X_test),
        'training_class_counts': {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        'test_class_counts': {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        'positive_class': 'malignant (1)', 'negative_class': 'benign (0)',
        'feature_columns': list(features.columns),
        'target_column': 'diagnosis', 'random_state': SEED,
        'test_size': TEST_SIZE, 'sklearn_version': sklearn.__version__,
        'model_files': {label: fn for label, (fn, _) in MODEL_SPECS.items()},
    }
    (MODEL_DIR / 'metadata.json').write_text(json.dumps(details, indent=2), encoding='utf-8')

    # Simple explanatory chart used in the report and README.
    fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=150)
    x = np.arange(len(metrics))
    bar_width = 0.36
    ax.bar(x - bar_width / 2, metrics['F1'], width=bar_width, label='F1')
    ax.bar(x + bar_width / 2, metrics['MCC'], width=bar_width, label='MCC')
    ax.set_xticks(x, ['Logistic\nRegression', 'Decision\nTree', 'KNN', 'Naive\nBayes', 'SVM'])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score (higher is better)')
    ax.set_title('Held-out test performance, 114 observations')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'performance_comparison.png', bbox_inches='tight')
    plt.close(fig)
    print('Dataset 569 x 30 | train', len(X_train), '| test', len(X_test))
    print(metrics.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    return metrics


if __name__ == '__main__':
    run_training()
