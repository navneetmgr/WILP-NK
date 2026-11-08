"""Basic offline checks; run after training or after copying project to BITS Lab."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

root = Path(__file__).resolve().parent
meta = json.loads((root / 'model' / 'metadata.json').read_text())
test_df = pd.read_csv(root / 'test_data.csv')
benchmark = pd.read_csv(root / 'results' / 'model_comparison.csv')
assert len(test_df) == 114
assert len(meta['feature_columns']) == 30
assert len(meta['model_files']) == 5
assert set(meta['feature_columns'] + ['diagnosis']) == set(test_df.columns)
X = test_df[meta['feature_columns']]
y = test_df['diagnosis']
for label, filename in meta['model_files'].items():
    model = joblib.load(root / 'model' / filename)
    y_hat = model.predict(X)
    score = model.predict_proba(X)[:, list(model.classes_).index(1)]
    row = benchmark.loc[benchmark['ML Model Name'] == label].iloc[0]
    current = dict(Accuracy=accuracy_score(y, y_hat), AUC=roc_auc_score(y, score),
                   Precision=precision_score(y, y_hat), Recall=recall_score(y, y_hat),
                   F1=f1_score(y, y_hat), MCC=matthews_corrcoef(y, y_hat))
    for k, v in current.items():
        assert np.isclose(row[k], v, atol=1e-8), f'{label} {k}: {row[k]} versus {v}'
    print(f'PASS | {label:<23} F1 {current["F1"]:.4f} | AUC {current["AUC"]:.4f}')
print('ALL CHECKS PASSED: 5 saved models, complete CSV, metrics reproducible.')
