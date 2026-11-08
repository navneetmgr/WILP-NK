"""Interactive five-classifier evaluation studio for BITS WILP Machine Learning.

Crucially, this app only loads models saved by train_models.py.
Uploading data NEVER triggers training or refitting.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'model'

st.set_page_config(
    page_title='WDBC | Five-Model Evaluation Studio',
    page_icon='🔬',
    layout='wide',
)


@st.cache_data

def get_metadata():
    return json.loads((MODEL_DIR / 'metadata.json').read_text(encoding='utf-8'))


@st.cache_resource

def get_classifier(model_file: str):
    """joblib files are trusted, local assets committed with this repository."""
    return joblib.load(MODEL_DIR / model_file)


def validate_dataset(data: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, np.ndarray]:
    required = metadata['feature_columns'] + [metadata['target_column']]
    missing = [field for field in required if field not in data.columns]
    if missing:
        raise ValueError('The CSV is missing required columns: ' + ', '.join(missing))
    if data.empty:
        raise ValueError('The uploaded CSV has no observations.')
    X = data.loc[:, metadata['feature_columns']].copy()
    for col in X:
        X[col] = pd.to_numeric(X[col], errors='raise')
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('All model inputs must be finite numbers with no missing values.')
    y_column = pd.to_numeric(data[metadata['target_column']], errors='raise')
    if y_column.isna().any() or not y_column.isin([0, 1]).all():
        raise ValueError("Target 'diagnosis' must only contain 0 (benign) or 1 (malignant), without blanks.")
    return X, y_column.to_numpy(dtype=int)


def calculate_results(model, X: pd.DataFrame, y: np.ndarray):
    prediction = model.predict(X)
    classes = list(model.classes_)
    positive_probability = model.predict_proba(X)[:, classes.index(1)]
    auc = float(roc_auc_score(y, positive_probability)) if len(np.unique(y)) == 2 else np.nan
    return {
        'Accuracy': float(accuracy_score(y, prediction)),
        'AUC': auc,
        'Precision': float(precision_score(y, prediction, zero_division=0)),
        'Recall': float(recall_score(y, prediction, zero_division=0)),
        'F1': float(f1_score(y, prediction, zero_division=0)),
        'MCC': float(matthews_corrcoef(y, prediction)),
    }, prediction, positive_probability


def confusion_figure(y: np.ndarray, prediction: np.ndarray):
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    image = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), va='center', ha='center',
                    fontsize=17, color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    ax.set_xticks([0, 1], ['Benign (0)', 'Malignant (1)'])
    ax.set_yticks([0, 1], ['Benign (0)', 'Malignant (1)'])
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('Actual class')
    ax.set_title('Confusion matrix')
    fig.tight_layout()
    return fig


def roc_figure(y, probabilities):
    fpr, tpr, _ = roc_curve(y, probabilities)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(fpr, tpr, lw=2, label='Selected model')
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', label='Chance')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curve', xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc='lower right')
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def show_dashboard():
    metadata = get_metadata()
    st.title('🔬 WDBC | Five-Model Evaluation Studio')
    st.caption('End-to-end classification benchmark  •  UCI dataset  •  saved models  •  live test-set evaluation')
    st.markdown('**Core question:** How differently do five standard algorithms classify the same observations?')
    st.info('**Positive class:** malignant = 1; benign = 0. All models are loaded from disk, **never retrained** on the uploaded CSV.')

    with st.sidebar:
        st.header('Experiment controls')
        chosen_model = st.selectbox('Select classifier', list(metadata['model_files']), index=4)
        st.divider()
        st.subheader('Test data upload')
        uploaded = st.file_uploader('Upload CSV with 30 features + diagnosis', type=['csv'], help='Use the downloadable 114-row test CSV to reproduce the reference measurements.')
        st.caption('No real patient records should be uploaded. This app is an educational example, not a diagnostic tool.')
        st.divider()
        st.markdown('**Reference experiment**')
        st.write(f"Dataset: {metadata['instance_count']} rows × {metadata['feature_count']} features")
        st.write(f"Train / test: {metadata['train_rows']} / {metadata['test_rows']}")
        st.write(f"Split seed: {metadata['random_state']}, stratified")
        st.markdown('[Original dataset (UCI)](' + metadata['source_url'] + ')')

    if uploaded is not None:
        try:
            test_df = pd.read_csv(uploaded)
            st.success('Uploaded test CSV selected. Evaluations below use your uploaded data.')
        except Exception as err:
            st.error(f'Unable to parse CSV: {err}')
            st.stop()
    else:
        test_df = pd.read_csv(ROOT / 'test_data.csv')
        st.caption('Using the packaged 114-row hold-out test CSV. To test another set, upload a compatible CSV in the sidebar.')
        st.download_button('Download reference test_data.csv', data=(ROOT / 'test_data.csv').read_bytes(), file_name='test_data.csv', mime='text/csv')

    try:
        X, y = validate_dataset(test_df, metadata)
    except (ValueError, TypeError) as err:
        st.error(str(err))
        with st.expander('Expected column names'):
            st.code(', '.join(metadata['feature_columns'] + [metadata['target_column']]))
        st.stop()

    first, second, third = st.columns(3)
    first.metric('Evaluated records', f'{len(y):,}')
    second.metric('Features', str(X.shape[1]))
    third.metric('Positive (malignant) labels', f'{np.mean(y):.1%}')

    comparisons = []
    outputs = {}
    for name, filename in metadata['model_files'].items():
        model = get_classifier(filename)
        measures, predictions, probability = calculate_results(model, X, y)
        comparisons.append({'ML Model Name': name, **measures})
        outputs[name] = (predictions, probability)
    table = pd.DataFrame(comparisons)

    selected = table.loc[table['ML Model Name'] == chosen_model].iloc[0]
    st.subheader(f'Selected model: {chosen_model}')
    cols = st.columns(6)
    for col, key in zip(cols, ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']):
        value = selected[key]
        col.metric(key, f'{value:.4f}' if np.isfinite(value) else 'N/A')

    tabs = st.tabs(['Confusion matrix + ROC', 'Classification report', 'Compare all five', 'Data preview'])
    predicted, probability = outputs[chosen_model]
    with tabs[0]:
        left, right = st.columns(2)
        with left:
            fig = confusion_figure(y, predicted)
            st.pyplot(fig)
            plt.close(fig)
        with right:
            if len(np.unique(y)) == 2:
                fig = roc_figure(y, probability)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning('ROC/AUC is undefined when the uploaded set contains only one actual class.')
            st.markdown('**Reading the confusion matrix:** rows are actual classes; columns are predicted classes. False negatives are malignant (1) records predicted as benign (0).')
    with tabs[1]:
        report = classification_report(y, predicted, labels=[0, 1],
                                       target_names=['Benign (0)', 'Malignant (1)'], zero_division=0)
        st.code(report, language='text')
        st.caption('Macro averages weight the classes equally. The six rubric scores use malignant as the positive class.')
    with tabs[2]:
        display_df = table.copy()
        metric_cols = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        st.dataframe(display_df.style.format({key: '{:.4f}' for key in metric_cols}), use_container_width=True, hide_index=True)
        st.download_button('Download calculated comparison CSV', data=table.to_csv(index=False).encode(),
                           file_name='model_comparison_uploaded_data.csv', mime='text/csv')
        st.caption('All five scores are recomputed on the same test CSV using pre-trained artifacts.')
        st.bar_chart(table.set_index('ML Model Name')[['F1', 'MCC']])
        if len(np.unique(y)) == 2:
            best_f1_idx = table['F1'].idxmax()
            st.markdown('**Highest F1 on this CSV:** ' + str(table.loc[best_f1_idx, 'ML Model Name']))
    with tabs[3]:
        st.dataframe(test_df.head(12), use_container_width=True)
        st.caption('The target column is mandatory, even for the uploaded data. Extra columns are ignored.')

    predictions_df = test_df.copy()
    predictions_df['predicted_diagnosis'] = predicted
    predictions_df['predicted_malignant_probability'] = probability
    st.download_button('Download selected model predictions',
                       data=predictions_df.to_csv(index=False).encode(),
                       file_name='predictions.csv', mime='text/csv')
    st.divider()
    st.caption('Research/teaching only — the WDBC example does not constitute an approved clinical diagnostic service.')


if __name__ == '__main__':
    try:
        show_dashboard()
    except FileNotFoundError as err:
        st.error(f'Missing bundled training artifact: {err}. Run train_models.py and commit all model files.')
