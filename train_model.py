"""Train the machine failure model and save it to model.joblib.

Run:  python train_model.py
"""
import os

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_FILE = 'ai4i2020.csv'
MODEL_FILE = 'model.joblib'

# The five failure-mode columns define the target, so they are excluded as inputs
# (data leakage). UDI and Product ID are identifiers with no predictive value.
LEAKY_COLUMNS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
ID_COLUMNS = ['UDI', 'Product ID']


def load_features(path=DATA_FILE):
    """Return the input table X and the target y."""
    df = pd.read_csv(path)
    X = df.drop(columns=ID_COLUMNS + ['Machine failure'] + LEAKY_COLUMNS)
    X = pd.get_dummies(X, columns=['Type'], drop_first=True, dtype=int)
    y = df['Machine failure']
    return X, y


def train_and_save(path=DATA_FILE, out=MODEL_FILE, verbose=True):
    """Train a class-weighted Random Forest on a stratified 80/20 split and save it."""
    X, y = load_features(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                   random_state=42, n_jobs=1)
    model.fit(X_train, y_train)

    if verbose:
        pred = model.predict(X_test)
        print('Confusion matrix (rows = actual, columns = predicted):')
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, digits=3))

    bundle = {'model': model,
              'columns': list(X.columns),
              'sklearn_version': sklearn.__version__}
    joblib.dump(bundle, out)
    if verbose:
        print(f'Saved {out}')
    return bundle


def load_or_train(path=DATA_FILE, out=MODEL_FILE):
    """Load the saved model; retrain if it is missing, unreadable, or was saved
    with a different scikit-learn version than the one installed."""
    if os.path.exists(out):
        try:
            bundle = joblib.load(out)
            if bundle.get('sklearn_version') == sklearn.__version__:
                return bundle
        except Exception:
            pass
    return train_and_save(path, out, verbose=False)


if __name__ == '__main__':
    train_and_save()
