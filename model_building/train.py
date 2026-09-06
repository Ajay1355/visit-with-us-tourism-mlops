from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / 'artifacts'
DEPLOY_DIR = ROOT / 'deployment'
MLRUNS_DIR = ROOT / 'mlruns'
DEPLOY_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(ARTIFACT_DIR / 'train.csv')
test_df = pd.read_csv(ARTIFACT_DIR / 'test.csv')

X_train = train_df.drop(columns=['ProdTaken'])
y_train = train_df['ProdTaken'].astype(int)
X_test = test_df.drop(columns=['ProdTaken'])
y_test = test_df['ProdTaken'].astype(int)

numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=['number']).columns.tolist()

numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipe, numeric_features),
    ('cat', categorical_pipe, categorical_features)
])

model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=2
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgbclassifier', model)
])

param_grid = {
    'xgbclassifier__n_estimators': [200, 300],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.05, 0.10],
    'xgbclassifier__subsample': [0.8],
    'xgbclassifier__colsample_bytree': [0.8]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid=param_grid, scoring='f1', cv=cv, n_jobs=2, verbose=1)
search.fit(X_train, y_train)

best_model = search.best_estimator_
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.50).astype(int)

metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
    'f1': float(f1_score(y_test, y_pred, zero_division=0)),
    'roc_auc': float(roc_auc_score(y_test, y_prob)),
    'cv_best_f1': float(search.best_score_),
}

best_params = {k: (float(v) if hasattr(v, 'item') and not isinstance(v, int) else v.item() if hasattr(v, 'item') else v) for k, v in search.best_params_.items()}

joblib.dump(best_model, DEPLOY_DIR / 'model.joblib')
(ARTIFACT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
(ARTIFACT_DIR / 'best_params.json').write_text(json.dumps(best_params, indent=2), encoding='utf-8')

mlflow.set_tracking_uri((MLRUNS_DIR).resolve().as_uri())
mlflow.set_experiment('visit-with-us-tourism-purchase')
with mlflow.start_run(run_name='xgboost_gridsearch_best_model'):
    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)
    mlflow.log_param('classification_threshold', 0.50)
    mlflow.log_artifact(str(DEPLOY_DIR / 'model.joblib'))
    mlflow.log_artifact(str(ARTIFACT_DIR / 'metrics.json'))

experiment_log = {
    'experiment': 'visit-with-us-tourism-purchase',
    'model': 'XGBClassifier',
    'best_params': best_params,
    'metrics': metrics,
    'threshold': 0.50
}
(ARTIFACT_DIR / 'experiment_log.json').write_text(json.dumps(experiment_log, indent=2), encoding='utf-8')

print('MODEL TRAINING SUCCESSFUL')
print('Best parameters:', best_params)
print('Metrics:', json.dumps(metrics, indent=2))
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))