from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'data' / 'tourism.csv'
ARTIFACT_DIR = ROOT / 'artifacts'
ARTIFACT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
df = df.drop(columns=['CustomerID'])

X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

train_df = X_train.copy()
train_df['ProdTaken'] = y_train.values
test_df = X_test.copy()
test_df['ProdTaken'] = y_test.values

train_df.to_csv(ARTIFACT_DIR / 'train.csv', index=False)
test_df.to_csv(ARTIFACT_DIR / 'test.csv', index=False)

print('DATA PREPARATION SUCCESSFUL')
print('Train shape:', train_df.shape)
print('Test shape:', test_df.shape)
print('Train target distribution:')
print(y_train.value_counts(normalize=True).sort_index().round(4))
print('Test target distribution:')
print(y_test.value_counts(normalize=True).sort_index().round(4))