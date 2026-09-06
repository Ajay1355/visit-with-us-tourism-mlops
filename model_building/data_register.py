from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'data' / 'tourism.csv'

EXPECTED_COLUMNS = [
    'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
    'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
]

df = pd.read_csv(DATA_FILE)
missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
extra_columns = [c for c in df.columns if c not in EXPECTED_COLUMNS]

if missing_columns:
    raise ValueError(f'Missing expected columns: {missing_columns}')
if not set(df['ProdTaken'].dropna().unique()).issubset({0, 1}):
    raise ValueError('ProdTaken must contain only 0 and 1.')
if df['CustomerID'].duplicated().any():
    raise ValueError('CustomerID must be unique.')

print('DATA REGISTRATION SUCCESSFUL')
print('Rows:', len(df))
print('Columns:', len(df.columns))
print('Duplicate rows:', int(df.duplicated().sum()))
print('Missing values:', int(df.isna().sum().sum()))
print('Extra columns:', extra_columns)
print('Target distribution:')
print(df['ProdTaken'].value_counts().sort_index())
print('Target percentage:')
print((df['ProdTaken'].value_counts(normalize=True).sort_index() * 100).round(2))