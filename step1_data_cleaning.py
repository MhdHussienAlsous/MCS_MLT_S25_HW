import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data_all1.csv', sep=';')


print(df.head())

print("\n1. Missing Values:")
print(df.isnull().sum())

print(df.dtypes)


df_clean = df.copy()

# 1. Handle missing activity_date
missing_dates = df_clean['activity_date'].isnull().sum()
if missing_dates > 0:
    df_clean = df_clean.dropna(subset=['activity_date'])


# 2. Convert activity_date to datetime
df_clean['activity_date'] = pd.to_datetime(df_clean['activity_date'], format='%d/%m/%Y', errors='coerce')
# Remove any rows where date conversion failed
invalid_dates = df_clean['activity_date'].isnull().sum()
if invalid_dates > 0:
    df_clean = df_clean.dropna(subset=['activity_date'])


# 3. Handle missing values in other columns
df_clean['who_id'] = df_clean['who_id'].fillna('Unknown')
df_clean['opportunity_id'] = df_clean['opportunity_id'].fillna('No_Opportunity')
df_clean['opportunity_stage'] = df_clean['opportunity_stage'].fillna('no_opp')


# 4. Clean and standardize text columns
for col in ['account_id', 'SourceSystem', 'opportunity_stage', 'types', 'Country', 'solution']:
    df_clean[col] = df_clean[col].astype(str).str.strip()

df_clean['opportunity_stage'] = df_clean['opportunity_stage'].str.lower()

# 5. Convert is_lead to integer
df_clean['is_lead'] = pd.to_numeric(df_clean['is_lead'], errors='coerce').fillna(1).astype(int)

# 6. Sort by account and date
df_clean = df_clean.sort_values(['account_id', 'activity_date']).reset_index(drop=True)

# 7. Create additional features

# Extract date components
df_clean['year'] = df_clean['activity_date'].dt.year
df_clean['month'] = df_clean['activity_date'].dt.month
df_clean['day_of_week'] = df_clean['activity_date'].dt.dayofweek

# Create touch sequence number for each account
df_clean['touch_sequence'] = df_clean.groupby('account_id').cumcount() + 1

# Calculate days since last touch for each account
df_clean['days_since_last_touch'] = df_clean.groupby('account_id')['activity_date'].diff().dt.days

# Create journey outcome (Won, Lost, or Ongoing)
df_clean['journey_outcome'] = df_clean.groupby('account_id')['opportunity_stage'].transform(
    lambda x: 'Won' if 'won' in x.values else ('Lost' if 'lost' in x.values else 'Ongoing')
)

# 8. Remove duplicates
duplicates = df_clean.duplicated().sum()
if duplicates > 0:
    df_clean = df_clean.drop_duplicates()


output_file = 'data_cleaned.csv'
df_clean.to_csv(output_file, index=False)
print(f"✓ Cleaned data saved to: {output_file}")

# Save summary statistics
summary_file = 'step1_data_cleaning_summary.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("DATA CLEANING SUMMARY\n")
    f.write("="*50 + "\n\n")
    f.write(f"Original rows: {len(df)}\n")
    f.write(f"Cleaned rows: {len(df_clean)}\n")
    f.write(f"Rows removed: {len(df) - len(df_clean)}\n")
    f.write(f"Percentage retained: {(len(df_clean)/len(df)*100):.2f}%\n\n")
    
    f.write("Columns in cleaned dataset:\n")
    for col in df_clean.columns:
        f.write(f"  - {col}\n")
    
    f.write("\nMissing values after cleaning:\n")
    f.write(df_clean.isnull().sum().to_string())
    
    f.write("\n\nData types:\n")
    f.write(df_clean.dtypes.to_string())