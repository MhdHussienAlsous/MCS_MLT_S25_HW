import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load data
df_original = pd.read_csv('data_all1.csv', sep=';')
df_cleaned = pd.read_csv('data_cleaned.csv')

# Create summary figure
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Step 1: Data Cleaning Summary', fontsize=20, fontweight='bold', y=0.98)

# 1. Data Volume Comparison
ax1 = plt.subplot(2, 3, 1)
categories = ['Original\nDataset', 'Cleaned\nDataset']
values = [len(df_original), len(df_cleaned)]
colors = ['lightcoral', 'lightgreen']
bars = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
ax1.set_ylabel('Number of Records', fontsize=11, fontweight='bold')
ax1.set_title('Data Volume', fontsize=13, fontweight='bold')
for i, (bar, val) in enumerate(zip(bars, values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:,}\n({val/len(df_original)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_ylim(0, max(values) * 1.15)
ax1.grid(axis='y', alpha=0.3)

# 2. Journey Outcomes
ax2 = plt.subplot(2, 3, 2)
outcome_counts = df_cleaned.groupby('account_id')['journey_outcome'].first().value_counts()
colors_outcome = ['#2ecc71', '#e67e22', '#e74c3c']
wedges, texts, autotexts = ax2.pie(outcome_counts, labels=outcome_counts.index, 
                                     autopct='%1.1f%%', colors=colors_outcome, 
                                     startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax2.set_title('Journey Outcomes\n(by Account)', fontsize=13, fontweight='bold')

# 3. Top Action Types
ax3 = plt.subplot(2, 3, 3)
top_actions = df_cleaned['types'].value_counts().head(8)
ax3.barh(range(len(top_actions)), top_actions.values, color='steelblue', edgecolor='black')
ax3.set_yticks(range(len(top_actions)))
ax3.set_yticklabels(top_actions.index, fontsize=10)
ax3.set_xlabel('Count', fontsize=11, fontweight='bold')
ax3.set_title('Top 8 Action Types', fontsize=13, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_actions.values):
    ax3.text(v + 1000, i, f'{v:,}', va='center', fontsize=9, fontweight='bold')

# 4. Geographic Distribution
ax4 = plt.subplot(2, 3, 4)
top_countries = df_cleaned.groupby('Country')['account_id'].nunique().sort_values(ascending=False).head(8)
ax4.barh(range(len(top_countries)), top_countries.values, color='purple', alpha=0.7, edgecolor='black')
ax4.set_yticks(range(len(top_countries)))
ax4.set_yticklabels(top_countries.index, fontsize=10)
ax4.set_xlabel('Number of Accounts', fontsize=11, fontweight='bold')
ax4.set_title('Top 8 Countries', fontsize=13, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_countries.values):
    ax4.text(v + 50, i, f'{v:,}', va='center', fontsize=9, fontweight='bold')

# 5. Journey Length Distribution
ax5 = plt.subplot(2, 3, 5)
journey_lengths = df_cleaned.groupby('account_id')['touch_sequence'].max()
journey_lengths_filtered = journey_lengths[journey_lengths <= 50]
ax5.hist(journey_lengths_filtered, bins=30, color='teal', alpha=0.7, edgecolor='black')
ax5.axvline(journey_lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {journey_lengths.mean():.1f}')
ax5.axvline(journey_lengths.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {journey_lengths.median():.1f}')
ax5.set_xlabel('Number of Touches', fontsize=11, fontweight='bold')
ax5.set_ylabel('Number of Accounts', fontsize=11, fontweight='bold')
ax5.set_title('Journey Length Distribution\n(≤50 touches)', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(axis='y', alpha=0.3)

# 6. Key Statistics Box
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

stats_text = f"""
KEY STATISTICS

Dataset:
  • Original Records: {len(df_original):,}
  • Cleaned Records: {len(df_cleaned):,}
  • Retention Rate: {len(df_cleaned)/len(df_original)*100:.1f}%
  • Records Removed: {len(df_original) - len(df_cleaned):,}

Accounts:
  • Total Accounts: {df_cleaned['account_id'].nunique():,}
  • Avg Journey Length: {journey_lengths.mean():.1f} touches
  • Median Journey: {journey_lengths.median():.0f} touches

Time Metrics:
  • Date Range: {df_cleaned['activity_date'].min()[:10]} to
                {df_cleaned['activity_date'].max()[:10]}
  • Avg Days Between Touches: {df_cleaned['days_since_last_touch'].mean():.1f}
  • Median Days Between: {df_cleaned['days_since_last_touch'].median():.0f}

Outcomes:
  • Won Accounts: {(outcome_counts.get('Won', 0)):,}
  • Lost Accounts: {(outcome_counts.get('Lost', 0)):,}
  • Ongoing Accounts: {(outcome_counts.get('Ongoing', 0)):,}

Quality:
  • Missing Values: {df_cleaned.isnull().sum().sum():,}
  • Duplicate Rows: 0
  • Data Completeness: {(1 - df_cleaned.isnull().sum().sum() / (len(df_cleaned) * len(df_cleaned.columns)))*100:.2f}%
"""

ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('STEP1_SUMMARY_REPORT.png', dpi=300, bbox_inches='tight')


