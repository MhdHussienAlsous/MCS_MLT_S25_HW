import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load cleaned data
df = pd.read_csv('data_cleaned.csv')
df['activity_date'] = pd.to_datetime(df['activity_date'])


# Create visualizations
fig = plt.figure(figsize=(20, 15))

# 1. Action types distribution
ax1 = plt.subplot(3, 3, 1)
df['types'].value_counts().head(10).plot(kind='barh', color='steelblue')
plt.title('Top 10 Action Types', fontsize=14, fontweight='bold')
plt.xlabel('Count')
plt.tight_layout()

# 2. Journey outcomes
ax2 = plt.subplot(3, 3, 2)
outcome_counts = df.groupby('account_id')['journey_outcome'].first().value_counts()
colors = ['green', 'orange', 'red']
plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Journey Outcomes', fontsize=14, fontweight='bold')

# 3. Solutions distribution
ax3 = plt.subplot(3, 3, 3)
df['solution'].value_counts().plot(kind='bar', color='coral')
plt.title('Solution Types', fontsize=14, fontweight='bold')
plt.xlabel('Solution')
plt.ylabel('Count')
plt.xticks(rotation=45)

# 4. Top countries
ax4 = plt.subplot(3, 3, 4)
df['Country'].value_counts().head(10).plot(kind='barh', color='purple')
plt.title('Top 10 Countries', fontsize=14, fontweight='bold')
plt.xlabel('Count')

# 5. Touch sequence distribution
ax5 = plt.subplot(3, 3, 5)
touch_per_account = df.groupby('account_id')['touch_sequence'].max()
touch_per_account[touch_per_account <= 50].hist(bins=30, color='teal', edgecolor='black')
plt.title('Touch Sequence per Account (≤50 touches)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Touches')
plt.ylabel('Number of Accounts')

# 6. Opportunity stages
ax6 = plt.subplot(3, 3, 6)
df['opportunity_stage'].value_counts().head(10).plot(kind='barh', color='salmon')
plt.title('Top 10 Opportunity Stages', fontsize=14, fontweight='bold')
plt.xlabel('Count')

# 7. Activities over time
ax7 = plt.subplot(3, 3, 7)
df.groupby(df['activity_date'].dt.to_period('M')).size().plot(color='navy', linewidth=2)
plt.title('Activities Over Time (Monthly)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Number of Activities')
plt.xticks(rotation=45)

# 8. Days between touches
ax8 = plt.subplot(3, 3, 8)
days_data = df['days_since_last_touch'].dropna()
days_data[days_data <= 365].hist(bins=50, color='olive', edgecolor='black')
plt.title('Days Between Touches (≤365 days)', fontsize=14, fontweight='bold')
plt.xlabel('Days')
plt.ylabel('Frequency')

# 9. Journey outcome by solution
ax9 = plt.subplot(3, 3, 9)
outcome_solution = pd.crosstab(df.groupby('account_id')['solution'].first(), 
                                df.groupby('account_id')['journey_outcome'].first())
outcome_solution.plot(kind='bar', stacked=False, color=['green', 'red', 'orange'])
plt.title('Journey Outcome by Solution', fontsize=14, fontweight='bold')
plt.xlabel('Solution')
plt.ylabel('Count')
plt.legend(title='Outcome')
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('step1_data_exploration_visualizations.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved to: data_exploration_visualizations.png")


print("\n1. Action Types by Outcome:")
action_outcome = pd.crosstab(df['types'], df['journey_outcome'])

print("\n2. Average Journey Length by Outcome:")
journey_stats = df.groupby(['account_id', 'journey_outcome'])['touch_sequence'].max().reset_index()
journey_stats_summary = journey_stats.groupby('journey_outcome')['touch_sequence'].agg(['mean', 'median', 'std'])

print("\n3. Country Statistics:")
country_stats = df.groupby('Country').agg({
    'account_id': 'nunique',
    'types': 'count'
}).rename(columns={'account_id': 'unique_accounts', 'types': 'total_activities'})
country_stats = country_stats.sort_values('unique_accounts', ascending=False).head(10)

print("\n4. Solution Statistics:")
solution_stats = df.groupby('solution').agg({
    'account_id': 'nunique',
    'types': 'count'
}).rename(columns={'account_id': 'unique_accounts', 'types': 'total_activities'})

print("\n5. Action Type Success Rate (Won vs Total):")
action_success = df.groupby('types')['journey_outcome'].apply(
    lambda x: (x == 'Won').sum() / len(x) * 100
).sort_values(ascending=False)

# Save statistics to file
with open('step1_data_exploration_statistics.txt', 'w', encoding='utf-8') as f:
    f.write("DETAILED DATA STATISTICS\n")
    f.write("="*60 + "\n\n")
    
    f.write("1. Action Types by Outcome:\n")
    f.write(action_outcome.to_string())
    f.write("\n\n")
    
    f.write("2. Average Journey Length by Outcome:\n")
    f.write(journey_stats_summary.to_string())
    f.write("\n\n")
    
    f.write("3. Top 10 Countries:\n")
    f.write(country_stats.to_string())
    f.write("\n\n")
    
    f.write("4. Solution Statistics:\n")
    f.write(solution_stats.to_string())
    f.write("\n\n")
    
    f.write("5. Action Type Success Rate (Won %):\n")
    f.write(action_success.to_string())

