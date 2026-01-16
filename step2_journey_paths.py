import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# Load cleaned data
print("\nLoading cleaned data...")
df = pd.read_csv('data_cleaned.csv')
df['activity_date'] = pd.to_datetime(df['activity_date'])


# Function to create journey path
def create_journey_path(actions, max_length=10):
    if len(actions) > max_length:
        actions = list(actions[:max_length//2]) + ['...'] + list(actions[-(max_length//2):])
    return ' → '.join(actions)

# Function to create simple path (first 5 actions)
def create_simple_path(actions, length=5):
    return ' → '.join(list(actions[:length]))

# Extract journey paths for each account
journey_data = []

for account_id, group in df.groupby('account_id'):
    # Sort by date
    group = group.sort_values('activity_date')
    
    # Extract information
    actions = group['types'].tolist()
    country = group['Country'].iloc[0]
    solution = group['solution'].iloc[0]
    outcome = group['journey_outcome'].iloc[0]
    num_touches = len(actions)
    
    # Convert actions to strings
    actions_str = [str(a) if pd.notna(a) else 'Unknown' for a in actions]
    
    # Create different path representations
    full_path = create_journey_path(actions_str, max_length=20)
    simple_path = create_simple_path(actions_str, length=5)
    first_3_actions = ' → '.join(actions_str[:3]) if len(actions_str) >= 3 else ' → '.join(actions_str)
    
    journey_data.append({
        'account_id': account_id,
        'country': country,
        'solution': solution,
        'outcome': outcome,
        'num_touches': num_touches,
        'full_path': full_path,
        'simple_path': simple_path,
        'first_3_actions': first_3_actions,
        'actions_list': ','.join(actions_str)
    })

journeys_df = pd.DataFrame(journey_data)
# Save all journey paths
journeys_df.to_csv('step2_journey_paths_all.csv', index=False)

# Function to find top paths
def find_top_paths(df_subset, path_column='simple_path', top_n=5):
    path_stats = []
    
    for path, group in df_subset.groupby(path_column):
        total = len(group)
        won = (group['outcome'] == 'Won').sum()
        lost = (group['outcome'] == 'Lost').sum()
        ongoing = (group['outcome'] == 'Ongoing').sum()
        
        win_rate = (won / total * 100) if total > 0 else 0
        avg_touches = group['num_touches'].mean()
        
        path_stats.append({
            'path': path,
            'frequency': total,
            'won': won,
            'lost': lost,
            'ongoing': ongoing,
            'win_rate': win_rate,
            'avg_touches': avg_touches
        })
    
    # Sort by frequency
    if not path_stats:
        return pd.DataFrame()
    path_stats_df = pd.DataFrame(path_stats).sort_values('frequency', ascending=False)
    return path_stats_df.head(top_n)

# Analyze by Country
country_results = []

for country in journeys_df['country'].value_counts().head(10).index:
    country_data = journeys_df[journeys_df['country'] == country]
    top_paths = find_top_paths(country_data)
    
    for _, row in top_paths.iterrows():
        country_results.append({
            'segment_type': 'Country',
            'segment_value': country,
            'rank': len([r for r in country_results if r['segment_value'] == country]) + 1,
            'path': row['path'],
            'frequency': row['frequency'],
            'won': row['won'],
            'lost': row['lost'],
            'ongoing': row['ongoing'],
            'win_rate': row['win_rate'],
            'avg_touches': row['avg_touches']
        })

# Analyze by Solution
solution_results = []

for solution in journeys_df['solution'].unique():
    solution_data = journeys_df[journeys_df['solution'] == solution]
    top_paths = find_top_paths(solution_data)
    
    if len(top_paths) == 0:
        continue
    
    for _, row in top_paths.iterrows():
        solution_results.append({
            'segment_type': 'Solution',
            'segment_value': solution,
            'rank': len([r for r in solution_results if r['segment_value'] == solution]) + 1,
            'path': row['path'],
            'frequency': row['frequency'],
            'won': row['won'],
            'lost': row['lost'],
            'ongoing': row['ongoing'],
            'win_rate': row['win_rate'],
            'avg_touches': row['avg_touches']
        })

# Analyze by Country + Solution
combo_results = []

for (country, solution), combo_data in journeys_df.groupby(['country', 'solution']):
    if len(combo_data) < 5:  # Skip very small segments
        continue
    
    top_paths = find_top_paths(combo_data)
    
    for _, row in top_paths.iterrows():
        combo_results.append({
            'segment_type': 'Country_x_Solution',
            'segment_value': f"{country}_x_{solution}",
            'country': country,
            'solution': solution,
            'rank': len([r for r in combo_results if r['segment_value'] == f"{country}_x_{solution}"]) + 1,
            'path': row['path'],
            'frequency': row['frequency'],
            'won': row['won'],
            'lost': row['lost'],
            'ongoing': row['ongoing'],
            'win_rate': row['win_rate'],
            'avg_touches': row['avg_touches']
        })

# Save results
country_df = pd.DataFrame(country_results)
solution_df = pd.DataFrame(solution_results)
combo_df = pd.DataFrame(combo_results)

# Combine all results
all_results = pd.concat([country_df, solution_df, combo_df], ignore_index=True)
all_results.to_csv('step2_top_paths_by_segment.csv', index=False)

# Save separate files for each segment type
country_df.to_csv('step2_top_paths_by_country.csv', index=False)
solution_df.to_csv('step2_top_paths_by_solution.csv', index=False)
combo_df.to_csv('step2_top_paths_by_country_solution.csv', index=False)

# Find paths with high win rates
successful_paths = []

for segment_type in ['Country', 'Solution', 'Country_x_Solution']:
    segment_results = all_results[all_results['segment_type'] == segment_type]
    
    # Filter for high win rate and reasonable frequency
    high_win = segment_results[
        (segment_results['win_rate'] > 50) & 
        (segment_results['frequency'] > 3)
    ].sort_values('win_rate', ascending=False)
    
    if len(high_win) > 0:
        successful_paths.append(high_win)

if successful_paths:
    successful_df = pd.concat(successful_paths, ignore_index=True)
    successful_df.to_csv('step2_successful_paths_high_winrate.csv', index=False)

# Analyze first actions by outcome
first_actions = df[df['touch_sequence'] == 1].groupby(['types', 'journey_outcome']).size().reset_index(name='count')
first_actions_pivot = first_actions.pivot(index='types', columns='journey_outcome', values='count').fillna(0)
first_actions_pivot['total'] = first_actions_pivot.sum(axis=1)
first_actions_pivot['win_rate'] = (first_actions_pivot.get('Won', 0) / first_actions_pivot['total'] * 100).round(2)
first_actions_pivot = first_actions_pivot.sort_values('total', ascending=False)

# Analyze action sequences that lead to Win vs Lost
won_journeys = journeys_df[journeys_df['outcome'] == 'Won']
lost_journeys = journeys_df[journeys_df['outcome'] == 'Lost']

# Most common paths for Won outcomes
won_paths = find_top_paths(won_journeys, top_n=10)
won_paths.to_csv('step2_top_paths_won_outcomes.csv', index=False)

# Most common paths for Lost outcomes
lost_paths = find_top_paths(lost_journeys, top_n=10)
lost_paths.to_csv('step2_top_paths_lost_outcomes.csv', index=False)

summary = {
    'total_journeys': len(journeys_df),
    'unique_paths': journeys_df['simple_path'].nunique(),
    'countries_analyzed': journeys_df['country'].nunique(),
    'solutions_analyzed': journeys_df['solution'].nunique(),
    'country_solution_combinations': len(combo_df['segment_value'].unique()),
    'avg_journey_length': journeys_df['num_touches'].mean(),
    'won_journeys': len(won_journeys),
    'lost_journeys': len(lost_journeys),
    'ongoing_journeys': len(journeys_df[journeys_df['outcome'] == 'Ongoing'])
}
