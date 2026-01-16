import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

with open('decision_tree_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

df_cleaned = pd.read_csv('data_cleaned.csv')

def predict_top4_actions(account_state, model, encoders):
    feature_order = [
        'unique_actions_so_far',
        'current_action_encoded',
        'action_index',
        'country_encoded',
        'solution_encoded',
        'opportunity_stage_encoded',
        'is_lead',
        'touch_sequence',
        'days_since_last',
        'repeat_ratio',
        'month',
        'day_of_week',
        'prev_action_encoded',
        'journey_outcome_encoded'
    ]
    
    # Get current action
    current_action = account_state.get('current_action', 'Email')
    
    # Encode current_action
    if 'current_action' in encoders and current_action in encoders['current_action'].classes_:
        current_action_encoded = encoders['current_action'].transform([current_action])[0]
    else:
        current_action_encoded = 0
    
    # Get action_index (categorical code for current action)
    action_index = current_action_encoded  # Use encoded value as index
    
    # Encode other categorical features
    country = account_state.get('country', 'US')
    if 'country' in encoders and country in encoders['country'].classes_:
        country_encoded = encoders['country'].transform([country])[0]
    else:
        country_encoded = 0
    
    solution = account_state.get('solution', 'MRS')
    if 'solution' in encoders and solution in encoders['solution'].classes_:
        solution_encoded = encoders['solution'].transform([solution])[0]
    else:
        solution_encoded = 0
    
    opportunity_stage = account_state.get('opportunity_stage', 'Prospecting')
    if 'opportunity_stage' in encoders and opportunity_stage in encoders['opportunity_stage'].classes_:
        opportunity_stage_encoded = encoders['opportunity_stage'].transform([opportunity_stage])[0]
    else:
        opportunity_stage_encoded = 0
    
    prev_action = account_state.get('prev_action', 'None')
    if 'prev_action' in encoders and prev_action in encoders['prev_action'].classes_:
        prev_action_encoded = encoders['prev_action'].transform([prev_action])[0]
    else:
        prev_action_encoded = 0
    
    journey_outcome = account_state.get('journey_outcome', 'Ongoing')
    if 'journey_outcome' in encoders and journey_outcome in encoders['journey_outcome'].classes_:
        journey_outcome_encoded = encoders['journey_outcome'].transform([journey_outcome])[0]
    else:
        journey_outcome_encoded = 0
    
    # Build feature vector with all 14 features
    features = [
        account_state.get('unique_actions_so_far', 1),  # Default to 1 for first action
        current_action_encoded,
        action_index,
        country_encoded,
        solution_encoded,
        opportunity_stage_encoded,
        account_state.get('is_lead', 1),
        account_state.get('touch_sequence', 1),
        account_state.get('days_since_last', 0),
        account_state.get('repeat_ratio', 0.0),  # Default to 0.0 for first action
        account_state.get('month', datetime.now().month),
        account_state.get('day_of_week', datetime.now().weekday()),
        prev_action_encoded,
        journey_outcome_encoded
    ]
    
    # Create feature vector
    X = np.array([features])
    
    # Get predictions
    probabilities = model.predict_proba(X)[0]
    action_names = encoders['next_action'].classes_
    
    # Create results DataFrame
    results = pd.DataFrame({
        'action': action_names,
        'base_weight': probabilities
    })
    
    # Get top 4
    results = results.sort_values('base_weight', ascending=False).head(4).reset_index(drop=True)
    results['rank'] = range(1, len(results) + 1)
    
    return results


def apply_weight_adjustment(base_weights_df, last_action, adjustment_factor=0.3):
    df = base_weights_df.copy()
    
    # Calculate last touch weight (penalty for action just taken)
    df['last_touch_weight'] = df['action'].apply(
        lambda x: adjustment_factor if x == last_action else 0.0
    )
    
    # Apply formula
    df['adjusted_weight'] = df['base_weight'] * (1 - df['last_touch_weight'])
    
    # Renormalize to sum to 1.0
    total = df['adjusted_weight'].sum()
    if total > 0:
        df['adjusted_weight'] = df['adjusted_weight'] / total
    
    # Calculate percentage
    df['percentage'] = (df['adjusted_weight'] * 100).round(1)
    
    # Re-rank by adjusted weight
    df = df.sort_values('adjusted_weight', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def generate_recommendations(account_info, action_history=None):
    # Build current state
    current_state = account_info.copy()
    
    # Update state based on action history
    if action_history and len(action_history) > 0:
        current_state['current_action'] = action_history[-1]['action']
        current_state['touch_sequence'] = len(action_history)
        
        # Calculate unique_actions_so_far
        unique_actions = set(a['action'] for a in action_history)
        current_state['unique_actions_so_far'] = len(unique_actions)
        
        # Calculate repeat_ratio (rolling average of repeats)
        if len(action_history) >= 2:
            # Check if current action is repeat of previous
            recent_actions = [a['action'] for a in action_history[-5:]]  # Last 5 actions
            repeats = sum(1 for i in range(1, len(recent_actions)) if recent_actions[i] == recent_actions[i-1])
            current_state['repeat_ratio'] = repeats / max(1, len(recent_actions) - 1)
        else:
            current_state['repeat_ratio'] = 0.0
        
        if len(action_history) > 1:
            # Calculate days since last
            last_date = action_history[-1].get('date', datetime.now())
            prev_date = action_history[-2].get('date', datetime.now() - timedelta(days=7))
            current_state['days_since_last'] = (last_date - prev_date).days
            current_state['prev_action'] = action_history[-2]['action']
        else:
            current_state['days_since_last'] = 0
            current_state['prev_action'] = 'None'
    else:
        # First action defaults
        current_state['unique_actions_so_far'] = 1
        current_state['repeat_ratio'] = 0.0
        current_state['touch_sequence'] = 1
        current_state['days_since_last'] = 0
        current_state['prev_action'] = 'None'
    
    # 1. Recommendations by Country
    top4_country = predict_top4_actions(current_state, dt_model, label_encoders)
    
    # Apply weight adjustment if there's history
    if action_history and len(action_history) > 0:
        last_action = action_history[-1]['action']
        top4_country = apply_weight_adjustment(top4_country, last_action)
    else:
        top4_country['adjusted_weight'] = top4_country['base_weight']
        top4_country['percentage'] = (top4_country['adjusted_weight'] * 100).round(1)
        top4_country['last_touch_weight'] = 0.0
    
    # 2. Recommendations by Solution
    top4_solution = predict_top4_actions(current_state, dt_model, label_encoders)
    
    if action_history and len(action_history) > 0:
        last_action = action_history[-1]['action']
        top4_solution = apply_weight_adjustment(top4_solution, last_action)
    else:
        top4_solution['adjusted_weight'] = top4_solution['base_weight']
        top4_solution['percentage'] = (top4_solution['adjusted_weight'] * 100).round(1)
        top4_solution['last_touch_weight'] = 0.0
    
    # 3. Recommendations by Country × Solution  
    top4_combined = predict_top4_actions(current_state, dt_model, label_encoders)
    
    if action_history and len(action_history) > 0:
        last_action = action_history[-1]['action']
        top4_combined = apply_weight_adjustment(top4_combined, last_action)
    else:
        top4_combined['adjusted_weight'] = top4_combined['base_weight']
        top4_combined['percentage'] = (top4_combined['adjusted_weight'] * 100).round(1)
        top4_combined['last_touch_weight'] = 0.0
    
    return {
        'by_country': top4_country,
        'by_solution': top4_solution,
        'by_country_solution': top4_combined
    }


def display_recommendations(recommendations, account_info, action_history=None):
    """Display formatted recommendations"""
    print("="*80)
    print("RECOMMENDATION RESULTS")
    print("="*80)
    print()
    
    # Account info
    print("ACCOUNT INFORMATION:")
    print(f"  Country: {account_info.get('country', 'N/A')}")
    print(f"  Solution: {account_info.get('solution', 'N/A')}")
    print(f"  Stage: {account_info.get('opportunity_stage', 'N/A')}")
    
    if action_history and len(action_history) > 0:
        print(f"  Journey Progress: {len(action_history)} touches")
        print(f"  Last Action: {action_history[-1]['action']}")
    else:
        print(f"  Journey Progress: Not started")
    
    print()
    print("-"*80)
    
    # 1. By Country
    print()
    print(f"TOP 4 ACTIONS BY COUNTRY ({account_info.get('country', 'N/A')})")
    print()
    df = recommendations['by_country'][['rank', 'action', 'percentage']]
    for _, row in df.iterrows():
        print(f"  {int(row['rank'])}. {row['action']:<25} {row['percentage']:>6.1f}%")
    
    print()
    print("-"*80)
    
    # 2. By Solution
    print()
    print(f"TOP 4 ACTIONS BY SOLUTION ({account_info.get('solution', 'N/A')})")
    print()
    df = recommendations['by_solution'][['rank', 'action', 'percentage']]
    for _, row in df.iterrows():
        print(f"  {int(row['rank'])}. {row['action']:<25} {row['percentage']:>6.1f}%")
    
    print()
    print("-"*80)
    
    # 3. By Country × Solution
    print()
    print(f"TOP 4 ACTIONS BY COUNTRY x SOLUTION ({account_info.get('country', 'N/A')} x {account_info.get('solution', 'N/A')})")
    print()
    df = recommendations['by_country_solution'][['rank', 'action', 'percentage']]
    for _, row in df.iterrows():
        print(f"  {int(row['rank'])}. {row['action']:<25} {row['percentage']:>6.1f}%")
    


def save_recommendations(recommendations, account_info, action_history, filename):
    all_data = []
    
    for rec_type, df in recommendations.items():
        df_copy = df.copy()
        df_copy['recommendation_type'] = rec_type
        df_copy['country'] = account_info.get('country')
        df_copy['solution'] = account_info.get('solution')
        df_copy['opportunity_stage'] = account_info.get('opportunity_stage')
        df_copy['touch_count'] = len(action_history) if action_history else 0
        df_copy['last_action'] = action_history[-1]['action'] if action_history and len(action_history) > 0 else 'None'
        all_data.append(df_copy)
    
    result = pd.concat(all_data, ignore_index=True)
    result.to_csv(filename, index=False)
    print(f"[OK] Saved to: {filename}")


def run_demonstration():
    scenarios = [
        {
            'name': 'Scenario 1: US MRS Customer - New Journey',
            'account': {
                'country': 'US',
                'solution': 'MRS',
                'opportunity_stage': 'Prospecting',
                'is_lead': 1,
                'month': 12,
                'day_of_week': 1
            },
            'actions': []
        },
        {
            'name': 'Scenario 2: UK Digital Customer - After Email',
            'account': {
                'country': 'UK',
                'solution': 'Digital',
                'opportunity_stage': 'Qualification',
                'is_lead': 2,
                'month': 12,
                'day_of_week': 2
            },
            'actions': [
                {'action': 'Email', 'date': datetime.now() - timedelta(days=3)}
            ]
        },
        {
            'name': 'Scenario 3: Germany MRS - Mid-Journey',
            'account': {
                'country': 'DE',
                'solution': 'MRS',
                'opportunity_stage': 'Proposal',
                'is_lead': 2,
                'month': 12,
                'day_of_week': 3
            },
            'actions': [
                {'action': 'Email', 'date': datetime.now() - timedelta(days=30)},
                {'action': 'Inbound Call', 'date': datetime.now() - timedelta(days=23)},
                {'action': '1st Appointment', 'date': datetime.now() - timedelta(days=14)},
                {'action': 'Meeting', 'date': datetime.now() - timedelta(days=7)}
            ]
        }
    ]
    
    all_results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print()
        print("="*80)
        print(scenario['name'])
        print("="*80)
        print()
        
        # Generate recommendations
        recs = generate_recommendations(scenario['account'], scenario['actions'])
        
        # Display
        display_recommendations(recs, scenario['account'], scenario['actions'])
        
        # Save
        filename = f"step4_scenario_{i}_recommendations.csv"
        save_recommendations(recs, scenario['account'], scenario['actions'], filename)
        print()
        
        # Collect for summary
        for rec_type, df in recs.items():
            df_copy = df.copy()
            df_copy['scenario'] = scenario['name']
            df_copy['scenario_number'] = i
            df_copy['recommendation_type'] = rec_type
            df_copy['country'] = scenario['account']['country']
            df_copy['solution'] = scenario['account']['solution']
            df_copy['touch_count'] = len(scenario['actions'])
            all_results.append(df_copy)
    
    # Save summary
    summary = pd.concat(all_results, ignore_index=True)
    summary.to_csv('step4_all_scenarios_summary.csv', index=False)
    
    print()
    print("="*80)
    print("[OK] All scenarios complete!")
    print("[OK] Summary saved to: step4_all_scenarios_summary.csv")
    print("="*80)
    print()
    
    return summary

if __name__ == "__main__":
    # Run demonstration
    summary = run_demonstration()