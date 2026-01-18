import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


with open('decision_tree_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load cleaned data for win probability calculation
df_cleaned = pd.read_csv('data_cleaned.csv')

# Set random seed for reproducibility
np.random.seed(42)

def predict_next_actions(account_state, model, encoders):
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
    action_index = current_action_encoded
    
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
    
    journey_outcome = 'Ongoing'
    if 'journey_outcome' in encoders and journey_outcome in encoders['journey_outcome'].classes_:
        journey_outcome_encoded = encoders['journey_outcome'].transform([journey_outcome])[0]
    else:
        journey_outcome_encoded = 0
    
    # Build feature vector with all 14 features
    features = [
        account_state.get('unique_actions_so_far', 1),
        current_action_encoded,
        action_index,
        country_encoded,
        solution_encoded,
        opportunity_stage_encoded,
        account_state.get('is_lead', 1),
        account_state.get('touch_sequence', 1),
        account_state.get('days_since_last', 0),
        account_state.get('repeat_ratio', 0.0),
        account_state.get('month', datetime.now().month),
        account_state.get('day_of_week', datetime.now().weekday()),
        prev_action_encoded,
        journey_outcome_encoded
    ]
    
    # Create feature vector and predict
    X = np.array([features])
    probabilities = model.predict_proba(X)[0]
    action_names = encoders['next_action'].classes_
    
    # Create results
    results = pd.DataFrame({
        'action': action_names,
        'probability': probabilities
    }).sort_values('probability', ascending=False).reset_index(drop=True)
    
    return results


def select_next_action(predictions, action_history):
    if len(predictions) == 0:
        return None, 0.0
    
    # Avoid last 3 actions
    if len(action_history) >= 3:
        recent = set(action_history[-3:])
        available = predictions[~predictions['action'].isin(recent)]
        
        if len(available) > 0:
            # Pick from top 3 available actions
            top_available = available.head(5)
            
            if len(top_available) > 1:
                probs = top_available['probability'].values
                # Handle edge cases
                if np.any(np.isnan(probs)) or np.sum(probs) == 0:
                    return top_available.iloc[0]['action'], top_available.iloc[0]['probability']
                probs = probs / probs.sum()  # Normalize
                chosen_idx = np.random.choice(len(top_available), p=probs)
                chosen = top_available.iloc[chosen_idx]
                return chosen['action'], chosen['probability']
            else:
                return available.iloc[0]['action'], available.iloc[0]['probability']
    
    # Fallback: avoid just last action
    if len(action_history) > 0:
        last_action = action_history[-1]
        available = predictions[predictions['action'] != last_action]
        if len(available) > 0:
            return available.iloc[0]['action'], available.iloc[0]['probability']
    
    return predictions.iloc[0]['action'], predictions.iloc[0]['probability']


def estimate_win_probability(journey_state, action_history, historical_data):
    country = journey_state.get('country')
    solution = journey_state.get('solution')
    touch_count = len(action_history)
    
    # Base probability from historical data
    similar = historical_data[
        (historical_data['Country'] == country) &
        (historical_data['solution'] == solution)
    ]
    
    if len(similar) > 0:
        won = similar[similar['journey_outcome'] == 'Won']
        lost = similar[similar['journey_outcome'] == 'Lost']
        total = len(won) + len(lost)
        base_prob = len(won) / total if total > 0 else 0.5
    else:
        base_prob = 0.5
    
    # Journey length bonus (1.2% per touch, max 20%)
    length_bonus = min(0.20, touch_count * 0.012)
    
    # Diversity bonus (2.5% per unique action, max 15%)
    unique_count = len(set(action_history))
    diversity_bonus = min(0.15, unique_count * 0.025)
    
    # Key action presence bonus
    key_actions = {
        'On-Site': 0.08,
        'Demo': 0.07,
        'Proposal': 0.08,
        'Trial': 0.06,
        'Meeting': 0.04,
        '2nd Appointment': 0.03
    }
    
    key_bonus = sum([key_actions.get(action, 0) for action in set(action_history)])
    key_bonus = min(0.15, key_bonus)
    
    # Calculate final probability
    final_prob = base_prob + length_bonus + diversity_bonus + key_bonus
    final_prob = min(0.95, max(0.15, final_prob))
    
    return final_prob


def progress_stage(current_stage, touch_count):
    stages = {
        'Prospecting': ('Qualification', 3),
        'Qualification': ('Discovery', 5),
        'Discovery': ('Proposal', 8),
        'Proposal': ('Negotiation', 11),
        'Negotiation': ('Closed Won', 14)
    }
    
    if current_stage in stages:
        next_stage, threshold = stages[current_stage]
        if touch_count >= threshold:
            return next_stage
    return current_stage


def generate_optimal_journey(account_info, max_steps=10):
    journey = []
    current_state = account_info.copy()
    current_state['touch_sequence'] = 0
    current_state['prev_action'] = 'None'
    current_state['days_since_last'] = 0
    current_state['unique_actions_so_far'] = 1
    current_state['repeat_ratio'] = 0.0
    
    action_history = []
    
    for step in range(max_steps):
        current_state['touch_sequence'] = step + 1
        
        # Progress opportunity stage
        current_state['opportunity_stage'] = progress_stage(
            current_state.get('opportunity_stage', 'Prospecting'),
            step + 1
        )
        
        # Get model predictions
        predictions = predict_next_actions(current_state, dt_model, label_encoders)
        
        # Select next action with diversity
        next_action, action_prob = select_next_action(predictions, action_history)
        
        if next_action is None:
            break
        
        # Add to history
        action_history.append(next_action)
        
        # Calculate unique_actions_so_far
        unique_actions = set(action_history)
        current_state['unique_actions_so_far'] = len(unique_actions)
        
        # Calculate repeat_ratio (rolling average of repeats)
        if len(action_history) >= 2:
            recent_actions = action_history[-5:]  # Last 5 actions
            repeats = sum(1 for i in range(1, len(recent_actions)) if recent_actions[i] == recent_actions[i-1])
            current_state['repeat_ratio'] = repeats / max(1, len(recent_actions) - 1)
        else:
            current_state['repeat_ratio'] = 0.0
        
        # Calculate win probability
        win_prob = estimate_win_probability(current_state, action_history, df_cleaned)
        
        # Record step
        journey_step = {
            'step': step + 1,
            'action': next_action,
            'confidence': action_prob,
            'win_probability': win_prob,
            'stage': current_state['opportunity_stage']
        }
        journey.append(journey_step)
        
        # Update state for next iteration
        current_state['prev_action'] = current_state.get('current_action', 'None')
        current_state['current_action'] = next_action
        current_state['days_since_last'] = 7
        
        # Stop if win probability very high
        if win_prob >= 0.90:
            break
    
    return journey, action_history

def run_demonstration():
    scenarios = {
        'Scenario 1: US MRS Customer': {
            'country': 'US',
            'solution': 'MRS',
            'opportunity_stage': 'Prospecting',
            'is_lead': 1,
            'month': 12,
            'day_of_week': 1
        },
        'Scenario 2: UK Digital Customer': {
            'country': 'UK',
            'solution': 'Digital',
            'opportunity_stage': 'Qualification',
            'is_lead': 1,
            'month': 12,
            'day_of_week': 2
        },
        'Scenario 3: Germany MRS Customer': {
            'country': 'DE',
            'solution': 'MRS',
            'opportunity_stage': 'Discovery',
            'is_lead': 1,
            'month': 12,
            'day_of_week': 3
        }
    }
    
    all_results = []
    
    for scenario_name, account_info in scenarios.items():
        # Generate optimal journey
        journey, action_history = generate_optimal_journey(account_info, max_steps=10)

        # Collect results
        all_results.append({
            'scenario': scenario_name,
            'journey_length': len(journey),
            'unique_actions': len(set(action_history)),
            'diversity_score': len(set(action_history)) / len(action_history) if journey else 0,
            'action_sequence': ' → '.join(action_history),
            'final_win_probability': journey[-1]['win_probability'] if journey else 0,
            'avg_confidence': np.mean([s['confidence'] for s in journey]) if journey else 0
        })
    
    # Create summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv('step5_optimal_journeys_summary.csv', index=False)
    
    # Save detailed journeys
    for i, (scenario_name, account_info) in enumerate(scenarios.items(), 1):
        journey, action_history = generate_optimal_journey(account_info, max_steps=10)
        
        journey_data = []
        for step in journey:
            journey_data.append({
                'scenario': scenario_name,
                'step': step['step'],
                'action': step['action'],
                'opportunity_stage': step['stage'],
                'action_confidence': step['confidence'],
                'win_probability': step['win_probability']
            })
        
        journey_df = pd.DataFrame(journey_data)
        journey_df.to_csv(f'step5_scenario_{i}_journey_details.csv', index=False)
        print(f"[OK] Saved: step5_scenario_{i}_journey_details.csv")
    
    return summary_df

if __name__ == "__main__":
    summary = run_demonstration()