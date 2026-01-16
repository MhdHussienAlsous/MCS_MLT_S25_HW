import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

FEATURE_ORDER = [
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

df = pd.read_csv('data_cleaned.csv')
df['activity_date'] = pd.to_datetime(df['activity_date'])

features_list = []

for account_id, group in df.groupby('account_id'):
    group = group.sort_values('activity_date').reset_index(drop=True)
    
    for i in range(len(group) - 1):  # Exclude last touch (no next action)
        current_touch = group.iloc[i]
        next_touch = group.iloc[i + 1]
        
        # Current state features
        features = {
            # Current action
            'current_action': str(current_touch['types']),
            
            # Context features
            'country': str(current_touch['Country']),
            'solution': str(current_touch['solution']),
            'opportunity_stage': str(current_touch['opportunity_stage']),
            'is_lead': int(current_touch['is_lead']),
            
            # Journey progress features
            'touch_sequence': int(current_touch['touch_sequence']),
            'days_since_last': float(current_touch['days_since_last_touch']) if pd.notna(current_touch['days_since_last_touch']) else 0,
            
            # Temporal features
            'month': int(current_touch['month']),
            'day_of_week': int(current_touch['day_of_week']),
            
            # Previous action (if exists)
            'prev_action': str(group.iloc[i-1]['types']) if i > 0 else 'None',
            
            # Journey outcome (known at end)
            'journey_outcome': str(current_touch['journey_outcome']),
            
            # Target: Next action
            'next_action': str(next_touch['types'])
        }
        
        features_list.append(features)

features_df = pd.DataFrame(features_list)

# 1) Unique actions so far (string-safe)
def cumulative_unique_count(series):
    seen = set()
    output = []
    for item in series:
        seen.add(item)
        output.append(len(seen))
    return output

features_df["unique_actions_so_far"] = (
    features_df.groupby("country")["current_action"]
    .transform(cumulative_unique_count)
)

# 2) Repeat ratio (string-safe)
features_df["is_repeat"] = (
    features_df.groupby("country")["current_action"]
    .transform(lambda x: (x.shift() == x))
).astype(int)

features_df["repeat_ratio"] = (
    features_df.groupby("country")["is_repeat"]
    .transform(lambda x: x.rolling(5).mean())
).fillna(0)

# 3) Action index (categorical code)
features_df["action_index"] = features_df["current_action"].astype("category").cat.codes

# Save feature dataset
features_df.to_csv('step3_ml_features_dataset.csv', index=False)

label_encoders = {}
categorical_cols = ['current_action', 'country', 'solution', 'opportunity_stage', 
                    'prev_action', 'journey_outcome']

for col in categorical_cols:
    le = LabelEncoder()
    features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].astype(str))
    label_encoders[col] = le


# Encode target
target_encoder = LabelEncoder()
features_df['next_action_encoded'] = target_encoder.fit_transform(features_df['next_action'])
label_encoders['next_action'] = target_encoder

# Save encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Prepare features and target
feature_columns = [
    'current_action_encoded',
    'country_encoded',
    'solution_encoded',
    'opportunity_stage_encoded',
    'is_lead',
    'touch_sequence',
    'days_since_last',
    'month',
    'day_of_week',
    'prev_action_encoded',
    'journey_outcome_encoded',

    # NEW FEATURES
    'unique_actions_so_far',
    'repeat_ratio',
    'action_index'
]

X = features_df[feature_columns]
y = features_df['next_action_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model with optimized parameters
dt_model = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)
y_pred_proba = dt_model.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Top-k accuracy (if next action is in top 4 predictions)
def top_k_accuracy(y_true, y_pred_proba, k=4):
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    correct = sum([y_true.iloc[i] in top_k_preds[i] for i in range(len(y_true))])
    return correct / len(y_true)

top4_acc = top_k_accuracy(y_test, y_pred_proba, k=4)

# Classification report
target_names = target_encoder.classes_
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)

# Show top 10 most common actions
top_actions = features_df['next_action'].value_counts().head(10).index
for action in top_actions:
    if action in report:
        metrics = report[action]

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

# Save feature importance
feature_importance.to_csv('step3_feature_importance.csv', index=False)

def predict_top4_actions(model, encoder, current_state, top_n=4):
    """
    Predict top N next actions given current state
    
    Parameters:
    - model: trained model
    - encoder: label encoder for actions
    - current_state: dict with current features
    - top_n: number of recommendations
    
    Returns:
    - List of (action, probability) tuples
    """
    # Prepare features --- ensure we use the SAME order the model was trained with
    # `feature_columns` (defined earlier) is the training order. Use it and
    # provide safe defaults if a key is missing from `current_state`.
    cols = feature_columns
    row = [current_state.get(col, 0) for col in cols]
    features = pd.DataFrame([row], columns=cols)

    # Get probabilities
    probabilities = model.predict_proba(features)[0]
    
    # Get top N
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    
    recommendations = []
    for idx in top_indices:
        action = encoder.inverse_transform([idx])[0]
        prob = probabilities[idx]
        recommendations.append((action, prob))
    
    return recommendations


# Test with a sample
sample_idx = 100
sample = features_df.iloc[sample_idx]

current_state_encoded = {
    'unique_actions_so_far': sample['unique_actions_so_far'],
    'current_action_encoded': sample['current_action_encoded'],
    'action_index': sample['action_index'],
    'country_encoded': sample['country_encoded'],
    'solution_encoded': sample['solution_encoded'],
    'opportunity_stage_encoded': sample['opportunity_stage_encoded'],
    'is_lead': sample['is_lead'],
    'touch_sequence': sample['touch_sequence'],
    'days_since_last': sample['days_since_last'],
    'repeat_ratio': sample['repeat_ratio'],
    'month': sample['month'],
    'day_of_week': sample['day_of_week'],
    'prev_action_encoded': sample['prev_action_encoded'],
    'journey_outcome_encoded': sample['journey_outcome_encoded']
}

recommendations = predict_top4_actions(dt_model, target_encoder, current_state_encoded)


# Save model
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

# Function to get recommendations by segment
def get_recommendations_by_segment(df, model, encoders, segment_type, segment_value, touch_num=5):
    """Get top 4 actions for a specific segment"""
    
    # Filter data
    if segment_type == 'country':
        subset = df[df['country'] == segment_value]
    elif segment_type == 'solution':
        subset = df[df['solution'] == segment_value]
    elif segment_type == 'country_solution':
        country, solution = segment_value.split('_')
        subset = df[(df['country'] == country) & (df['solution'] == solution)]
    
    if len(subset) == 0:
        return []
    
    # Get typical state for this segment at touch N
    typical_state = subset[subset['touch_sequence'] == touch_num].iloc[0] if len(subset[subset['touch_sequence'] == touch_num]) > 0 else subset.iloc[0]
    
    state_encoded = {
        'unique_actions_so_far': typical_state['unique_actions_so_far'],
        'current_action_encoded': typical_state['current_action_encoded'],
        'action_index': typical_state['action_index'],
        'country_encoded': typical_state['country_encoded'],
        'solution_encoded': typical_state['solution_encoded'],
        'opportunity_stage_encoded': typical_state['opportunity_stage_encoded'],
        'is_lead': typical_state['is_lead'],
        'touch_sequence': typical_state['touch_sequence'],
        'days_since_last': typical_state['days_since_last'],
        'repeat_ratio': typical_state['repeat_ratio'],
        'month': typical_state['month'],
        'day_of_week': typical_state['day_of_week'],
        'prev_action_encoded': typical_state['prev_action_encoded'],
        'journey_outcome_encoded': typical_state['journey_outcome_encoded']
    }
    
    return predict_top4_actions(model, encoders['next_action'], state_encoded)

# Generate recommendations for top segments
recommendations_by_segment = []

print("\n📍 Top 4 Actions by COUNTRY:")
for country in ['US', 'UK', 'DE']:
    recs = get_recommendations_by_segment(features_df, dt_model, label_encoders, 'country', country)
    print(f"\n{country}:")
    for i, (action, prob) in enumerate(recs, 1):
        print(f"  {i}. {action} ({prob*100:.1f}%)")
    
    for rank, (action, prob) in enumerate(recs, 1):
        recommendations_by_segment.append({
            'segment_type': 'Country',
            'segment_value': country,
            'rank': rank,
            'action': action,
            'confidence': prob * 100
        })

print("\n🎯 Top 4 Actions by SOLUTION:")
for solution in ['MRS', 'Digital']:
    recs = get_recommendations_by_segment(features_df, dt_model, label_encoders, 'solution', solution)
    print(f"\n{solution}:")
    for i, (action, prob) in enumerate(recs, 1):
        print(f"  {i}. {action} ({prob*100:.1f}%)")
    
    for rank, (action, prob) in enumerate(recs, 1):
        recommendations_by_segment.append({
            'segment_type': 'Solution',
            'segment_value': solution,
            'rank': rank,
            'action': action,
            'confidence': prob * 100
        })

print("\n🌍 Top 4 Actions by COUNTRY × SOLUTION:")
for combo in ['US_MRS', 'US_Digital', 'UK_MRS']:
    recs = get_recommendations_by_segment(features_df, dt_model, label_encoders, 'country_solution', combo)
    print(f"\n{combo.replace('_', ' × ')}:")
    for i, (action, prob) in enumerate(recs, 1):
        print(f"  {i}. {action} ({prob*100:.1f}%)")
    
    for rank, (action, prob) in enumerate(recs, 1):
        recommendations_by_segment.append({
            'segment_type': 'Country_x_Solution',
            'segment_value': combo,
            'rank': rank,
            'action': action,
            'confidence': prob * 100
        })

# Save recommendations
recommendations_df = pd.DataFrame(recommendations_by_segment)
recommendations_df.to_csv('step3_ml_recommendations_by_segment.csv', index=False)


# Create visualizations
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Step 3: Decision Tree Model - Results', fontsize=20, fontweight='bold', y=0.98)

# 1. Feature Importance
ax1 = plt.subplot(2, 3, 1)
top_features = feature_importance.head(10)
ax1.barh(range(len(top_features)), top_features['importance'], color='steelblue')
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['feature'], fontsize=9)
ax1.set_xlabel('Importance', fontsize=11, fontweight='bold')
ax1.set_title('Top 10 Feature Importance', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# 2. Model Accuracy
ax2 = plt.subplot(2, 3, 2)
accuracies = ['Overall\nAccuracy', 'Top-4\nAccuracy']
values = [accuracy * 100, top4_acc * 100]
colors = ['coral' if v < 50 else 'yellowgreen' if v < 70 else 'green' for v in values]
bars = ax2.bar(accuracies, values, color=colors, alpha=0.7)
ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Model Performance', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

# 3. Action Distribution
ax3 = plt.subplot(2, 3, 3)
top_next_actions = features_df['next_action'].value_counts().head(8)
ax3.barh(range(len(top_next_actions)), top_next_actions.values, color='purple', alpha=0.7)
ax3.set_yticks(range(len(top_next_actions)))
ax3.set_yticklabels(top_next_actions.index, fontsize=9)
ax3.set_xlabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Top 8 Next Actions in Data', fontsize=13, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. Confusion Matrix (top actions only)
ax4 = plt.subplot(2, 3, 4)
top_action_indices = [list(target_encoder.classes_).index(action) for action in top_next_actions.head(5).index]
mask = np.isin(y_test, top_action_indices) & np.isin(y_pred, top_action_indices)
cm = confusion_matrix(y_test[mask], y_pred[mask], labels=top_action_indices)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=[target_encoder.classes_[i] for i in top_action_indices],
            yticklabels=[target_encoder.classes_[i] for i in top_action_indices])
ax4.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax4.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax4.set_title('Confusion Matrix (Top 5 Actions)', fontsize=13, fontweight='bold')
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax4.get_yticklabels(), rotation=0, fontsize=8)

# 5. Touch sequence vs accuracy
ax5 = plt.subplot(2, 3, 5)
touch_acc = []
for touch in sorted(features_df['touch_sequence'].unique())[:20]:
    mask = features_df.loc[X_test.index, 'touch_sequence'] == touch
    if mask.sum() > 10:
        acc = accuracy_score(y_test[mask], y_pred[mask])
        touch_acc.append((touch, acc * 100))

if touch_acc:
    touches, accs = zip(*touch_acc)
    ax5.plot(touches, accs, marker='o', linewidth=2, markersize=6, color='teal')
    ax5.set_xlabel('Touch Sequence', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Prediction Accuracy by Touch Number', fontsize=13, fontweight='bold')
    ax5.grid(alpha=0.3)

# 6. Stats box
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

stats_text = f"""
MODEL STATISTICS

Dataset:
  • Training samples: {len(X_train):,}
  • Test samples: {len(X_test):,}
  • Features: {X.shape[1]}
  • Target classes: {len(target_encoder.classes_)}

Performance:
  • Overall Accuracy: {accuracy*100:.2f}%
  • Top-4 Accuracy: {top4_acc*100:.2f}%

Feature Importance (Top 5):
  1. {feature_importance.iloc[0]['feature']}: {feature_importance.iloc[0]['importance']:.3f}
  2. {feature_importance.iloc[1]['feature']}: {feature_importance.iloc[1]['importance']:.3f}
  3. {feature_importance.iloc[2]['feature']}: {feature_importance.iloc[2]['importance']:.3f}
  4. {feature_importance.iloc[3]['feature']}: {feature_importance.iloc[3]['importance']:.3f}
  5. {feature_importance.iloc[4]['feature']}: {feature_importance.iloc[4]['importance']:.3f}

Model: Decision Tree
  • Max depth: 15
  • Min samples split: 50
  • Min samples leaf: 20
  • Class weight: balanced
"""

ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('STEP3_DECISION_TREE_RESULTS.png', dpi=300, bbox_inches='tight')

