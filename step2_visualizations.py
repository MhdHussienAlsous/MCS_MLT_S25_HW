import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 15)

# Load data
journeys = pd.read_csv('step2_journey_paths_all.csv')
top_paths = pd.read_csv('step2_top_paths_by_segment.csv')
won_paths = pd.read_csv('step2_top_paths_won_outcomes.csv')
lost_paths = pd.read_csv('step2_top_paths_lost_outcomes.csv')


# Create comprehensive visualization
fig = plt.figure(figsize=(22, 14))
fig.suptitle('Step 2: Journey Path Analysis - Key Findings', fontsize=22, fontweight='bold', y=0.98)

# 1. Journey length by outcome
ax1 = plt.subplot(3, 4, 1)
outcome_order = ['Won', 'Lost', 'Ongoing']
journey_by_outcome = []
for outcome in outcome_order:
    lengths = journeys[journeys['outcome'] == outcome]['num_touches']
    journey_by_outcome.append(lengths)

bp = ax1.boxplot(journey_by_outcome, labels=outcome_order, patch_artist=True)
colors = ['green', 'red', 'orange']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax1.set_ylabel('Number of Touches', fontsize=11, fontweight='bold')
ax1.set_title('Journey Length by Outcome', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)

# 2. Top paths by frequency (overall)
ax2 = plt.subplot(3, 4, 2)
overall_top = top_paths.nlargest(8, 'frequency')
paths_short = [p[:30] + '...' if len(p) > 30 else p for p in overall_top['path']]
ax2.barh(range(len(paths_short)), overall_top['frequency'], color='steelblue')
ax2.set_yticks(range(len(paths_short)))
ax2.set_yticklabels(paths_short, fontsize=8)
ax2.set_xlabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Top 8 Most Common Paths', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. Win rate by path type (top successful paths)
ax3 = plt.subplot(3, 4, 3)
successful = top_paths[(top_paths['win_rate'] > 50) & (top_paths['frequency'] > 5)].nlargest(8, 'win_rate')
paths_short2 = [p[:30] + '...' if len(p) > 30 else p for p in successful['path']]
colors_winrate = ['darkgreen' if x > 75 else 'green' if x > 60 else 'yellowgreen' for x in successful['win_rate']]
ax3.barh(range(len(paths_short2)), successful['win_rate'], color=colors_winrate)
ax3.set_yticks(range(len(paths_short2)))
ax3.set_yticklabels(paths_short2, fontsize=8)
ax3.set_xlabel('Win Rate (%)', fontsize=11, fontweight='bold')
ax3.set_title('Highest Win Rate Paths (>5 occurrences)', fontsize=13, fontweight='bold')
ax3.set_xlim(0, 100)
ax3.grid(axis='x', alpha=0.3)

# 4. Top countries - journey counts
ax4 = plt.subplot(3, 4, 4)
top_countries = journeys['country'].value_counts().head(8)
ax4.bar(range(len(top_countries)), top_countries.values, color='purple', alpha=0.7)
ax4.set_xticks(range(len(top_countries)))
ax4.set_xticklabels(top_countries.index, rotation=45, ha='right')
ax4.set_ylabel('Number of Journeys', fontsize=11, fontweight='bold')
ax4.set_title('Top 8 Countries by Journey Count', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. Path diversity by segment
ax5 = plt.subplot(3, 4, 5)
segment_diversity = top_paths.groupby('segment_type')['path'].nunique()
colors_seg = ['coral', 'skyblue', 'lightgreen']
bars = ax5.bar(range(len(segment_diversity)), segment_diversity.values, color=colors_seg)
ax5.set_xticks(range(len(segment_diversity)))
ax5.set_xticklabels(segment_diversity.index, rotation=15)
ax5.set_ylabel('Unique Paths in Top 5', fontsize=11, fontweight='bold')
ax5.set_title('Path Diversity by Segment Type', fontsize=13, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, segment_diversity.values)):
    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             str(val), ha='center', fontsize=10, fontweight='bold')

# 6. Win rate comparison: Won vs Lost paths
ax6 = plt.subplot(3, 4, 6)
comparison_data = {
    'Won Paths\nAvg Length': journeys[journeys['outcome'] == 'Won']['num_touches'].mean(),
    'Lost Paths\nAvg Length': journeys[journeys['outcome'] == 'Lost']['num_touches'].mean(),
    'Ongoing Paths\nAvg Length': journeys[journeys['outcome'] == 'Ongoing']['num_touches'].mean()
}
bars = ax6.bar(range(len(comparison_data)), comparison_data.values(), 
               color=['green', 'red', 'orange'], alpha=0.7)
ax6.set_xticks(range(len(comparison_data)))
ax6.set_xticklabels(comparison_data.keys(), fontsize=10)
ax6.set_ylabel('Average Touches', fontsize=11, fontweight='bold')
ax6.set_title('Average Journey Length by Outcome', fontsize=13, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, comparison_data.values())):
    ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')

# 7. Solution distribution
ax7 = plt.subplot(3, 4, 7)
solution_counts = journeys['solution'].value_counts()
colors_sol = ['#3498db', '#e74c3c', '#2ecc71']
wedges, texts, autotexts = ax7.pie(solution_counts, labels=solution_counts.index,
                                     autopct='%1.1f%%', colors=colors_sol, startangle=90)
for text in texts:
    text.set_fontsize(10)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')
    autotext.set_color('white')
ax7.set_title('Journey Distribution by Solution', fontsize=13, fontweight='bold')

# 8. Top 5 Won paths
ax8 = plt.subplot(3, 4, 8)
top_won = won_paths.head(5)
paths_won_short = [p[:25] + '...' if len(p) > 25 else p for p in top_won['path']]
ax8.barh(range(len(paths_won_short)), top_won['frequency'], color='darkgreen', alpha=0.7)
ax8.set_yticks(range(len(paths_won_short)))
ax8.set_yticklabels(paths_won_short, fontsize=8)
ax8.set_xlabel('Frequency', fontsize=11, fontweight='bold')
ax8.set_title('Top 5 Paths for WON Outcomes', fontsize=13, fontweight='bold')
ax8.grid(axis='x', alpha=0.3)

# 9. Top 5 Lost paths
ax9 = plt.subplot(3, 4, 9)
top_lost = lost_paths.head(5)
paths_lost_short = [p[:25] + '...' if len(p) > 25 else p for p in top_lost['path']]
ax9.barh(range(len(paths_lost_short)), top_lost['frequency'], color='darkred', alpha=0.7)
ax9.set_yticks(range(len(paths_lost_short)))
ax9.set_yticklabels(paths_lost_short, fontsize=8)
ax9.set_xlabel('Frequency', fontsize=11, fontweight='bold')
ax9.set_title('Top 5 Paths for LOST Outcomes', fontsize=13, fontweight='bold')
ax9.grid(axis='x', alpha=0.3)

# 10. Win rate distribution
ax10 = plt.subplot(3, 4, 10)
win_rates = top_paths[top_paths['frequency'] > 3]['win_rate']
ax10.hist(win_rates, bins=20, color='teal', alpha=0.7, edgecolor='black')
ax10.axvline(win_rates.mean(), color='red', linestyle='--', linewidth=2, 
             label=f'Mean: {win_rates.mean():.1f}%')
ax10.set_xlabel('Win Rate (%)', fontsize=11, fontweight='bold')
ax10.set_ylabel('Number of Paths', fontsize=11, fontweight='bold')
ax10.set_title('Win Rate Distribution (freq>3)', fontsize=13, fontweight='bold')
ax10.legend()
ax10.grid(axis='y', alpha=0.3)

# 11. Path frequency by segment type
ax11 = plt.subplot(3, 4, 11)
segment_freq = top_paths.groupby('segment_type')['frequency'].mean()
bars = ax11.bar(range(len(segment_freq)), segment_freq.values, 
                color=['coral', 'skyblue', 'lightgreen'])
ax11.set_xticks(range(len(segment_freq)))
ax11.set_xticklabels(segment_freq.index, rotation=15)
ax11.set_ylabel('Average Frequency', fontsize=11, fontweight='bold')
ax11.set_title('Avg Path Frequency by Segment', fontsize=13, fontweight='bold')
ax11.grid(axis='y', alpha=0.3)

# 12. Key Statistics Box
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')

stats_text = f"""
KEY STATISTICS - STEP 2

Journey Analysis:
  • Total Journeys: {len(journeys):,}
  • Unique Paths: {journeys['simple_path'].nunique():,}
  • Countries Analyzed: {journeys['country'].nunique()}
  • Solutions: {journeys['solution'].nunique()}

Outcomes:
  • Won: {len(journeys[journeys['outcome']=='Won']):,} ({len(journeys[journeys['outcome']=='Won'])/len(journeys)*100:.1f}%)
  • Lost: {len(journeys[journeys['outcome']=='Lost']):,} ({len(journeys[journeys['outcome']=='Lost'])/len(journeys)*100:.1f}%)
  • Ongoing: {len(journeys[journeys['outcome']=='Ongoing']):,} ({len(journeys[journeys['outcome']=='Ongoing'])/len(journeys)*100:.1f}%)

Journey Lengths:
  • Won Avg: {journeys[journeys['outcome']=='Won']['num_touches'].mean():.1f} touches
  • Lost Avg: {journeys[journeys['outcome']=='Lost']['num_touches'].mean():.1f} touches
  • Ongoing Avg: {journeys[journeys['outcome']=='Ongoing']['num_touches'].mean():.1f} touches

Top Insights:
  • Longest journey: {journeys['num_touches'].max()} touches
  • Shortest journey: {journeys['num_touches'].min()} touch
  • Most common path occurs: {top_paths['frequency'].max()} times
  • Highest win rate: {top_paths[top_paths['frequency']>3]['win_rate'].max():.1f}%

Files Created: 8
"""

ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('STEP2_JOURNEY_PATHS_VISUALIZATIONS.png', dpi=300, bbox_inches='tight')
