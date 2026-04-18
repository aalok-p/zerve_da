import pandas as pd

users_df = pd.read_csv('users_seg.csv')
feature_importance = pd.read_csv('feature_resultcsv')
early_indicators = pd.read_csv('early_results.csv')
first_week = pd.read_csv('first_week_analysis.csv')

report =[]
report.append(f"Total Users Analyzed: {len(users_df):,}")
report.append(f"Average Success Score: {users_df['success_score'].mean():.1f} / 100")
report.append(f"Success Score Std Dev: {users_df['success_score'].std():.1f}")

success_dist = users_df['success_category'].value_counts().to_dict()
report.append("Success Distribution:")
for category in ['Low', 'Medium', 'High', 'Very High']:
    count = success_dist.get(category, 0)
    pct = count / len(users_df) * 100
    report.append(f"  • {category}: {count:,} users ({pct:.1f}%)")
report.append("")

agent_users = users_df[users_df['uses_agent'] == True]
non_agent_users = users_df[users_df['uses_agent'] == False]
agent_lift = agent_users['success_score'].mean() - non_agent_users['success_score'].mean()
retained = first_week[first_week['returned_after_week_1'] == True]
churned = first_week[first_week['returned_after_week_1'] == False]
retention_rate = len(retained) / len(first_week) * 100

avg_execution_rate = users_df['block_execution_rate'].mean()
high_exec = users_df[users_df['block_execution_rate'] > avg_execution_rate]
low_exec = users_df[users_df['block_execution_rate'] <= avg_execution_rate]

segment_summary = users_df.groupby('segment').agg({
    'person_id': 'count',
    'success_score': 'mean',
    'days_active': 'mean',
    'total_events': 'mean'
}).round(1)
segment_summary.columns = ['Count', 'Avg Success', 'Avg Days', 'Avg Events']

for segment in segment_summary.index:
    data = segment_summary.loc[segment]
    report.append(f"{segment}")
    report.append(f"  • Users: {int(data['Count']):,} ({data['Count']/len(users_df)*100:.1f}%)")
    report.append(f"  • Success Score: {data['Avg Success']:.1f}")
    report.append(f"  • Avg Days Active: {data['Avg Days']:.1f}")
    report.append(f"  • Avg Events: {data['Avg Events']:.1f}")