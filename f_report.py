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

