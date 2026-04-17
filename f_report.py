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
