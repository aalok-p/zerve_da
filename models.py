import pandas as pd
import numpy as np

users_df = pd.read_csv('users_features.csv')
numeric_cols = users_df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['person_id', 'success_score', 'engagement_score', 'technical_score', 'commitment_score']
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

correlations = users_df[feature_cols + ['success_score']].corr()['success_score'].drop('success_score')
correlations_sorted = correlations.abs().sort_values(ascending=False)

print(correlations_sorted.head(15))

