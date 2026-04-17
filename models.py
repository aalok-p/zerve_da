import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score


users_df = pd.read_csv('users_features.csv')
numeric_cols = users_df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['person_id', 'success_score', 'engagement_score', 'technical_score', 'commitment_score']
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

correlations = users_df[feature_cols + ['success_score']].corr()['success_score'].drop('success_score')
correlations_sorted = correlations.abs().sort_values(ascending=False)

print(correlations_sorted.head(15))

top_15_features = correlations_sorted.head(15).index
correlations[top_15_features].sort_values().plot(kind='barh', color='steelblue')

#data modeling
X = users_df[feature_cols].fillna(0)
y = users_df['success_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train 
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"model R² Score: {r2:.3f}")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)


#segment users on behaviour pattern
def segment_user(row):
    if row['addon_credits_count'] > 0:
        return 'Power User (Paid)'
    elif row['uses_agent'] and row['days_active'] >= 5:
        return 'Active Agent User'
    elif row['days_active'] >= 5 and row['total_events'] > 50:
        return 'Engaged User'
    elif row['days_active'] <= 1:
        return 'One-Time Visitor'
    else:
        return 'Casual User'

users_df['segment'] = users_df.apply(segment_user, axis=1)

segment_stats = users_df.groupby('segment').agg({
    'person_id': 'count',
    'success_score': 'mean',
    'days_active': 'mean',
    'total_events': 'mean',
    'uses_agent': 'sum'
}).round(2)
segment_stats.columns = ['Count', 'Avg Success Score', 'Avg Days Active', 'Avg Events', 'Agent Users']

print(segment_stats)

high_performers = users_df[users_df['success_score'] >= 60]
low_performers = users_df[users_df['success_score'] < 30]

comparison_metrics = [
    'days_active', 'unique_sessions', 'block_create_count', 
    'run_block_count', 'agent_new_chat_count', 'credits_used_events'
]

comparison_data = pd.DataFrame({
    'High Performers': high_performers[comparison_metrics].mean(),
    'Low Performers': low_performers[comparison_metrics].mean()
})
early_indicators = [
    'completed_onboarding', 'skipped_onboarding', 'sign_up_count',
    'block_create_count', 'run_block_count', 'uses_agent',
    'unique_event_types', 'days_active', 'events_per_day'
]

X_class = users_df[early_indicators].fillna(0)

for col in X_class.select_dtypes(include=['bool']).columns:
    X_class[col] = X_class[col].astype(int)
    
y_class = users_df['high_performer']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf_classifier.fit(X_train_c, y_train_c)

y_pred_c = rf_classifier.predict(X_test_c)

#imp features
early_importance = pd.DataFrame({
    'feature': early_indicators,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)