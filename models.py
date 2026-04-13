import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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