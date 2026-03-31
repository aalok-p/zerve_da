import pandas as pd
import numpy as np

df = pd.read_csv('erve_data.csv', low_memory=False)
print(f"shape: {df.shape}")
print(f"col: {df.shape[1]}")

print(f"unique users (person_id): {df['person_id'].nunique():,}")
print(f"unique events: {df['event'].nunique():,}")
print(f"date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

#timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

#events
event_counts =df['event'].value_counts().head(20)
print(event_counts)

