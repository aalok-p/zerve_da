import pandas as pd
import numpy as np

df = pd.read_csv('zerve_data.csv', low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

user_first_week = []

for user_id, user_data in df.groupby('person_id'):
    user_data = user_data.sort_values('timestamp')
    first_event = user_data.iloc[0]['timestamp']

    first_week_data = user_data[user_data['timestamp'] <= first_event + timedelta(days=7)]
    
    if len(first_week_data) == 0:
        continue
    
    #first week metrics
    first_week_metrics = {
        'person_id': user_id,
        'first_event_date': first_event,
        'first_week_events': len(first_week_data),
        'first_week_days_active': first_week_data['date'].nunique(),
        'first_week_sessions': first_week_data['prop_$session_id'].nunique(),
        'signed_up_in_week': (first_week_data['event'] == 'sign_up').any(),
        'created_blocks_in_week': (first_week_data['event'] == 'block_create').any(),
        'ran_blocks_in_week': (first_week_data['event'] == 'run_block').any(),
        'used_agent_in_week': (first_week_data['event'].str.contains('agent', na=False)).any(),
        'completed_onboarding_in_week': (first_week_data['event'] == 'submit_onboarding_form').any(),
        
        # Check if they came back after week 1
        'returned_after_week_1': len(user_data[user_data['timestamp'] > first_event + timedelta(days=7)]) > 0,
        'total_lifetime_events': len(user_data),
        'lifetime_days': (user_data['timestamp'].max() - first_event).days + 1
    }
    
    user_first_week.append(first_week_metrics)

first_week_df = pd.DataFrame(user_first_week)
retention_rate = first_week_df['returned_after_week_1'].mean() * 100

retained_users = first_week_df[first_week_df['returned_after_week_1']]
churned_users = first_week_df[~first_week_df['returned_after_week_1']]


metrics_to_plot = [
    ('first_week_events', 'Events in Week 1'),
    ('first_week_days_active', 'Days Active'),
    ('first_week_sessions', 'Sessions'),
    ('created_blocks_in_week', 'Created Blocks (%)'),
    ('ran_blocks_in_week', 'Ran Blocks (%)'),
    ('used_agent_in_week', 'Used Agent (%)')
]