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

#group by user
user_features =[]
for user_id, user_data in df.groupby('person_id'):
    features= {
        'person_id': len(user_data),
        'uique_sessions':   
         user_data['prop_$session_id'].nunique(),
        'days_active': user_data['date'].nunique(),
        'first_seen': user_data['timestamp'].min(),
        'last_seen': user_data['timestamp'].max(),
         
        # specific event counts
        'unique_event_types': user_data['event'].nunique(),

        #specific event counts
        'sign_in_count': (user_data['event'] == 'sign_in').sum(),
        'sign_up_count': (user_data['event'] == 'sign_up').sum(),
        'block_create_count': (user_data['event'] == 'block_create').sum(),
        'block_delete_count': (user_data['event'] == 'block_delete').sum(),
        'run_block_count': (user_data['event'] == 'run_block').sum(),
        'agent_new_chat_count': (user_data['event'] == 'agent_new_chat').sum(),
        'agent_message_count': (user_data['event'] == 'agent_message').sum(),

        #agent tool
        'agent_create_block_count': (user_data['event'] == 'agent_tool_call_create_block_tool').sum(),
        'agent_run_block_count': (user_data['event'] =='agent_tool_call_run_block_tool').sum(),
        'agent_get_block_count': (user_data['event'] == 'agent_tool_call_get_block_tool').sum(),
        'agent_refactor_block_count': (user_data['event'] == 'agent_tool_call_refactor_block_tool').sum(),
        
        #credits
        'credits_used_events': (user_data['event'] =='credits_used').sum(),
        'credits_exceeded_count': (user_data['event'] == 'credits_exceeded').sum(),
        'addon_credits_count': (user_data['event'] == 'addon_credits_used').sum(),
    }
    user_features.append(features)

users_df = pd.DataFrame(user_features)
