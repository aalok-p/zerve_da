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

#dervied feat
users_df['lifetime_days'] =(users_df['last_seen']-users_df['first_seen']).dt.total_seconds()/86400
users_df['events_per_day'] =users_df['total_events'] / (users_df['lifetime_days']+1)
users_df['sessions_per_day']= users_df['unique_sessions'] /(users_df['days_active']+1)
users_df['events_per_session']= users_df['total_events'] /(users_df['unique_sessions']+1)

users_df['uses_agent'] = (users_df['agent_new_chat_count']>0)|(users_df['agent_message_count'] > 0)
users_df['block_execution_rate'] = users_df['run_block_count']/(users_df['block_create_count'] + 1)

print(f"\n created features for {len(users_df):,} user")
print(f" total feature: {len(users_df.columns)}")


users_df['engagement_score'] = ((users_df['days_active'].clip(0, 30) / 30) * 40 + (users_df['unique_sessions'].clip(0, 20)/20)*30+ (users_df['events_per_day'].clip(0,50)/50)* 0) * 100 / 100
users_df['technical_score']=(((users_df['block_create_count'] > 0).astype(int)) * 20 + ((users_df['run_block_count'] > 0).astype(int)) * 20 + ((users_df['uses_agent']).astype(int)) * 30 +((users_df['agent_refactor_block_count'] > 0).astype(int)) * 15 +((users_df['unique_event_types'] > 10).astype(int)) * 15)
users_df['commitment_score'] = ((users_df['credits_used_events'].clip(0, 50) / 50) * 50 +((users_df['addon_credits_count'] > 0).astype(int))*50)
users_df['success_score'] = (users_df['engagement_score'] * 0.4 +users_df['technical_score'] * 0.35 +users_df['commitment_score'] * 0.25)
users_df['success_category'] = pd.cut(users_df['success_score'],bins=[0, 25, 50, 75, 100],labels=['Low', 'Medium', 'High', 'Very High'])

print(users_df['success_category'].value_counts().sort_index())
print(users_df['success_score'].describe())