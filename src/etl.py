import utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    try:
        events = pd.read_csv(filepath + 'events.csv', parse_dates=['timestamp'])
        events = events.sort_values('timestamp')
    except IOError:
        events = None
    #Columns in mortality_event.csv - patient_id,timestamp,label
    try:
        mortality = pd.read_csv(filepath + 'mortality_events.csv', parse_dates=['timestamp'])
        mortality = mortality.sort_values('timestamp')
    except IOError:
        mortality = None
    #Columns in event_feature_map.csv - idx,event_id
    try:
        feature_map = pd.read_csv(filepath + 'event_feature_map.csv')
    except IOError:
        feature_map = None

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''

    patient_ids = events.patient_id.unique()
    dead_ids = mortality.patient_id
    alive_ids = pd.Series(list(set(patient_ids).difference(set(dead_ids))))
    dead_events = events[events.patient_id.isin(dead_ids)] 
    alive_events = events[events.patient_id.isin(alive_ids)]
    def findIndx_dead(df):
        return mortality[mortality.patient_id == df.iloc[0, :].patient_id].iloc[0, :].timestamp - pd.Timedelta(days=30)
    
    def findIndx_alive(df):
        return df.iloc[-1, :].timestamp

    indx_dead = pd.DataFrame(dead_events.groupby('patient_id').apply(findIndx_dead))
    indx_alive = pd.DataFrame(alive_events.groupby('patient_id').apply(findIndx_alive))
    indx_date = indx_alive.append(indx_dead).sort_index()
    indx_date = indx_date.reset_index()
    indx_date.columns = ['patient_id', 'indx_date']
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', index=False)
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''

    indx_date = indx_date.set_index('patient_id')
    events_requried = events.join(indx_date, on='patient_id', how='outer', rsuffix='_right')
    cond_diff = (events_requried.indx_date - events_requried.timestamp)
    cond = np.logical_and(cond_diff >= pd.Timedelta(days=0),cond_diff <= pd.Timedelta(days=2000))
    filtered_events = events_requried[cond]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    join_df = filtered_events_df.join(feature_map_df.set_index('event_id'), on='event_id')
    filtered_events_df['feature_id'] = join_df['idx']
    filtered_events_df = filtered_events_df.dropna()
    def agg_events_sum_count(df):
        enc_label = df.event_id.iloc[0]

        if 'LAB' in enc_label:
            return df.patient_id.count()

        elif 'DIAG' in enc_label or 'DRUG' in enc_label:
            return df.value.sum() 

    aggregate_cols = ['patient_id', 'event_id', 'feature_id']
    aggregated_events = filtered_events_df.groupby(aggregate_cols)
    lab_score = filtered_events_df[filtered_events_df.event_id.str.contains('LAB')]
    lab_score = lab_score.groupby(aggregate_cols).patient_id.count()
    other_score = filtered_events_df[np.logical_or(filtered_events_df.event_id.str.contains('DIAG'),
                                    filtered_events_df.event_id.str.contains('DRUG'),)]
    other_score = other_score.groupby(aggregate_cols).value.sum()
    aggregated_events = pd.concat((other_score, lab_score)).reset_index()
    aggregated_events.columns = aggregate_cols + ['feature_value']
    aggregated_events = aggregated_events[['patient_id', 'feature_id', 'feature_value']]
    norm_pivoted = aggregated_events.pivot(index='patient_id', columns='feature_id', values='feature_value')
    norm = norm_pivoted/norm_pivoted.max()
    norm = norm.reset_index()
    aggregated_events = pd.melt(norm, id_vars='patient_id', value_name='feature_value').dropna()
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    feature_tuple = aggregated_events.groupby('patient_id').apply(lambda x : list(x.sort_values('feature_id').apply(lambda y:
                                                            (y.feature_id, y.feature_value),
                                                            axis=1)))
    patient_features = feature_tuple.to_dict()
    all_patient_id = aggregated_events.patient_id.unique()
    dead_patient_id = list(mortality.patient_id)
    mortaility_label = [(id, int(id in dead_patient_id)) for id in list(all_patient_id)]
    mortality = dict(mortaility_label)

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    
    for patient, features in patient_features.items():
        features = pd.DataFrame(features).sort_values(0)
        features = features.values.tolist()
        deliverable1.write(bytes("{} {} \n".format(mortality.get(patient, 0), utils.bag_to_svmlight(features)),'UTF-8'))
        deliverable2.write(bytes("{} {} {} \n".format(int(patient), mortality.get(patient, 0), utils.bag_to_svmlight(features)),'UTF-8'))

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()