import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv', parse_dates=['timestamp'])
    events = events.sort_values('timestamp')
    mortality = pd.read_csv(filepath + 'mortality_events.csv', parse_dates=['timestamp'])
    mortality = mortality.sort_values('timestamp')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    patient_ids = events.patient_id.unique()
    dead_ids = mortality.patient_id
    alive_ids = pd.Series(list(set(patient_ids).difference(set(dead_ids))))
    dead_events = events[events.patient_id.isin(dead_ids)] 
    alive_events = events[events.patient_id.isin(alive_ids)]
    dead_group = dead_events.groupby('patient_id')
    alive_group = alive_events.groupby('patient_id')
    dead_count = dead_group.event_id.count()
    alive_count = alive_group.event_id.count()
    avg_dead_event_count = dead_count.mean()
    max_dead_event_count = dead_count.max()
    min_dead_event_count = dead_count.min()
    avg_alive_event_count = alive_count.mean()
    max_alive_event_count = alive_count.max()
    min_alive_event_count = alive_count.min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    encounter_label = ['DIAG', 'DRUG', 'LAB']
    patient_ids = events.patient_id.unique()
    dead_ids = mortality.patient_id
    alive_ids = pd.Series(list(set(patient_ids).difference(set(dead_ids))))
    dead_events = events[events.patient_id.isin(dead_ids)] 
    alive_events = events[events.patient_id.isin(alive_ids)]
    encounter_dead = dead_events[pd.Series(np.any([dead_events.event_id.str.contains(x)
                                                        for x in encounter_label]),
                                                        index=dead_events.index)]
    encounter_alive = alive_events[pd.Series(np.any([alive_events.event_id.str.contains(x)
                                                        for x in encounter_label]),
                                                        index=alive_events.index)]
    dead_group = encounter_dead.groupby('patient_id')
    alive_group = encounter_alive.groupby('patient_id')
    dead_counts = dead_group.apply(lambda x: x.timestamp.unique().size)
    alive_counts = alive_group.apply(lambda x: x.timestamp.unique().size)
    avg_dead_encounter_count = dead_counts.mean()
    max_dead_encounter_count = dead_counts.max()
    min_dead_encounter_count = dead_counts.min()
    avg_alive_encounter_count = alive_counts.mean()
    max_alive_encounter_count = alive_counts.max()
    min_alive_encounter_count = alive_counts.min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    patient_ids = events.patient_id.unique()
    dead_ids = mortality.patient_id
    alive_ids = pd.Series(list(set(patient_ids).difference(set(dead_ids))))
    dead_events = events[events.patient_id.isin(dead_ids)] 
    alive_events = events[events.patient_id.isin(alive_ids)]
    dead_group = dead_events.groupby('patient_id')
    alive_group = alive_events.groupby('patient_id')
    dead_len = dead_group.apply(lambda x: (x.timestamp.iloc[-1] -
                                 x.timestamp.iloc[0]).days)
    alive_len = alive_group.apply(lambda x: (x.timestamp.iloc[-1] -
                                 x.timestamp.iloc[0]).days)
    avg_dead_rec_len = dead_len.mean()
    max_dead_rec_len = dead_len.max()
    min_dead_rec_len = dead_len.min()
    avg_alive_rec_len = alive_len.mean()
    max_alive_rec_len = alive_len.max()
    min_alive_rec_len = alive_len.min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following path variable in coding but switch it back when submission.
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()
