import os
import pandas as pd
import numpy as np
import holidays



def clean_data(dataset, is_train: bool):
    """
    Does the dataset cleaning that it was specified in the cleaning.ipynb

    Args:
    dataset: Which dataset wants to be cleaned 

    Returns:
    Cleaned dataset
    """
    dataset = dataset.copy()
    
    #Transform the datetime, needed for the submission file, the localize makes sure that the format is correct (rather than having the UCT reference)
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], utc=True).dt.tz_localize(None)  

    # Boolean check, that it is true when hour, minute and second are all 0
    is_midnight = (dataset['datetime'].dt.hour == 0) & (dataset['datetime'].dt.minute == 0) & (dataset['datetime'].dt.second == 0)
    
    # What where does is, that when it is midnight, then it 
    datetime_str = np.where(
        is_midnight,
        dataset['datetime'].dt.strftime('%Y-%m-%d'),
        dataset['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    )

    # The insert method takes (position_index, column_name, values)
    dataset.insert(0, 'id', datetime_str + "_" + dataset.iloc[:, 1].astype(str))

    if is_train:    
        # If there are rows with NA, then drop them
        if dataset.isna().any(axis=1).any():
            dataset = dataset.dropna()

        # If there are duplicates, then drop them.
        if dataset.duplicated().any():
            dataset = dataset.drop_duplicates()

    #I want to split my datainto the datapart (how already fastai tabular uses to learn)
    dataset['hour'] = dataset['datetime'].dt.hour
    dataset['minute'] = dataset['datetime'].dt.minute  # 0 or 30
    dataset['dayofweek'] = dataset['datetime'].dt.dayofweek
    dataset['month'] = dataset['datetime'].dt.month
    dataset['is_weekend'] = dataset['dayofweek'].isin([5, 6]).astype(int)

    # For cyclical data I can help the model differenciate between simple ordered rankings and actual circular data. In this case I do it with the hour (so it does not jump from 23 to 0 directly)
    dataset['hour_sin'] = np.sin(2 * np.pi * dataset['hour'] / 24)
    dataset['hour_cos'] = np.cos(2 * np.pi * dataset['hour'] / 24)

    # Create Austria holidays object
    at_holidays = holidays.Austria(subdiv='9') #subdiv = 9 locates the specific holidays of viena


    # Create the holiday column
    dataset['is_holiday'] = dataset['datetime'].dt.date.apply(lambda x: x in at_holidays)

    return dataset

    

    
