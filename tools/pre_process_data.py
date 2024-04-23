
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from typing import List

# Assuming that eventweight has name weigth and channel has name label and eventType (bkg or signal) has name eventType

def pre_process_data(df: pd.DataFrame,
                     TRAIN_DATA_SIZE: float,
                     RANDOM_SEED: int,
                     EXPERIMENT_ID: str,
                     DATA_RELATIVE_FOLDER_PATH: str,
                     DATA_FILENAME_WITHOUT_FILETYPE: str,
                     K_FOLD: int,
                     CLASS_WEIGHT: str,
                     SIGNAL_CHANNEL: List[str],
                     BACKGROUND_CHANNEL: List[str]
                    ) -> None:

    print(f"You have chosen class weight: {CLASS_WEIGHT}.")

    # comment out if needed, this takes time to run
    """
    # check for nan values in data

    print(f"\nFound {df.isnull().sum().sum()} NaN values in MC Samples\n") 

    df.dropna() 

    print(f"Found {df.isnull().sum().sum()} NaN values in MC Samples after dropping\n")

    # Counting all duplicates in data 
    num_duplicates = df.duplicated(keep=False).sum() # comment out if needed, this takes time to run

    print(f"Found {num_duplicates} duplicates (two equal rows) in MC Samples\n")

    df.drop_duplicates(inplace=True) # keeping first occurence 

    print(f"Found {df.duplicated(keep=False).sum()} duplicates (two equal rows) in MC Samples after dropping\n")
    """
    # No selections
    if CLASS_WEIGHT == 'raw':
        # The 'stratify' parameter is set to the 'label' column to maintain its distribution in the train and test datasets
        df_train, df_test =train_test_split(df, train_size=TRAIN_DATA_SIZE, random_state=RANDOM_SEED, shuffle=True, stratify=df['label']) 
    
    # Same number of signal, bkg1, bkg2,... MC samples
    elif CLASS_WEIGHT == 'MC_EACH_bkg_as_sgn': 

        signal_MCSamples = df[df['label'] == SIGNAL_CHANNEL[0]].shape[0]

        df_sgn = df[df['label'] == SIGNAL_CHANNEL[0]]

        dfs_bkg = []
        for bkg in BACKGROUND_CHANNEL:
            dfs_bkg.append(df[df['label'] == bkg].sample(n=signal_MCSamples, random_state=RANDOM_SEED))
        
        df = pd.concat([df_sgn] + dfs_bkg)

        df_train, df_test =train_test_split(df, train_size=TRAIN_DATA_SIZE, random_state=RANDOM_SEED, shuffle=True, stratify=df['label']) 
    
    # Same number of signal as sum(bkg1, bkg2,...) MC samples
    elif CLASS_WEIGHT == 'MC_TOTAL_bkg_as_sgn':
        signal_MCSamples = df[df['label'] == SIGNAL_CHANNEL[0]].shape[0]

        df_sgn = df[df['label'] == SIGNAL_CHANNEL[0]]

        dfs_bkg = []
        for bkg in BACKGROUND_CHANNEL:
            dfs_bkg.append(df[df['label'] == bkg].sample(n=int(signal_MCSamples/len(BACKGROUND_CHANNEL)), random_state=RANDOM_SEED))
        
        df = pd.concat([df_sgn] + dfs_bkg)

        df_train, df_test =train_test_split(df, train_size=TRAIN_DATA_SIZE, random_state=RANDOM_SEED, shuffle=True, stratify=df['label']) 
    
    # Same number of signal, bkg1, bkg2,... class weight/events
    # Algorithm to balance the background events
    # First take 1000 events from each background
    # Then keep adding 1000 events from each background until the sum of the background events is equal to the signal event weight
    # If the sum of the background events is greater than the signal event weight, remove 1 event from the background events at a time

    elif CLASS_WEIGHT == 'CW_EACH_bkg_as_sgn':
        
        df_sgn = df[df['label'] == SIGNAL_CHANNEL[0]]

        # sum class weights for df_signal
        signal_event_weight = df_sgn['weight'].sum()

        print(f"Signal event weight: {signal_event_weight}")

        dfs_bkg = [] 

        for bkg in BACKGROUND_CHANNEL:

            df_bkg = pd.DataFrame()
            df_bkg_sample = df[df['label'] == bkg].sample(n=1000, random_state=RANDOM_SEED)
            df_bkg = pd.concat([df_bkg, df_bkg_sample])
            while df_bkg[df_bkg['label'] == bkg]['weight'].sum() < signal_event_weight:
                df_bkg_sample = df[df['label'] == bkg].sample(n=1000, random_state=RANDOM_SEED)
                df_bkg = pd.concat([df_bkg, df_bkg_sample])
                print(f"Current bkg ({bkg}) event weight: {df_bkg['weight'].sum()}")
            
            while df_bkg[df_bkg['label'] == bkg]['weight'].sum() > signal_event_weight:
                df_bkg = df_bkg.sample(n=df_bkg.shape[0]-1, random_state=RANDOM_SEED)
                print(f"Current bkg ({bkg})event weight: {df_bkg['weight'].sum()}")

            print("Finished balancing bkg - sgn is now : ", df_bkg['weight'].sum() - signal_event_weight)
            
            dfs_bkg.append(df_bkg)
        
        df = pd.concat([df_sgn] + dfs_bkg)

        df_train, df_test =train_test_split(df, train_size=TRAIN_DATA_SIZE, random_state=RANDOM_SEED, shuffle=True, stratify=df['label']) 

    
    # Same number of signal as sum(bkg1, bkg2,...) class weight/events
    # Same algorithm as above but but devided by len(BACKGROUND_CHANNEL)
    elif CLASS_WEIGHT == 'CW_TOTAL_bkg_as_sgn':

        df_sgn = df[df['label'] == SIGNAL_CHANNEL[0]] 
        
        # sum class weights for df_signal
        signal_event_weight = df_sgn['weight'].sum() 

        print(f"Signal event weight: {signal_event_weight}")

        signal_event_weight = df_sgn['weight'].sum() / len(BACKGROUND_CHANNEL)

        print(f"Signal event weight scaled with number of background channels: {signal_event_weight}")
        

        dfs_bkg = [] 

        for bkg in BACKGROUND_CHANNEL:

            df_bkg = pd.DataFrame()
            df_bkg_sample = df[df['label'] == bkg].sample(n=1000, random_state=RANDOM_SEED)
            df_bkg = pd.concat([df_bkg, df_bkg_sample])
            while df_bkg[df_bkg['label'] == bkg]['weight'].sum() < signal_event_weight:
                df_bkg_sample = df[df['label'] == bkg].sample(n=1000, random_state=RANDOM_SEED)
                df_bkg = pd.concat([df_bkg, df_bkg_sample])
                print(f"Current bkg ({bkg}) event weight: {df_bkg['weight'].sum()}")
            
            while df_bkg[df_bkg['label'] == bkg]['weight'].sum() > signal_event_weight:
                df_bkg = df_bkg.sample(n=df_bkg.shape[0]-1, random_state=RANDOM_SEED)
                print(f"Current bkg ({bkg})event weight: {df_bkg['weight'].sum()}")

            print("Finished balancing bkg - sgn is now : ", df_bkg['weight'].sum() - signal_event_weight)
            
            dfs_bkg.append(df_bkg)
        
        df = pd.concat([df_sgn] + dfs_bkg)

        df_train, df_test =train_test_split(df, train_size=TRAIN_DATA_SIZE, random_state=RANDOM_SEED, shuffle=True, stratify=df['label']) 
   
    else:
        raise ValueError("CLASS_WEIGHT must be one of 'raw', 'MC_EACH_bkg_as_sgn', 'MC_TOTAL_bkg_as_sgn', 'CW_EACH_bkg_as_sgn', 'CW_TOTAL_bkg_as_sgn'")
    
    ##############################################################################
    ###### BELOW CALCULATIONS IS ONLY TO PRINT OUT THE DATAFRAME STATISTICS ######
    ##############################################################################
    
    def calculate_dataframe_statistics(df, signal_channel = SIGNAL_CHANNEL, background_channel = BACKGROUND_CHANNEL):
        num_MCSamples_tot = df.shape[0]
        num_signal_MCSamples_tot = df[df['label'] == signal_channel[0]].shape[0]
        num_bkg_MCSamples_tot = df[df['label'] != signal_channel[0]].shape[0]
        
        num_bkg_MCSamples_each = {f'{bkg}': df[df['label'] == bkg].shape[0] for bkg in background_channel}
        
        num_events_tot = df['weight'].sum()
        num_bkg_events_tot = df[df['label'] != signal_channel[0]]['weight'].sum()
        num_signal_events_tot = df[df['label'] == signal_channel[0]]['weight'].sum()

        num_bkg_events_each = {f'{bkg}': df[df['label'] == bkg]['weight'].sum() for bkg in background_channel}
        
        statistics_MC = {
            f'Signal ({SIGNAL_CHANNEL[0]})': num_signal_MCSamples_tot,
            **num_bkg_MCSamples_each,
            'Tot. Bkg': num_bkg_MCSamples_tot,
            'Tot.': num_MCSamples_tot,
        }
        
        statistics_MC_ratio = {
            f'Signal ({SIGNAL_CHANNEL[0]})': num_signal_MCSamples_tot/num_MCSamples_tot,
            **{f'{bkg}': num_bkg_MCSamples_each[bkg]/num_MCSamples_tot for bkg in background_channel},
            'Tot. Bkg': num_bkg_MCSamples_tot/num_MCSamples_tot,
            'Tot.': 1,
        }
        statistics_Events = {
            f'Signal ({SIGNAL_CHANNEL[0]})': num_signal_events_tot,
            **num_bkg_events_each,
            'Tot. Bkg': num_bkg_events_tot,
            'Tot.': num_events_tot
        }

        statistics_Events_ratio = {
            f'Signal ({SIGNAL_CHANNEL[0]})': num_signal_events_tot/num_events_tot,
            **{f'{bkg}': num_bkg_events_each[bkg]/num_events_tot for bkg in background_channel},
            'Tot. Bkg': num_bkg_events_tot/num_events_tot,
            'Tot.': 1
        }

        return statistics_MC, statistics_MC_ratio, statistics_Events, statistics_Events_ratio
    
    # Assume df, df_train, df_test are already defined
    stats_total_MC, stats_total_MC_ratio, stats_total_Events, stats_total_Events_ratio = calculate_dataframe_statistics(df)
    stats_train_MC, stats_train_MC_ratio, stats_train_Events, stats_train_Events_ratio = calculate_dataframe_statistics(df_train)
    stats_test_MC, stats_test_MC_ratio, stats_test_Events, stats_test_Events_ratio = calculate_dataframe_statistics(df_test)

    # Convert the dictionaries into a DataFrame
    df_statistics_MC = pd.DataFrame({
        'Train_num': stats_train_MC,
        'Train_ratio': stats_train_MC_ratio,
        'Test_num': stats_test_MC,
        'Test_ratio': stats_test_MC_ratio,
        'Total_num': stats_total_MC,
        'Total_ratio': stats_total_MC_ratio
    })

    df_statistics_Events = pd.DataFrame({
        'Train_num': stats_train_Events,
        'Train_ratio': stats_train_Events_ratio,
        'Test_num': stats_test_Events,
        'Test_ratio': stats_test_Events_ratio,
        'Total_num': stats_total_Events,
        'Total_ratio': stats_total_Events_ratio
    })

    # Calculate signal/bkg ratio for each variable
    signal_row_events = df_statistics_Events.loc[f'Signal ({SIGNAL_CHANNEL[0]})'].to_dict()
    tot_bkg_row_events = df_statistics_Events.loc['Tot. Bkg'].to_dict()    
    
    
    signal_bkg_ratio_events = {key: f'{signal_row_events[key]/tot_bkg_row_events[key]:.1%}'if key.endswith('_num') else '' for key in signal_row_events.keys()}

    signal_row_MC = df_statistics_MC.loc[f'Signal ({SIGNAL_CHANNEL[0]})'].to_dict()
    tot_bkg_row_MC = df_statistics_MC.loc['Tot. Bkg'].to_dict()

    signal_bkg_ratio_MC = {key: f'{signal_row_MC[key]/tot_bkg_row_MC[key]:.1%}'if key.endswith('_num') else '' for key in signal_row_MC.keys()}
   
    # format the dataframe and add last row with other format 
    df_statistics_MC['Train_num'] = df_statistics_MC['Train_num'].map('{:,.1f}'.format)
    df_statistics_MC['Test_num'] = df_statistics_MC['Test_num'].map('{:,.1f}'.format)
    df_statistics_MC['Total_num'] = df_statistics_MC['Total_num'].map('{:,.1f}'.format)
    df_statistics_MC['Train_ratio'] = df_statistics_MC['Train_ratio'].map('{:.1%}'.format)
    df_statistics_MC['Test_ratio'] = df_statistics_MC['Test_ratio'].map('{:.1%}'.format)
    df_statistics_MC['Total_ratio'] = df_statistics_MC['Total_ratio'].map('{:.1%}'.format)

    df_statistics_MC.loc['Signal/Bkg'] = signal_bkg_ratio_MC

    df_statistics_Events['Train_num'] = df_statistics_Events['Train_num'].map('{:,.1f}'.format)
    df_statistics_Events['Test_num'] = df_statistics_Events['Test_num'].map('{:,.1f}'.format)
    df_statistics_Events['Total_num'] = df_statistics_Events['Total_num'].map('{:,.1f}'.format)
    df_statistics_Events['Train_ratio'] = df_statistics_Events['Train_ratio'].map('{:.1%}'.format)
    df_statistics_Events['Test_ratio'] = df_statistics_Events['Test_ratio'].map('{:.1%}'.format)
    df_statistics_Events['Total_ratio'] = df_statistics_Events['Total_ratio'].map('{:.1%}'.format)

    df_statistics_Events.loc['Signal/Bkg'] = signal_bkg_ratio_events

    print(f"\nMC Samples statistics\n")
    print(df_statistics_MC)

    print(f"\nEvents statistics\n")
    print(df_statistics_Events)

    # save the dataframes to pickle files
    os.chdir(DATA_RELATIVE_FOLDER_PATH)
    os.makedirs(EXPERIMENT_ID, exist_ok=True)
    df_train.to_pickle(f'{EXPERIMENT_ID}/{DATA_FILENAME_WITHOUT_FILETYPE}_fold{K_FOLD}_{CLASS_WEIGHT}_train.pkl')
    df_test.to_pickle(f'{EXPERIMENT_ID}/{DATA_FILENAME_WITHOUT_FILETYPE}_fold{K_FOLD}_{CLASS_WEIGHT}_test.pkl')
    os.chdir('..')