import pandas as pd
import uproot
import os 
import pickle

from typing import List

def create_dataframe(DATA_RELATIVE_FOLDER_PATH: str, 
                     DATA_FILENAME_WITHOUT_FILETYPE: str, 
                     SIGNAL_CHANNEL: List[str], 
                     BACKGROUND_CHANNEL : List[str],
                     SELECTED_OTHER_VARIABLES : List[str], 
                     SELECTED_PHYSICAL_VARIABLES : List[str]
                     ) -> None:

    os.chdir(DATA_RELATIVE_FOLDER_PATH)

    with uproot.open(f'{DATA_FILENAME_WITHOUT_FILETYPE}.root') as root_file: # context manager to automatically close the file

        print("You successfully loaded the following trees/channels:")

        # Create a dataframe for each trees/channel in the root file
        dfs = []

        for tree in root_file.keys():
            channel = root_file[tree] # load the tree
            df_tmp = channel.arrays(library='pd') # convert to pandas dataframe
            df_tmp.insert(0, 'label', tree) # add a column with the channel name
            dfs.append(df_tmp) # add the dataframe to the list of dataframes
            print(f"* Tree/channel: {tree}, with {df_tmp.shape[0]:,} MC simulations and {df_tmp['weight'].sum():,.2f} total event weight.")
            
    df = pd.concat(dfs) # Add all channels to a large df

    ### CHECK HERE ###
    # Add label trimming relevant for my naming convention for my rootfile
    df['label'] = df['label'].str.replace(';1', '').str.replace('HWW_', '') # clean the channel names for better readability
    
    df.insert(0, 'eventType', df['label'].apply(lambda x: 'Signal' if x in SIGNAL_CHANNEL else 'Background')) # add a column with the event type

    print("\n")
    print(f"Data points from each channel in  {len(df.columns)} columns: {', '.join(df.columns)}")
    print("\n")
    
    total_weight = df.groupby('label')['weight'].sum().sum()
    formatted_total_weight = "{:,.0f}".format(total_weight)
    
    print(f"Dataframe sample without selections with {df.shape[0]:,} MC samples and total event weight {formatted_total_weight}:\n{df.sample(5)}\n")

    print(f"Selected channels: {SIGNAL_CHANNEL + BACKGROUND_CHANNEL} with {SIGNAL_CHANNEL[0]} as signal.\n")
    df_selected_channel = df[df['label'].isin(SIGNAL_CHANNEL + BACKGROUND_CHANNEL)]    

    print(f"Selected physical variables {SELECTED_PHYSICAL_VARIABLES}\n")
    print(f"Selected other variables {SELECTED_OTHER_VARIABLES}\n")

    df_selected_channel_and_variables = df_selected_channel[SELECTED_OTHER_VARIABLES + SELECTED_PHYSICAL_VARIABLES]

    total_weight = df_selected_channel_and_variables.groupby('label')['weight'].sum().sum()
    formatted_total_weight = "{:,.0f}".format(total_weight)

    print(f"Dataframe sample with applied selections with {df_selected_channel_and_variables.shape[0]:,} MC samples and total event weight {formatted_total_weight}:\n{df_selected_channel_and_variables.sample(5)}")

    # Save selected dataframe to a pickle file
    
    df_selected_channel_and_variables.to_pickle(f'{DATA_FILENAME_WITHOUT_FILETYPE}.pkl')

    print(f"\nDataframe saved to {DATA_FILENAME_WITHOUT_FILETYPE}.pkl in folder {DATA_RELATIVE_FOLDER_PATH}\n")
    
    # change back to the original directory
    os.chdir('..')
