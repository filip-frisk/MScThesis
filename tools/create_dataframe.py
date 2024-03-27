import pandas as pd
import uproot
import os 
import pickle

#DATA_RELATIVE_FOLDER_PATH = '../data/'
#DATA_PATH_WITHOUT_FILETYPE = 'ntuples-ggFVBF2jet-SF-28Jan24'


#SIGNAL_CHANNEL = ['VBF']
#BACKGROUND_CHANNEL = ['WW', 'Zjets', 'ttbar']

#SELECTED_OTHER_VARIABLES = ['eventType','label','eventNumber','weight']
#SELECTED_PHYSICAL_VARIABLES = ['DPhijj', 'mll', 'mT', 'DYjj', 'mjj', 'ptTot', 'mL1J1', 'mL1J2', 'mL2J1', 'mL2J2','ptJ1','ptJ2','ptJ3','METSig']

# Change the working directory to the data folder

# wrap the code in a function
def create_dataframe(DATA_RELATIVE_FOLDER_PATH, DATA_PATH_WITHOUT_FILETYPE, SIGNAL_CHANNEL, BACKGROUND_CHANNEL, SELECTED_OTHER_VARIABLES, SELECTED_PHYSICAL_VARIABLES):

    os.chdir(DATA_RELATIVE_FOLDER_PATH)
    print(f"Current working directory: {os.getcwd()}\n")

    with uproot.open(f'{DATA_PATH_WITHOUT_FILETYPE}.root') as root_file: # context manager to automatically close the file

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

    # cleaning the data
    df['label'] = df['label'].str.replace(';1', '').str.replace('HWW_', '') # clean the channel names for better readability
    # df = df.replace([np.inf, -np.inf], np.nan) # replace inf with nan in df if needed  In my case sumOfCentralitiesL and centralityL1 and centralityL2 are inf
    df.insert(0, 'eventType', df['label'].apply(lambda x: 'Signal' if x in SIGNAL_CHANNEL else 'Background')) # add a column with the event type

    print("\n")
    print(f"Variables from each channel: {', '.join(df.columns)}")
    print("\n")

    print(f"Dataframe sample without selections:\n{df.sample(5)}")


    print(f"Selected channels: {[SIGNAL_CHANNEL + BACKGROUND_CHANNEL]} with {SIGNAL_CHANNEL[0]} as signal.\n")
    df_selected = df[df['label'].isin(SIGNAL_CHANNEL + BACKGROUND_CHANNEL)]

    print(f"Selected physical variables {SELECTED_PHYSICAL_VARIABLES}\n")
    print(f"Selected other variables {SELECTED_OTHER_VARIABLES}\n")

    df_selected = df_selected[SELECTED_OTHER_VARIABLES + SELECTED_PHYSICAL_VARIABLES]

    print(f"Dataframe sample applied channel and variable selections:\n{df_selected.sample(5)}")

    # Save the dataframe to a pickle file
    df_selected.to_pickle(f'{DATA_PATH_WITHOUT_FILETYPE}.pkl')
