
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Assuming that eventweight has name weigth and channel has name label and eventType (bkg or signal) has name eventType

# The 'stratify' parameter is set to the 'label' column to maintain its distribution in the train and test datasets.
def pre_process_data(df,TRAIN_DATA_SIZE,RANDOM_SEED,EXPERIMENT_ID,DATA_RELATIVE_FOLDER_PATH,DATA_FILENAME_WITHOUT_FILETYPE,K_FOLD,CLASS_WEIGHT,SIGNAL_CHANNEL,BACKGROUND_CHANNEL):

    print(f"You have chosen class wieghting: {CLASS_WEIGHT}.")

    # check for nan values in data
    
    print(f"\nFound {df.isnull().sum().sum()} NaN values in MC Samples\n")

    # df.dropna() # if needed 

    # Counting all duplicates in data 
    num_duplicates = df.duplicated(keep=False).sum()

    # df.drop_duplicates(inplace=True) keeping first occurence 
    
    print(f"Found {num_duplicates} duplicates (two equal rows) in MC Samples\n")
    
    if CLASS_WEIGHT == 'as_is':
        df_train, df_test =train_test_split(df, train_size=TRAIN_DATA_SIZE, random_state=RANDOM_SEED, shuffle=True, stratify=df['label']) 
    
    elif CLASS_WEIGHT == 'bkg_as_sgn': # TODO : need to take as many as weight not as MC samples 

        signal_MCSamples = df[df['label'] == SIGNAL_CHANNEL[0]].shape[0]

        df_sgn = df[df['label'] == SIGNAL_CHANNEL[0]]

        dfs_bkg = []
        for bkg in BACKGROUND_CHANNEL:
            dfs_bkg.append(df[df['label'] == bkg].sample(n=signal_MCSamples, random_state=RANDOM_SEED))
        
        df = pd.concat([df_sgn] + dfs_bkg)

        df_train, df_test =train_test_split(df, train_size=TRAIN_DATA_SIZE, random_state=RANDOM_SEED, shuffle=True, stratify=df['label']) 
     
    elif CLASS_WEIGHT == 'tot_bkg_as_sgn':
        signal_MCSamples = df[df['label'] == SIGNAL_CHANNEL[0]].shape[0]

        df_sgn = df[df['label'] == SIGNAL_CHANNEL[0]]

        dfs_bkg = []
        for bkg in BACKGROUND_CHANNEL:
            dfs_bkg.append(df[df['label'] == bkg].sample(n=int(signal_MCSamples/len(BACKGROUND_CHANNEL)), random_state=RANDOM_SEED))
        
        df = pd.concat([df_sgn] + dfs_bkg)

        df_train, df_test =train_test_split(df, train_size=TRAIN_DATA_SIZE, random_state=RANDOM_SEED, shuffle=True, stratify=df['label']) 
        
    else:
        raise ValueError("CLASS_WEIGHT must be 'as_is' or 'bkg_as_VBF' or tot_bkg_as_VBF ")
    # proportions of train and test data if stratify is not None


    # count the total number of MC samples
    df_MCSamples= df.shape[0]
    df_train_MCSamples = df_train.shape[0]
    df_test_MCSamples = df_test.shape[0]

    # count as MC samples ratios of total
    df_MCSamples_ratio = df_MCSamples / df_MCSamples * 100
    df_train_MCSamples_ratio = df_train_MCSamples / df_MCSamples * 100
    df_test_MCSamples_ratio = df_test_MCSamples / df_MCSamples * 100
    
    # count event weight per data frame by label in fractions
    df_total_event_weight_by_label = df.groupby('label')['weight'].sum() / df['weight'].sum() * 100
    df_train_total_event_weight_by_label = df_train.groupby('label')['weight'].sum() / df_train['weight'].sum() * 100
    df_test_total_event_weight_by_label = df_test.groupby('label')['weight'].sum() / df_test['weight'].sum() * 100

    
    # count event weight per data frame of total
    df_total_event_weight = df.groupby('label')['weight'].sum().sum()
    df_train_total_event_weight = df_train.groupby('label')['weight'].sum().sum()
    df_test_total_event_weight = df_test.groupby('label')['weight'].sum().sum()



    # count as total event weight ratios of total
    df_total_event_weight_ratio = df_total_event_weight / df_total_event_weight * 100
    df_train_total_event_weight_ratio = df_train_total_event_weight / df_total_event_weight * 100
    df_test_total_event_weight_ratio = df_test_total_event_weight / df_total_event_weight * 100

    # count class label ratio as precentage of total
    df_label_ratio = df['label'].value_counts(normalize=True) * 100
    df_train_label_ratio = df_train['label'].value_counts(normalize=True) * 100
    df_test_label_ratio = df_test['label'].value_counts(normalize=True) * 100

    ORDERED_COLUMNS = df_label_ratio.index.tolist() # orders by size as default 

    # count eventType ratio as precentage of total
    df_eventType_ratio = df['eventType'].value_counts(normalize=True) * 100 
    df_train_eventType_ratio = df_train['eventType'].value_counts(normalize=True) * 100
    df_test_eventType_ratio = df_test['eventType'].value_counts(normalize=True) * 100
    
    # format all the above in a table for easy reading

    df_summary = pd.DataFrame({
        'MC Samples': [df_MCSamples, df_train_MCSamples, df_test_MCSamples],
        'MC Samples (%)': [df_MCSamples_ratio, df_train_MCSamples_ratio, df_test_MCSamples_ratio],
        'Total MC Event Weight': [df_total_event_weight, df_train_total_event_weight, df_test_total_event_weight],
        'Total MC Event Weight (%)': [df_total_event_weight_ratio, df_train_total_event_weight_ratio, df_test_total_event_weight_ratio]
    }, index=['Total', 'Train', 'Test'])

    print(df_summary)

    print("\n")
    print("Class label ratios in % of MCSamples:")
    label_ratios = pd.DataFrame({ #
        'Total': df_label_ratio,
        'Train': df_train_label_ratio,
        'Test': df_test_label_ratio
    }).transpose()
    
    print(label_ratios)

    print("\n")
    print("Class label ratios in % of Eventweight:")
    label_ratios = pd.DataFrame({
        'Total': df_total_event_weight_by_label[ORDERED_COLUMNS], 
        'Train': df_train_total_event_weight_by_label[ORDERED_COLUMNS],
        'Test': df_test_total_event_weight_by_label[ORDERED_COLUMNS]
    }).transpose()
    
    print(label_ratios)
    
    print("\n")
    print("EventType label (signal or process A or B .. or bkg) ratios in %:")
    eventType_ratios = pd.DataFrame({
        'Total': df_eventType_ratio,
        'Train': df_train_eventType_ratio,
        'Test': df_test_eventType_ratio
    }).transpose()

    print(eventType_ratios)

    print("\n")
    
    # save the dataframes to pickle files
    os.chdir(DATA_RELATIVE_FOLDER_PATH)
    os.makedirs(EXPERIMENT_ID, exist_ok=True)
    df_train.to_pickle(f'{EXPERIMENT_ID}/{DATA_FILENAME_WITHOUT_FILETYPE}_fold{K_FOLD}_{CLASS_WEIGHT}_train.pkl')
    df_test.to_pickle(f'{EXPERIMENT_ID}/{DATA_FILENAME_WITHOUT_FILETYPE}_fold{K_FOLD}_{CLASS_WEIGHT}_test.pkl')
    os.chdir('../..')




    
    

