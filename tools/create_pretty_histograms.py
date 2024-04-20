from atlasify import atlasify
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 

from typing import List

def create_pretty_histograms(df: pd.DataFrame,
                                plot_variable: str, 
                                UNIT: str, 
                                SIGNAL: List[str], 
                                BACKGROUND: List[str], 
                                cut: str,
                                DATA_FILENAME_WITHOUT_FILETYPE: str, 
                                OVERFLOW_UNDERFLOW_PERCENTILE: dict[str,float],
                                BINS: int,
                                PLOT_RELATIVE_FOLDER_PATH: str,
                                PLOT_TYPE: str,
                                SIGNAL_ENVELOPE_SCALE: int,
                                NORMALIZE_WEIGHTS: bool,
                                K_FOLD: int,
                                EXPERIMENT_ID: str
                                ) -> None:
    
    ############################################################################
    ######################## PLOT MAIN HISTOGRAM ###############################
    ############################################################################
    
    ####### SELECT COLORS: CHANGE IF NEEDED! #######

    # colors as https://arxiv.org/pdf/2207.00338.pdf
    # Zjets #18bc14
    # WW #6864cc
    # ttbar #f0ec44
    # VBF #ff6404

    colors = ['#18bc14','#6864cc','#f0ec44','#ff6404']

    ####### LOAD DATA #######
    
    plot_data = []
    plot_weights = []
    plot_labels = []
    
    # get background data
    for bkg in sorted(BACKGROUND, key=lambda x: len(df.loc[df['label'] == x]), reverse=True): # sort to get the largest background in the bottom of histogram
        plot_data.append(df.loc[df['label'] == bkg][plot_variable])
        plot_weights.append(df.loc[df['label'] == bkg]['weight'])
        plot_labels.append(bkg)
        
    # get signal data (added last to be on top of the stack)
    plot_data.append(df.loc[df['label'] == SIGNAL[0]][plot_variable]) 
    plot_weights.append(df.loc[df['label'] == SIGNAL[0]]['weight'])
    plot_labels.append(SIGNAL[0])

    ####### Overflow and underflow /Inclusive and exclusive #######

    if PLOT_TYPE == 'prefit': # Under/over flow not needed for the prefit
    
        # Calculate the 5th and 95th percentile to exclude extreme outliers
        flattened_data = np.concatenate(plot_data)
        lower_bound = np.percentile(flattened_data, OVERFLOW_UNDERFLOW_PERCENTILE['lower_bound'])    
        upper_bound = np.percentile(flattened_data, OVERFLOW_UNDERFLOW_PERCENTILE['upper_bound'])

        bounds = (lower_bound, upper_bound) 

        Overflow = len(flattened_data[flattened_data > upper_bound])
        Underflow = len(flattened_data[flattened_data < lower_bound])

        Overflow_ratio = Overflow/len(flattened_data)
        Underflow_ratio = Underflow/len(flattened_data)
        
        print(f"With added bounds of [{OVERFLOW_UNDERFLOW_PERCENTILE['lower_bound']},{OVERFLOW_UNDERFLOW_PERCENTILE['upper_bound']}] percentile we get {Underflow}/{Underflow_ratio*100:.2f}% Underflow events and {Overflow}/{Overflow_ratio*100:.2f}% Overflow events " )

        # Add overflow and underflow to the last and first bin respectively
        plot_data = [np.clip(channel, lower_bound, upper_bound) for channel in plot_data]

       
    if PLOT_TYPE == 'postfit': #MVA output is always between 0 and 1 no under/over flow needed
        lower_bound = 0
        upper_bound = 1
        bounds = (lower_bound, upper_bound)
     

    # Create envelope for signal
    signal_scaled = df.loc[df['label'] == SIGNAL[0]][plot_variable] 
    signal_weight = df.loc[df['label'] == SIGNAL[0]]['weight'] * SIGNAL_ENVELOPE_SCALE
    signal_label = f"{SIGNAL[0]} x {SIGNAL_ENVELOPE_SCALE}"

    
    # Calculate bin width 
    bin_width = (upper_bound - lower_bound)/BINS
    print(f"Bin width of {bin_width:.2f}")

    ###### Normalize weights ######

    # Normalize weights by the bin width
    if NORMALIZE_WEIGHTS:    
        normalized_signal_weight = [weight / bin_width for weight in signal_weight]
        normalized_plot_weights = [[weight / bin_width for weight in single_stack] for single_stack in plot_weights]
    
    # Normal weights
    if not NORMALIZE_WEIGHTS:
        normalized_signal_weight = signal_weight
        normalized_plot_weights = plot_weights


    plt.figure()

    # plot signal shape and stacked shape of background and signal
    plt.hist(signal_scaled, bins=BINS, label=signal_label, weights=normalized_signal_weight, range=bounds, histtype='step', color=colors[-1])
    plt.hist(plot_data, bins=BINS, label=plot_labels, weights=normalized_plot_weights, stacked=True, range=bounds, alpha=0.5, histtype='stepfilled', color=colors)
    plt.xlim(lower_bound, upper_bound) # No white space 
    # plt.xlim(0, 1) # No white space 

    if PLOT_TYPE == 'prefit':   
        plt.xlabel(r'${} \ [{}]$'.format(plot_variable, UNIT))
        plt.ylabel(f'Events/{bin_width:.2f} {UNIT}')
    if PLOT_TYPE == 'postfit':
        plt.xlabel(f'{plot_variable}')
        plt.ylabel(f'Events')
    
    plt.legend(loc='upper right')
    
    subtext_latex = r'Internal KTH: Run 2 Monte Carlo data, all campaigns' + '\n' + r'$HWW \rightarrow WW^* \rightarrow l\nu l\nu$' + '\n' + r'SF VBF $N_{jet}$ $\geq$ 2' + '\n' + r'rootfilename: {}'.format(cut) + '\n' +f"{PLOT_TYPE}"
   
    if PLOT_TYPE == 'prefit':
        atlasify(subtext = subtext_latex + f" with {OVERFLOW_UNDERFLOW_PERCENTILE['lower_bound']}% overflow/underflow "  , sub_font_size = 8)
    if PLOT_TYPE == 'postfit':
        atlasify(subtext = subtext_latex   , sub_font_size = 8)
    
    ax = plt.gca() # Get the current axis
    ax.ticklabel_format(style='plain', axis='both', scilimits=(0,0))
    
    os.makedirs(PLOT_RELATIVE_FOLDER_PATH+EXPERIMENT_ID, exist_ok=True)

    os.chdir(PLOT_RELATIVE_FOLDER_PATH+EXPERIMENT_ID)
    
    plt.savefig(f'{PLOT_TYPE}_histogram_{DATA_FILENAME_WITHOUT_FILETYPE}_{plot_variable}.png',dpi = 600) # 
    os.chdir('../..')
    
    # print what you have saved in short format
    print(f"Saved {PLOT_TYPE}_histogram_{DATA_FILENAME_WITHOUT_FILETYPE}_{plot_variable}.png in plots/ .")
    
    #plt.show()
    plt.close()

    ############################################################################
    ######################## PLOT SIGNAL-TO-NOISE RATIO ########################
    ############################################################################

    plt.figure()
    
    # Calculate how many events we have in total and per label
    events_per_label = {}

    MCsamples_per_labels = df.groupby('label').size().to_dict()
    
    for bkg in BACKGROUND:
        events_per_label[bkg] = sum(df.loc[df['label'] == bkg]['weight'])
    events_per_label[SIGNAL[0]] = sum(df.loc[df['label'] == SIGNAL[0]]['weight'])
    
    print(f"Number of events per label: {events_per_label}")
    print(f"Number of MC samples per label: {MCsamples_per_labels}")

    # sgn + bkg = total
    # Get same histogram as above note that plt.hist uses np.histogram to get the binning
    total_hist, total_bin_edges = np.histogram(np.concatenate(plot_data), bins=BINS, range=bounds,weights=np.concatenate(normalized_plot_weights))
    # Need to scale pack to calculate correctly the signal-to-noise ratio
    signal_hist, signal_bin_edges = np.histogram(signal_scaled, bins=BINS, range=bounds,weights=[weight/SIGNAL_ENVELOPE_SCALE for weight in normalized_signal_weight])
    
    # check if total_bin_edges is same as signal_bin_edges
    if np.array_equal(total_bin_edges,signal_bin_edges):
        print("Bin edges are the same, OK!")
        print(f"Bin edges: {[round(bin_edge,2) for bin_edge in total_bin_edges.tolist()]}")
    else:
        print("Bin edges are not the same, ERROR CHECK BINNING")

    background_hist = total_hist - signal_hist # Tot. - bkg = sgn
    
    signal_to_noise_ratio = signal_hist/background_hist

    # if signal_to_noise_ratio has nan values print them
    if np.isnan(signal_to_noise_ratio).any():
        print(f"Signal-to-noise ratio has nan values at indices: {np.argwhere(np.isnan(signal_to_noise_ratio))}")
        signal_to_noise_ratio = np.nan_to_num(signal_to_noise_ratio, nan=0, posinf=0, neginf=0)
        print(f"Signal-to-noise ratio has been replaced with 0 at nan values.")
    
    print(f"Number of total events: {[round(hist,0) for hist in total_hist]} and in total {round(sum(total_hist),0)} events.")
    print(f"Number of signal events: {[round(hist,0) for hist in signal_hist]}")
    print(f"Number of background events: {[round(hist,0) for hist in background_hist]}")
    print(f"Signal-to-noise ratio: {[round(ratio,2) for ratio in signal_to_noise_ratio]}")

    plt.bar(total_bin_edges[:-1], signal_to_noise_ratio, width=bin_width, align='edge', color=colors[-1], alpha=0.5, edgecolor='black', linewidth=1.5)
    
    plt.xlim(bounds)
    
    # for styling of the x-axis
    total_bin_edges_rounded = [round(bin_edge,2) for bin_edge in total_bin_edges.tolist()]    
    x_labels = [f'[{round(total_bin_edges_rounded[i],2)},{round(total_bin_edges_rounded[i+1],2)}]' for i in range(len(total_bin_edges_rounded)-1)]
    print(f"X-labels: {x_labels}")

    plt.xticks(total_bin_edges[:-1]+bin_width/2, x_labels, rotation=0)
    
    plt.ylabel('(Tot. - Bkg)/Bkg')
    plt.xlabel(f'{plot_variable}')

    if PLOT_TYPE == 'prefit':
        plt.xticks(rotation=90)
        plt.tight_layout()

    atlasify(atlas = False)

    os.chdir(PLOT_RELATIVE_FOLDER_PATH+EXPERIMENT_ID)

    plt.savefig(f'{PLOT_TYPE}_histogram_{DATA_FILENAME_WITHOUT_FILETYPE}_{plot_variable}_signal_ratio.png',dpi = 600) #

    os.chdir('../..')

    print(f"Saved {PLOT_TYPE}_histogram_{DATA_FILENAME_WITHOUT_FILETYPE}_{plot_variable}_signal_ratio.png in plots/ .")

    
    plt.close()


    ############################################################################
    ################### COMBINED PLOT FOR MAIN AND SIGNAL-TO-NOISE #############
    ############################################################################

    
    _, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8,6), height_ratios = [3, 1])

    # Plot main histogram on ax_top
    ax_top.hist(plot_data, bins=BINS, label=plot_labels, weights=normalized_plot_weights, stacked=True, range=bounds, alpha=0.5, histtype='stepfilled', color=colors)
    ax_top.set_xlim(lower_bound, upper_bound) 
    
    ax_top.set_ylabel(f'Events')
    ax_top.legend(loc='upper right')
    ax_top.set_xticks([]) 

    if PLOT_TYPE == 'prefit':
        atlasify(atlas = True, axes=ax_top,subtext = subtext_latex + " " + str(int(sum(total_hist)))+ f" events with ca. {int(events_per_label[SIGNAL[0]])} signal events from {int(MCsamples_per_labels[SIGNAL[0]])} MC Samples, " + f"{OVERFLOW_UNDERFLOW_PERCENTILE['lower_bound']}% overflow/underflow " , sub_font_size = 8)
    if PLOT_TYPE == 'postfit':
        atlasify(atlas = True, axes=ax_top,subtext = subtext_latex + " " + str(int(sum(total_hist)))+ f" events with ca. {int(events_per_label[SIGNAL[0]])} signal events from {int(MCsamples_per_labels[SIGNAL[0]])} MC Samples." , sub_font_size = 8)
    
    atlasify(atlas = False, axes=ax_bottom, sub_font_size = 8)

    # Plot signal-to-noise ratio on ax_bottom
    ax_bottom.bar(total_bin_edges[:-1], signal_to_noise_ratio, width=bin_width, align='edge', color=colors[-1], alpha=0.5, edgecolor='black', linewidth=1.5)
    ax_bottom.set_xticks(total_bin_edges[:-1]+bin_width/2, x_labels)
    ax_bottom.set_xlim(bounds)
    
    ax_bottom_second_y_axis = ax_bottom.twinx()
    ax_bottom_second_y_axis.plot(total_bin_edges[:-1]+bin_width/2, signal_hist, color='black', marker='o', linestyle='--', linewidth=1.5)
    
    for i in range(len(signal_hist)):
        ax_bottom_second_y_axis.text(total_bin_edges[i]+bin_width/2, signal_hist[i]+0.3, f"{round((signal_to_noise_ratio[i]),1)}/{round((signal_hist[i]),1)}", ha='center', va='bottom')
    ax_bottom_second_y_axis.set_ylabel('Signal events')    
    
    ax_bottom.set_ylim(0, max(signal_to_noise_ratio)*1.1)
    ax_bottom.set_ylabel('(Tot. - Bkg)/Bkg')
    ax_bottom.set_xlabel(f'{plot_variable}')    
    
    if PLOT_TYPE == 'prefit':
        ax_bottom.set_xticklabels(x_labels, rotation=90)
        
    plt.tight_layout()

    os.chdir(PLOT_RELATIVE_FOLDER_PATH+EXPERIMENT_ID)

    plt.savefig(f'{PLOT_TYPE}_histogram_{DATA_FILENAME_WITHOUT_FILETYPE}_{plot_variable}_combined.png',dpi = 600) #

    print(f"Saved {PLOT_TYPE}_histogram_{DATA_FILENAME_WITHOUT_FILETYPE}_{plot_variable}_combined.png in plots/ .")

    os.chdir('../..')
    #plt.show()
    plt.close()

    ############################################################################
    ############################ 2207.00338 Table 5 ############################
    ############################################################################
  
    if PLOT_TYPE == 'postfit':
    
        # get total number of events in test data
        total_test_events = df.groupby('label')['weight'].sum().to_dict()
        total_test_events['Total'] = sum(total_test_events.values())
        total_test_events['Total bkg'] = total_test_events['Total'] - total_test_events[SIGNAL[0]]
        total_test_events['Sgn/Bkg'] = total_test_events[SIGNAL[0]]/total_test_events['Total bkg']
        

        bottom_edge_in_last_bin = total_bin_edges[-2]
        bottom_bin_string = x_labels[-1]

        # get total number of events in test data in the last bin

        # check if there are any events in the last bin 
        if len(df.loc[df[plot_variable] > bottom_edge_in_last_bin]) == 0:
            print(f"No events in the last bin {bottom_bin_string}.")
            last_bin_test_events = {key: 0 for key in total_test_events.keys()}
            last_bin_test_events['Sgn/Bkg'] = 0
        else:
            print(f"Events in the last bin {bottom_bin_string}.")
            last_bin_test_events = df.loc[df[plot_variable] > bottom_edge_in_last_bin].groupby('label')['weight'].sum().to_dict()
            last_bin_test_events['Total'] = sum(last_bin_test_events.values())
            last_bin_test_events['Total bkg'] = last_bin_test_events['Total'] - last_bin_test_events[SIGNAL[0]]
            last_bin_test_events['Sgn/Bkg'] = last_bin_test_events[SIGNAL[0]]/last_bin_test_events['Total bkg']
            

        # Creating DataFrame to save 
        df_tmp = pd.DataFrame({
            'Test Events': total_test_events,
            f'{plot_variable}:{bottom_bin_string}': last_bin_test_events
        })
        
        # if column starts with MVAOutput_ remove it
        if plot_variable.startswith('MVAOutput_'):
            df_tmp.columns = df_tmp.columns.str.replace('MVAOutput_', '') 
        
        df_tmp = df_tmp.round(decimals=1)
        # save to pickle
        os.chdir(PLOT_RELATIVE_FOLDER_PATH+EXPERIMENT_ID)

        table5_name = f'{PLOT_TYPE}_table5_{DATA_FILENAME_WITHOUT_FILETYPE}_fold{K_FOLD}'
        # if file exists
        if os.path.exists(table5_name+'.pkl'):
            df_tmp_existing = pd.read_pickle(table5_name+'.pkl')
            df_tmp = pd.concat([df_tmp_existing, df_tmp], axis=1)
            #df_tmp.round(decimals=1)
            df_tmp = df_tmp.loc[:,~df_tmp.columns.duplicated()]
            df_tmp.to_pickle(table5_name+'.pkl')
            print("File exists, adding to it.")
            # Save to csv and save over it if existing 
            df_tmp.to_csv(table5_name+'.csv')
        else:
            # delete csv and pickle if already exists
            if os.path.exists(table5_name+'.csv'):
                os.remove(table5_name+'.csv')
                
            if os.path.exists(table5_name+'.pkl'):
                os.remove(table5_name+'.pkl')

            df_tmp.to_pickle(table5_name+'.pkl')
            
            print("Saved temporary pickle file.")

        print(f"\nAll histograms for {plot_variable} are done\n")
        os.chdir('../..')