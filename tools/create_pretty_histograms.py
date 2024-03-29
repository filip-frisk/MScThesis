from atlasify import atlasify
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 


def plot_one_physical_variable(df, physical_variable, unit, signal, background, cut,DATA_FILENAME_WITHOUT_FILETYPE, OVERFLOW_UNDERFLOW_PERCENTILE,BINS):
    plt.figure()
    
    plot_data = []
    plot_weights = []
    plot_labels = []
    
    # colors as https://arxiv.org/pdf/2207.00338.pdf
    # Zjets #18bc14
    # WW #6864cc
    # ttbar #f0ec44
    # VBF #ff6404

    colors = ['#18bc14','#6864cc','#f0ec44','#ff6404']
    
    # get weight data, labels and weights
    for bkg in sorted(background, key=lambda x: len(df.loc[df['label'] == x]), reverse=True): # sort to get the largest background in the bottom of histogram
        plot_data.append(df.loc[df['label'] == bkg][physical_variable])
        plot_weights.append(df.loc[df['label'] == bkg]['weight'])
        plot_labels.append(bkg)
        
    # get signal data, label and weights 
    plot_data.append(df.loc[df['label'] == signal[0]][physical_variable]) 
    plot_weights.append(df.loc[df['label'] == signal[0]]['weight'])
    plot_labels.append(signal[0])
    
    
    # Calculate the 5th and 95th percentile to exclude extreme outliers
    flattened_data = np.concatenate(plot_data)
    lower_bound = np.percentile(flattened_data, OVERFLOW_UNDERFLOW_PERCENTILE['lower_bound'])    
    upper_bound = np.percentile(flattened_data, OVERFLOW_UNDERFLOW_PERCENTILE['upper_bound'])

    bounds = (lower_bound, upper_bound) 

    Overflow = len(flattened_data[flattened_data > upper_bound])
    Underflow = len(flattened_data[flattened_data < lower_bound])

    Overflow_ratio = Overflow/len(flattened_data)
    Underflow_ratio = Underflow/len(flattened_data)
    
    print(f"With added bounds of [{OVERFLOW_UNDERFLOW_PERCENTILE['lower_bound']},{OVERFLOW_UNDERFLOW_PERCENTILE['upper_bound']}] percentile we get {Underflow} Underflow events or {Underflow_ratio*100:.2f}% and {Overflow} Overflow events or {Overflow_ratio*100:.2f}%" )

    # Add overflow and underflow to the last and first bin respectively
    for channel in plot_data:
        channel = np.clip(channel, lower_bound, upper_bound)

    SIGNAL_ENVELOPE_SCALE = 5000 # easier to guess than to scale dynamically 

    # get signal data and scale it to the largest background
    signal_scaled = df.loc[df['label'] == signal[0]][physical_variable] 
    signal_weight = df.loc[df['label'] == signal[0]]['weight']*SIGNAL_ENVELOPE_SCALE
    signal_label = f"{signal[0]} x {SIGNAL_ENVELOPE_SCALE}"

    # Normalize weights by the bin width
    bin_width = (upper_bound - lower_bound)/BINS
    print(f"Bin width of {bin_width:.2f}")
    normalized_signal_weight = [w / bin_width for w in signal_weight]
    normalized_plot_weights = [[w / bin_width for w in single_stack] for single_stack in plot_weights]

    
    # plot signal shape and stacked shape of background and signal
    plt.hist(signal_scaled, bins=BINS, label=signal_label, weights=normalized_signal_weight, range=bounds, histtype='step', color=colors[-1])
    plt.hist(plot_data, bins=BINS, label=plot_labels, weights=normalized_plot_weights, stacked=True, range=bounds, alpha=0.5, histtype='stepfilled', color=colors)
    plt.xlim(lower_bound, upper_bound) # No white space 
    plt.xlabel(r'${} \ [{}]$'.format(physical_variable, unit))
    plt.ylabel(f'Events/{bin_width:.2f} {unit}')
    plt.legend(loc='upper right')
    
    subtext_latex = r'Internal KTH: Run 2 Monte Carlo data, all campaigns' + '\n' + r'$HWW \rightarrow WW^* \rightarrow l\nu l\nu$' + '\n' + r'SF VBF $N_{jet}$ $\geq$ 2' + '\n' + r'rootfilename: {}'.format(cut)
    
    atlasify(subtext = subtext_latex , sub_font_size = 8)
    
    ax = plt.gca() # Get the current axis
    ax.ticklabel_format(style='plain', axis='both', scilimits=(0,0))

    os.chdir('plots/')
    plt.savefig(f'prefit_histogram_{DATA_FILENAME_WITHOUT_FILETYPE}_{physical_variable}.png')
    os.chdir('..')
    
    # print what you have saved in short format
    print(f"Saved prefit_histogram_{DATA_FILENAME_WITHOUT_FILETYPE}_{physical_variable}.png in plots/ .")
    
    #plt.show()
    plt.close()

#if __name__ == '__main__':
    
    #plot_one_physical_variable(df, 'ptJ1','GeV', signal, bkg, (0,4*10**5))

