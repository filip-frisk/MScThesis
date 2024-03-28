from atlasify import atlasify
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 


def plot_one_physical_variable(df, physical_variable, unit, signal, background, cut,DATA_FILENAME_WITHOUT_FILETYPE):
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
    lower_bound = np.percentile(flattened_data, 5)    
    upper_bound = np.percentile(flattened_data, 95)

    # get data not included in bound from flattened data
    data_outside_bounds = flattened_data[(flattened_data < lower_bound) | (flattened_data > upper_bound)] # 10% of data of course
    print(len(data_outside_bounds)/len(flattened_data)*100, '% of data is outside the bounds')

    bounds = (lower_bound, upper_bound) # TODO implement overflow and underflow on last bin, add 50% of the data in last and first bin respectively?
    
    SCALE = 5000 # TODO fixate dynamically later

    # get signal data and scale it to the largest background
    signal_scaled = df.loc[df['label'] == signal[0]][physical_variable] 
    signal_weight = df.loc[df['label'] == signal[0]]['weight']*SCALE
    signal_label = f"{signal[0]} x {SCALE}"
    
    # plot signal shape and stacked shape of background and signal
    plt.hist(signal_scaled, bins=50, label=signal_label, weights=signal_weight, range=bounds, histtype='step', color=colors[-1])
    plt.hist(plot_data, bins=50, label=plot_labels, weights=plot_weights, stacked=True, range=bounds, alpha=0.5, histtype='stepfilled', color=colors)

    plt.xlabel(r'${} \ [{}]$'.format(physical_variable, unit))
    plt.ylabel('Events')
    plt.legend(loc='upper right')
    
    subtext_latex = r'Internal KTH: Run 2 Monte Carlo data, all campaigns' + '\n' + r'$HWW \rightarrow WW^* \rightarrow l\nu l\nu$' + '\n' + r'SF VBF $N_{jet}$ $\geq$ 2' + '\n' + r'rootfilename: {}'.format(cut)
    
    atlasify(subtext = subtext_latex , sub_font_size = 8)
    
    ax = plt.gca() # Get the current axis
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))

    os.chdir('plots/')
    plt.savefig(f'prefit_histogram_{DATA_FILENAME_WITHOUT_FILETYPE}_{physical_variable}.png')
    os.chdir('..')
    
    # print what you have saved in short format
    print(f"Saved prefit_histogram_{DATA_FILENAME_WITHOUT_FILETYPE}_{physical_variable}.png in plots/ folder.")
    
    #plt.show()
    plt.close()

#if __name__ == '__main__':
    
    #plot_one_physical_variable(df, 'ptJ1','GeV', signal, bkg, (0,4*10**5))

