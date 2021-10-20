# Copyright (c) 2020 brainlife.io
#
# This file is a MNE python-based brainlife.io App
#
# Author: Guiomar Niso
# Indiana University

# Required libraries
# pip install mne-bids coloredlogs tqdm pandas scikit-learn json_tricks fire

# set up environment
#import mne-study-template
import os
import json
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Current path
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Load brainlife config.json
with open(__location__+'/config.json') as config_json:
    config = json.load(config_json)

# == LOAD DATA ==
# FIF
fname = config['fif']
raw = mne.io.read_raw_fif(fname)

# CTF
# fname = config['ctf']
# raw = mne.io.read_raw_ctf(fname)

# == GET CONFIG VALUES ==

fmin = config['fmin']
fmax=config['fmax']
#fmax=config['fmax'] if config['fmax'] else inf
average = config['average']

if config['picks']:
    #If its a list starting with square braket, convert to list of strings
    if config['picks'].find("[") == 0:
        picks = config['picks'].replace('[','').replace(']','').split(", ")
    else:
        picks = config['picks']   
else: 
    picks=None

# Advanced parameters
tmin=config['tmin'] if config['tmin'] else None
tmax=config['tmax'] if config['tmax'] else None
n_fft = config['n_fft']
n_overlap = config['n_overlap']
n_per_seg=config['n_per_seg'] if config['n_per_seg'] else None
window = config['window']
proj = config['proj']
reject_by_annotation = config['reject_by_annotation']
#n_jobs = config['n_jobs']
#verbose = config['verbose']

#picks= ['MEG 0112', 'MEG 0113']

print(tmin)
print(picks)


# == GET SELECTED CHANNELS ==
# Find selected channels indexes
#info = mne.io.read_info(fname)
info=raw.info
# If picks is left to by default (GUIO) -- USAR PICKS CASO GENERICO!!
if picks==None:
    ichan = mne.pick_types(info, meg=True, eeg=True, ref_meg=False, exclude=info['bads'])
    # Get channel names
    canales = np.take(raw.ch_names,ichan)
else:
    canales=picks


# == COMPUTE PSD ==
if picks==None:

    picks_mag='mag'
    psd_welch_mag, freqs_mag = mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                             n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, window=window, picks=picks_mag, proj=proj,
                             reject_by_annotation=reject_by_annotation, average=average, n_jobs=1, verbose=None)
    # Convert power to dB scale.
    psd_welch_mag = 10*(np.log10(psd_welch_mag*1e15**2)) # T^2/hz -> fT^2/Hz

    picks_grad='grad'
    psd_welch_grad, freqs_grad = mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                             n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, window=window, picks=picks_grad, proj=proj,
                             reject_by_annotation=reject_by_annotation, average=average, n_jobs=1, verbose=None)
    # Convert power to dB scale.
    psd_welch_grad = 10*(np.log10(psd_welch_grad*1e13**2)) ## (T/m)^2/hz -> (fT/cm)^2/Hz

    '''picks_eeg='eeg'
    psd_welch_eeg, freqs_eeg = mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                             n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, window=window, picks=picks_eeg, proj=proj,
                             reject_by_annotation=reject_by_annotation, average=average, n_jobs=1, verbose=None)
    # Convert power to dB scale.
    psd_welch_eeg = 10*(np.log10(psd_welch_eeg*1e6**2)) ## V^2/hz -> uV^2/Hz'''

    # FIGURE 1
    # Plot computed Welch PSD
    plt.figure(1)
    fig, axs = plt.subplots(3)
    '''axs[0].plot(freqs_eeg, psd_welch_eeg.transpose(), zorder=1) 
    axs[0].set_xlim(xmin=0, xmax=max(freqs_eeg))
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Power Spectral Density')'''

    axs[1].plot(freqs_grad, psd_welch_grad.transpose(), zorder=1) 
    axs[1].set_xlim(xmin=0, xmax=max(freqs_grad))
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('PSD - grad')

    axs[2].plot(freqs_mag, psd_welch_mag.transpose(), zorder=1) 
    axs[2].set_xlim(xmin=0, xmax=max(freqs_mag))
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('PSD - mag')

    # Save fig
    plt.savefig(os.path.join('out_figs','psd_computed.png'))

else:
    #SPECIFIC CHANNELS

    psd_welch, freqs = mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                             n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, window=window, picks=picks, proj=proj,
                             reject_by_annotation=reject_by_annotation, average=average, n_jobs=1, verbose=None)

    # Convert power to dB scale.
    psd_welch = 10*(np.log10(psd_welch) + (2*15)) #psd_welch*(10**(2*15)) // psd_welch*1e30  # T**2/hz -> fT**2/Hz

    # FIGURE 1
    # Plot computed Welch PSD
    plt.figure(1)
    plt.plot(freqs, psd_welch.transpose(), zorder=1) 
    plt.xlim(xmin=0, xmax=max(freqs))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Computed PSD')
    # Save fig
    plt.savefig(os.path.join('out_figs','psd_computed.png'))



# == SAVE FILE ==
# Save to CSV file (could be also TSV)
df_psd = pd.DataFrame(psd_welch, index=canales, columns=freqs)
df_psd.index.name='channels'
df_psd.to_csv(os.path.join('out_dir','psd.csv')) #, sep = '\t', index=False)

# Read CSV file
#df = pd.read_csv("df_psd.csv")
#print(df)


# ==== PLOT FIGURES ====
# FIGURE 2
# Plot MNE PSD
plt.figure(2)
raw.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, proj=proj, n_fft=n_fft, n_overlap=n_overlap, window=window, 
            ax=None, color='black', xscale='linear', area_mode='std', area_alpha=0.33, 
            dB=True, estimate='auto', show=True, n_jobs=1, average=False, 
            line_alpha=None, spatial_colors=True, sphere=None, verbose=None)
# Save fig
plt.savefig(os.path.join('out_figs','psd_mne.png'))

