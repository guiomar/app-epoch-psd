# Copyright (c) 2020 brainlife.io
#
# This file is a MNE python-based brainlife.io App
#
# Author: Guiomar Niso
# Indiana University

# set up environment
import os
import json
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fnmatch

# Current path
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Load brainlife config.json
with open(__location__+'/config.json') as config_json:
    config = json.load(config_json)

# == LOAD DATA ==
fname = config['epo']
epochs = mne.read_epochs(fname)


# == GET CONFIG VALUES ==
fmin = config['fmin']
fmax = config['fmax']
# Advanced parameters
tmin = config['tmin'] if config['tmin'] else None
tmax = config['tmax'] if config['tmax'] else None

bandwidth = config['bandwidth']
adaptive  = config['adaptive'] 
low_bias  = config['low_bias']
normalization = config['normalization']
proj   = config['proj']
n_jobs = 1
picks  = None

'''#For Welch
n_fft=2048
n_overlap=1000
n_per_seg=None
window='hamming'
reject_by_annotation=True
average='mean''''

'''
# Better don't allow picks as they can mix grad/mag/eeg and the unit 
# conversion won't be straight forward to match raw.plot_psd resultsif config['picks']:
    #If its a list starting with square braket, convert to list of strings
    if config['picks'].find("[") == 0:
        picks = config['picks'].replace('[','').replace(']','').split(", ")
    else:
        picks = config['picks']   
else: 
    picks=None
'''

# == COMPUTE PSD ==

# psd_welch.shape: Nepochs x Nchannels x Nfreqs

# Types of channels in the data
# e.g. ['ecg', 'eog', 'grad', 'mag', 'eeg','misc', 'stim']
ch_types=np.unique(epochs.get_channel_types())

if picks==None:

    # FIGURE 1
    # Number of subplots
    bb=0
    for i in ['grad','mag','eeg']: 
        if i in ch_types: bb=bb+1
    plt.figure(1)
    fig, axs = plt.subplots(bb)
    fig.subplots_adjust(hspace =.5, wspace=.2)

    aa=0

    if 'eeg' in ch_types:
        epochs_eeg = epochs.copy().pick('eeg')
        ch_eeg = epochs_eeg.ch_names
        psd_welch_eeg, freqs_eeg = mne.time_frequency.psd_multitaper(epochs_eeg, 
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                            bandwidth=bandwidth, adaptive=adaptive, low_bias=low_bias, normalization=normalization, 
                            picks='eeg', proj=proj, n_jobs=n_jobs, verbose=None)
        # Convert power to dB scale: V^2/hz -> uV^2/Hz
        psd_welch_eeg = 10*(np.log10(psd_welch_eeg.mean(axis=0)*1e6**2))

        # Save to TSV file
        df_psd = pd.DataFrame(psd_welch_eeg, index=ch_eeg, columns=freqs_eeg)
        df_psd.index.name='channels'
        df_psd.columns.name = 'freqs'
        df_psd.to_csv(os.path.join('out_psd_eeg','psd.tsv'), sep='\t')

        # Figure
        axs[aa].plot(freqs_eeg, psd_welch_eeg.transpose(), zorder=1) 
        axs[aa].set_xlim(xmin=0, xmax=max(freqs_eeg))
        axs[aa].set_xlabel('Frequency (Hz)')
        axs[aa].set_ylabel('uV^2/Hz [dB]')
        axs[aa].set_title('PSD - EEG')
        axs[aa].grid(linestyle=':')
        aa=aa+1
        
    if 'grad' in ch_types:
        epochs_grad = epochs.copy().pick('grad')
        ch_grad = epochs_grad.ch_names
        psd_welch_grad, freqs_grad = mne.time_frequency.psd_multitaper(epochs_grad, 
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                            bandwidth=bandwidth, adaptive=adaptive, low_bias=low_bias, normalization=normalization, 
                            picks='grad', proj=proj, n_jobs=n_jobs, verbose=None)
       ''' psd_welch_grad, freqs_grad = mne.time_frequency.psd_welch(epochs_grad, 
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                            n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, window=window, 
                            reject_by_annotation=reject_by_annotation, average=average, 
                            picks='grad', proj=proj,n_jobs=n_jobs, verbose=None)'''
        # Convert power to dB scale: (T/m)^2/hz -> (fT/cm)^2/Hz
        psd_welch_grad = 10*(np.log10(psd_welch_grad.mean(axis=0)*1e13**2))

        # Save to TSV file
        df_psd = pd.DataFrame(psd_welch_grad, index=ch_grad, columns=freqs_grad)
        df_psd.index.name='channels'
        df_psd.columns.name = 'freqs'
        df_psd.to_csv(os.path.join('out_psd_grad','psd.tsv'), sep='\t')

        # Figure
        axs[aa].plot(freqs_grad, psd_welch_grad.transpose(), zorder=1) 
        axs[aa].set_xlim(xmin=0, xmax=max(freqs_grad))
        axs[aa].set_xlabel('Frequency (Hz)')
        axs[aa].set_ylabel('(fT/cm)^2/Hz [dB]')
        axs[aa].set_title('PSD - Gradieometers')
        axs[aa].grid(linestyle=':')
        aa=aa+1

    if 'mag' in ch_types:
        epochs_mag = epochs.copy().pick('mag')
        ch_mag = epochs_mag.ch_names
        psd_welch_mag, freqs_mag = mne.time_frequency.psd_multitaper(epochs_mag, 
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                            bandwidth=bandwidth, adaptive=adaptive, low_bias=low_bias, normalization=normalization, 
                            picks='mag', proj=proj, n_jobs=n_jobs, verbose=None)
        # Convert power to dB scale: T^2/hz -> fT^2/Hz
        psd_welch_mag = 10*(np.log10(psd_welch_mag.mean(axis=0)*1e15**2))

        # Save to TSV file
        df_psd = pd.DataFrame(psd_welch_mag, index=ch_mag, columns=freqs_mag)
        df_psd.index.name='channels'
        df_psd.columns.name = 'freqs'
        df_psd.to_csv(os.path.join('out_psd_mag','psd.tsv'), sep='\t')

        # Figure
        axs[aa].plot(freqs_mag, psd_welch_mag.transpose(), zorder=1) 
        axs[aa].set_xlim(xmin=0, xmax=max(freqs_mag))
        axs[aa].set_xlabel('Frequency (Hz)')
        axs[aa].set_ylabel('fT^2/Hz [dB]')
        axs[aa].set_title('PSD - Magnetometers')
        axs[aa].grid(linestyle=':')
        aa=aa+1

    # Save fig
    plt.savefig(os.path.join('out_figs','psd_computed.png'))


# ==== PLOT FIGURES ====
# FIGURE 2
# Plot MNE PSD
plt.figure(2)
epochs.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, 
            proj=proj, picks=picks,
            bandwidth=bandwidth, adaptive=adaptive, low_bias=low_bias, normalization=normalization,
            ax=None, color='black', xscale='linear', area_mode='std', area_alpha=0.33, 
            dB=True, estimate='auto', show=True, n_jobs=n_jobs, average=False, 
            line_alpha=None, spatial_colors=True, sphere=None, verbose=None)

# Save fig
plt.savefig(os.path.join('out_figs','psd_mne.png'))

