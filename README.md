<h1 align="center">PSD of epoched data</h1>

<p align="center">
  Computes the power spectral density of epoched MEG/EEG data per channel type (gradiometers, magnetometers, EEG) with the MNE-Python multitaper method, with PSD plots.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.587"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.587-blue.svg" alt="Run on brainlife.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-see_LICENSE-lightgrey.svg" alt="License"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Computes the power spectral density of epoched MEG/EEG data per channel type (gradiometers, magnetometers, EEG) with the MNE-Python multitaper method, with PSD plots.**

Reads MNE epochs, computes the multitaper PSD (mne.time_frequency.psd_multitaper) separately for grad, mag and EEG channels within the chosen frequency and time range, averages across epochs, converts to dB and saves one TSV (channels x frequencies) per channel type plus figures. Use it to inspect spectral content of epoched data. Only channel types present in the data produce output.

| | |
|---|---|
| **Inputs** | `neuro/meeg/mne/epochs` |
| **Outputs** | `neuro/meeg/psd` ×3 · `raw` |
| **Container** | `brainlife/mne:0.23dev` |
| **Resources** | 8 CPUs · 0.17 h walltime |
| **Run it** | [doi.org/10.25663/brainlife.app.587](https://doi.org/10.25663/brainlife.app.587) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `megfif` | `neuro/meeg/mne/epochs` (epochs) | yes | Epoched MEG/EEG data in MNE -epo.fif format. |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `out_psd_grad` | `neuro/meeg/psd` | psd.tsv with the epoch-averaged gradiometer PSD in dB (channels x frequencies). |
| `out_psd_mag` | `neuro/meeg/psd` | psd.tsv with the epoch-averaged magnetometer PSD in dB (channels x frequencies). |
| `out_psd_eeg` | `neuro/meeg/psd` | psd.tsv with the epoch-averaged EEG PSD in dB, uV^2/Hz (channels x frequencies). |
| `out_figs` | `raw` | PNG figures of the PSD (psd_mne.png and psd_computed.png). |

<details>
<summary><b>Parameters</b> (10)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fmin` | number | `0` | Min frequency of interest. |
| `fmax` | number | `100` | Max frequency of interest. |
| `tmin` | number |  | Min time of interest. |
| `tmax` | number |  | Max time of interest. |
| `bandwidth` | number | `4` | The bandwidth of the multi taper windowing function in Hz. The default value is a window half-bandwidth of 4. |
| `adaptive` | boolean | `false` | Use adaptive weights to combine the tapered spectra into PSD (slow, use n_jobs >> 1 to speed up computation). |
| `low_bias` | boolean | `true` | Only use tapers with more than 90% spectral concentration within bandwidth. |
| `normalization` | enum | `full` | Either “full” or “length” (default). If “full”, the PSD will be normalized by the sampling rate as well as the length of the signal (as in nitime). Options: `full`, `length`. |
| `proj` | boolean | `false` | Apply SSP projection vectors. |
| `picks` | string | `None` | Not read: the code always sets picks to None and processes grad, mag and EEG channels separately. |

</details>

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.587](https://doi.org/10.25663/brainlife.app.587), choose `neuro/meeg/mne/epochs` from your project, set the parameters and press **Execute**. The results are saved to your project as `neuro/meeg/psd`, `neuro/meeg/psd`, `neuro/meeg/psd`, `raw`.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/) (the App runs in `brainlife/mne:0.23dev`), [`jq`](https://jqlang.github.io/jq/).

1. Clone this repository.
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {
       "epo": "/path/to/megfif/epo",
       "fmin": 0,
       "fmax": 100,
       "tmin": "<number>",
       "tmax": "<number>",
       "bandwidth": 4,
       "adaptive": false,
       "low_bias": true,
       "normalization": "full",
       "proj": false,
       "picks": "None"
   }
   ```

3. Run `./main`.

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- PSD of epoched data. brainlife.io App. [doi:10.25663/brainlife.app.587](https://doi.org/10.25663/brainlife.app.587)
- Gramfort A, et al. MEG and EEG data analysis with MNE-Python. Front Neurosci 7:267 (2013). [doi:10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

- Guiomar Niso, Instituto Cajal, CSIC, Spain

## Funding

[![Wellcome-226486/Z/22/Z](https://img.shields.io/badge/Wellcome-226486%2FZ%2F22%2FZ-purple.svg)](https://wellcome.org/)
[![NIH-NINDS-U24NS140384](https://img.shields.io/badge/NIH_NINDS-U24NS140384-green.svg)](https://reporter.nih.gov/project-details/11261037)
[![NIH-NINDS-UM1NS132207](https://img.shields.io/badge/NIH_NINDS-UM1NS132207-green.svg)](https://reporter.nih.gov/project-details/11411328)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://reporter.nih.gov/project-details/10262925)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://reporter.nih.gov/project-details/10428625)
[![NIH-NIMH-R01MH126699](https://img.shields.io/badge/NIH_NIMH-R01MH126699-green.svg)](https://reporter.nih.gov/project-details/10460628)
[![NIH-NIDCD-R01DC020088](https://img.shields.io/badge/NIH_NIDCD-R01DC020088-green.svg)](https://reporter.nih.gov/project-details/11140284)
[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-2203524](https://img.shields.io/badge/NSF_BCS-2203524-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2203524)
[![NSF-IIS-1636893](https://img.shields.io/badge/NSF_IIS-1636893-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-OAC-2148729](https://img.shields.io/badge/NSF_OAC-2148729-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148729)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NSF-IIS-2148700](https://img.shields.io/badge/NSF_IIS-2148700-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148700)
[![NSF-OAC-1916518](https://img.shields.io/badge/NSF_OAC-1916518-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1916518)

<sub>brainlife.io, which runs this App, is publicly funded by Wellcome, the U.S. National Institutes of Health and the U.S. National Science Foundation; acknowledging that support helps keep it running.</sub>

## License

See [LICENSE](LICENSE).
