import mne
from mne.datasets import eegbci
#downloading EEG (Electroencephalogram) recordings
#coming from a free public database hosted by PhysioNet
# 1. Download/Load Subject 1's Motor Imagery data (Run 4 = imagining left vs right hand)
subject = 1
runs = [4] 

raw_fnames = eegbci.load_data(subject, runs)
raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
raw = mne.concatenate_raws(raws)

# Strip any white spaces from channel names to make them standard
mne.datasets.eegbci.standardize(raw)

# 2. Inspect what we have
print("--- EEG DATA SUMMARY ---")
print(f"Sampling Rate: {raw.info['sfreq']} Hz")
print(f"Number of Channels: {len(raw.ch_names)}")
print(f"Channel Names: {raw.ch_names[:10]} ...")
print(f"Data Shape (Channels x Timepoints): {raw.get_data().shape}")