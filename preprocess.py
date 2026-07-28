import mne
from mne.datasets import eegbci



def load_and_filter_data(subject, runs=[3, 4, 7, 8, 11, 12]):
    """Loads Left/Right hand execution + imagery runs (~90 trials total)."""
    raw_fnames = eegbci.load_data(subject, runs, verbose=False)
    raw_list = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    raw = mne.concatenate_raws(raw_list)
    
    # FIX: Strip trailing dots from channel names (e.g., 'Fc3.' -> 'FC3')
    raw.rename_channels(lambda x: x.strip('.').upper())
    
    # 8-30 Hz bandpass filter for motor rhythms
    raw.filter(l_freq=8.0, h_freq=30.0, fir_design='firwin', verbose=False)
    return raw
def get_epochs(raw_filtered):
    """
    Finds event markers in the filtered data and slices it 
    into individual trials (Epochs) for Left vs Right hand.
    """
    print("--- Slicing Data into Epochs (Trials) ---")
    
    events, event_dict = mne.events_from_annotations(raw_filtered, verbose=False)
    
    target_event_id = {}
    if 'T1' in event_dict:
        target_event_id['left_hand'] = event_dict['T1']
    if 'T2' in event_dict:
        target_event_id['right_hand'] = event_dict['T2']

    # FIX: Use all-uppercase 'FCZ', 'CZ', 'CPZ' to match raw.rename_channels()
    motor_channels = [
        'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
        'C3',  'C1',  'CZ',  'C2',  'C4',
        'CP3', 'CP1', 'CPZ', 'CP2', 'CP4'
    ]
    
    epochs = mne.Epochs(
        raw_filtered, 
        events, 
        event_id=target_event_id, 
        tmin=0.5, 
        tmax=3.5, 
        picks=motor_channels,
        baseline=None,
        preload=True,
        verbose=False
    )
    
    print(f"Slicing complete! Created {len(epochs)} total trials.")
    print(f"Left Hand trials: {len(epochs['left_hand'])}")
    print(f"Right Hand trials: {len(epochs['right_hand'])}")
    
    return epochs

if __name__ == "__main__":
    # Test our complete preprocessing pipeline
    filtered_raw = load_and_filter_data(subject=1, runs=[3, 4, 7, 8, 11, 12])
    epochs = get_epochs(filtered_raw)