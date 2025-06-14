import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import mne

def process_session(data_path: str, label_path: str, dataset: str = "2a", fs: int = 250):

    raw = mne.io.read_raw_gdf(data_path, preload=True, verbose=False)
    if dataset == "2b":
        target_eeg_names = ['EEG-C3', 'EEG-Cz', 'EEG-C4']
        seg_offset = 3 * fs
        seg_len = 4 * fs

    eeg_indices = [raw.ch_names.index(name) for name in target_eeg_names if name in raw.ch_names]
    data_full = raw.get_data(picks=eeg_indices)  # (ch, samples)

    if any(ch.startswith("STI") for ch in raw.ch_names):
        events = mne.find_events(raw, shortest_event=1, verbose=False)
    else:
        mapping = {k: int(k) for k in ["768", "769", "770", "771", "772"]}
        events, _ = mne.events_from_annotations(raw, event_id=mapping, verbose=False)
    pos = events[events[:, 2] == 768][:, 0]
    n_trial = len(pos)
    if n_trial == 0:
        raise RuntimeError(f"No trial events (768) found in {data_path}")

    if os.path.exists(label_path):
        lab = sio.loadmat(label_path)
        classlabel = lab.get("classlabel", np.zeros(n_trial)).squeeze()
    else:
        classlabel = np.zeros(n_trial)
    if len(classlabel) != n_trial:
        min_len = min(len(classlabel), n_trial)
        classlabel = classlabel[:min_len]
        pos = pos[:min_len]
        n_trial = min_len

    data = np.zeros((n_trial, len(target_eeg_names), seg_len))
    for k, p in enumerate(pos):
        start = p + seg_offset
        end = start + seg_len
        if end > data_full.shape[1]:
            raise ValueError(f"trial {k}: {end} > file len {data_full.shape[1]}")
        data[k] = data_full[:, start:end]

    data = np.nan_to_num(data)

    b, a = butter(4, [8 / (fs / 2), 32 / (fs / 2)], btype='bandpass')
    for k in range(n_trial):
        data[k] = filtfilt(b, a, data[k], axis=-1)

    return data, classlabel

def convert_subject(subject_index, gdf_dir='./gdf', label_dir='./true_labels', out_dir='./mat', dataset="2a"):
    os.makedirs(out_dir, exist_ok=True)
    sess_list = ["T", "E"]
    for sess in sess_list:
        gdf_path = os.path.join(gdf_dir, f"A0{subject_index}{sess}.gdf")
        lab_path = os.path.join(label_dir, f"A0{subject_index}{sess}.mat")
        print(f"\n[Check] {gdf_path} exists: {os.path.exists(gdf_path)}")
        print(f"[Check] {lab_path} exists: {os.path.exists(lab_path)}")
        data, label = process_session(gdf_path, lab_path, dataset=dataset)
        out_path = os.path.join(out_dir, f"A0{subject_index}{sess}.mat")
        sio.savemat(out_path, {"data": data, "label": label})
        print(f"Saved {out_path}  {data.shape}")


if __name__ == "__main__":
    for sub in range(1, 10):
        try:
            convert_subject(sub, dataset="2a")
        except Exception as e:
            print(f"[Subject {sub}] Error → {e}")



