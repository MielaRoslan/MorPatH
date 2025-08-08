import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
import os
import pickle
from lifelines import CoxPHFitter

def FindOutIndex(data):
    out_idx = []
    target = data.values.tolist()
    for i in range(data.shape[0]):
        try:
            np.array(target[i], dtype='int')
        except ValueError:
            out_idx.append(i)
    return out_idx

def preprocess_clinical_data(clinical_path):
    data_clinical = pd.read_csv(clinical_path, header=None)
    sample_ids = data_clinical.iloc[:, 0].tolist()  # Extract the sample IDs from column 0
    
    target_data = data_clinical[[10, 11]]
    out_idx = FindOutIndex(target_data)
    clin_variables = data_clinical[[1, 2, 3, 4, 5, 6, 7, 8, 9]]
    idx = clin_variables[clin_variables[[6, 7, 8, 9]].isnull().any(axis=1)].index
    g = list(idx)
    data_clinical.drop(index=out_idx + g, inplace=True)
    
    # Use columns 1-5 for categorical if that’s what you need
    clin_data_categorical = data_clinical[[1, 2, 3, 4, 5]]
    clin_data_continuous = data_clinical[[6, 7, 8, 9]]
    return clin_data_categorical, clin_data_continuous, target_data, out_idx + g, sample_ids


# Function to normalize CNV data
def normalize_cnv_data(cnv_data):
    mean = np.mean(cnv_data, axis=0)
    std = np.std(cnv_data, axis=0)
    std[std == 0] = 1  # Prevent division by zero for constant features
    normalized_data = (cnv_data - mean) / std
    return normalized_data

def normalize_wsi_features(features):
    """
    Standardizes each WSI tensor individually (mean 0, std 1).
    Handles variable-sized feature tensors by normalizing per sample.
    """
    normalized_features = []
    for feature in features:
        mean = feature.mean(dim=0, keepdim=True)
        std = feature.std(dim=0, keepdim=True)
        std[std == 0] = 1  # Prevent division by zero
        normalized_feature = (feature - mean) / std
        normalized_features.append(normalized_feature)
    return normalized_features  # Return a list of normalized tensors


def _load_wsi_feature_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pt", ".pth"]:
        return torch.load(path)
    elif ext in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            return pkl.load(f)
    else:
        raise ValueError(f"Unsupported WSI feature extension: {ext}")

class TCGA_Dataset(Dataset):
    def __init__(self, modalities, data_path, histogram_mode='all'):
        super(TCGA_Dataset, self).__init__()
        self.data_modalities = modalities
        self.histogram_mode = histogram_mode  # Add this line
        # self.sample_names = []  # To store filenames or unique sample IDs
        clin_data_categorical, clin_data_continuous, target_data, remove_idx, sample_ids = preprocess_clinical_data(data_path['clinical'])
        self.target = target_data.values.tolist()
        # Use the extracted sample IDs instead of the DataFrame index.
        self.sample_names = sample_ids
        
        if 'clinical' in modalities:
            self.clin_cat = clin_data_categorical.values.tolist()
            self.clin_cont = clin_data_continuous.values.tolist()
        
        if 'CNV' in modalities:
            data_cnv = pd.read_csv(data_path['CNV'], header=None)
            data_cnv.drop(index=remove_idx, inplace=True)
            cnv_values = data_cnv.values
            normalized_cnv = normalize_cnv_data(cnv_values)
            self.data_cnv = normalized_cnv.tolist()
        
        if 'WSI' in modalities:
            # Note: data_path must have keys 'WSI' (features) and 'WSI_coords' (coordinates)
            self.handle_wsi(data_path['WSI'], data_path['WSI_coords'])
        
        if 'Histogram' in modalities:
            self.handle_histogram(data_path['Histogram'])
        
    def handle_wsi(self, path_wsi_features, path_wsi_coords=None):
        # 0) 디렉토리 체크
        if not path_wsi_features or not os.path.isdir(path_wsi_features):
            print(f"[WSI] feature dir not found: {path_wsi_features}")
            self.data_wsi = []
            self.data_wsi_coords = []
            return

        # 1) 확장자 후보 전부 탐색
        patterns = ["*.pt", "*.pth", "*.pickle", "*.pkl"]
        feat_files = []
        for pat in patterns:
            feat_files.extend(glob(os.path.join(path_wsi_features, pat)))
        feat_files = sorted(set(feat_files))

        if len(feat_files) == 0:
            print(f"[WSI] no feature files found in {path_wsi_features}. tried patterns={patterns}")
            self.data_wsi = []
            self.data_wsi_coords = []
            return

        # 2) 좌표는 있으면 로드, 없으면 None
        coord_map = {}
        if path_wsi_coords and os.path.isdir(path_wsi_coords):
            coord_candidates = []
            for pat in ["*.npy", "*.pickle", "*.pkl"]:
                coord_candidates.extend(glob(os.path.join(path_wsi_coords, pat)))
            coord_candidates = sorted(set(coord_candidates))
            def stem(p): return os.path.splitext(os.path.basename(p))[0]
            coord_map = {stem(c): c for c in coord_candidates}

        feats, coords_all = [], []
        def stem(p): return os.path.splitext(os.path.basename(p))[0]

        print(f"[WSI] found {len(feat_files)} feature files (e.g., {feat_files[:3]})")
        for ff in feat_files:
            sid = stem(ff)
            feat = _load_wsi_feature_file(ff)
            # 텐서화
            if isinstance(feat, np.ndarray):
                feat = torch.tensor(feat, dtype=torch.float32)
            elif torch.is_tensor(feat):
                feat = feat.to(torch.float32)
            else:
                feat = torch.tensor(np.array(feat), dtype=torch.float32)
            feats.append(feat)

            # 좌표
            cf = coord_map.get(sid)
            if cf:
                ext = os.path.splitext(cf)[1].lower()
                if ext == ".npy":
                    c = np.load(cf)
                else:
                    with open(cf, "rb") as f:
                        c = pkl.load(f)
                coords_all.append(np.array(c))
            else:
                coords_all.append(None)

        self.data_wsi = feats
        self.data_wsi_coords = coords_all

        if len(self.data_wsi) != len(self.target):
            print(f"Warning: Mismatch in WSI features count: {len(self.data_wsi)} and target count: {len(self.target)}")

    
    def handle_histogram(self, path_histogram):
        file_paths = sorted(glob(os.path.join(path_histogram, '*.pt')))
        features = []

        # Define ranked feature row indices
        ranked_indices = [7, 3, 2, 1, 6, 0, 4, 8, 5]
        mode = getattr(self, 'histogram_mode', 'all')  # default to all if not passed

        if mode == 'top7':
            selected_rows = ranked_indices[:7]  # top 7
        elif mode == 'top5':
            selected_rows = ranked_indices[:5]  # top 5
        else:
            selected_rows = ranked_indices  # all 9

        print(f"Loading histogram data from {path_histogram} using mode: {mode}...")

        for file_path in file_paths:
            try:
                feature = torch.load(file_path)  # Expect shape (9, 10)

                if torch.isnan(feature).any():
                    print(f"NaNs detected in {file_path}.")
                    nan_indices = torch.isnan(feature).nonzero(as_tuple=True)
                    print(f"NaN indices in {file_path}: {nan_indices}")

                # Select top features based on rows and flatten to 1D vector
                feature = feature[selected_rows, :].flatten()  # shape (len(selected_rows) × 10,)
                features.append(feature)

                # Append sample name for tracking
                self.sample_names.append(os.path.basename(file_path))

            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

        self.data_histogram = features
        print(f"Loaded {len(self.data_histogram)} histogram features with shape: {self.data_histogram[0].shape} each.")
        
        if len(self.data_histogram) != len(self.target):
            print(f"Warning: Mismatch in Histogram features count: {len(self.data_histogram)} and target count: {len(self.target)}")

    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, index):
        data = {}
        data_label = {}
        target_y = np.array(self.target[index], dtype='int')
        data_label['label'] = torch.from_numpy(target_y).type(torch.LongTensor)
        
        if 'WSI' in self.data_modalities:
            if index >= len(self.data_wsi):
                raise IndexError(f"WSI index out of range. Index: {index}, Total WSI features: {len(self.data_wsi)}")
            wsi = self.data_wsi[index]
            coords = self.data_wsi_coords[index]
            data['WSI'] = wsi  # tensor containing features
            data['WSI_coords'] = torch.from_numpy(coords).type(torch.float32)
            # Return the sample id (as a string) in the label dictionary:
            data_label['WSI_id'] = self.sample_names[index]
        
        if 'Histogram' in self.data_modalities:
            if index >= len(self.data_histogram):
                raise IndexError(f"Histogram index out of range. Index: {index}, Total Histogram features: {len(self.data_histogram)}")
            data['Histogram'] = self.data_histogram[index]
        
        return data, data_label