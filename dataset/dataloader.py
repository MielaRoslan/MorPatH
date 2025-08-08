# dataset/dataloader.py
from __future__ import annotations
import os
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data.dataset import Dataset

import importlib


def _resolve_dataset_class(dataset_name: str):
    """
    dataset_name에 따라 모듈/클래스를 동적으로 선택.
    - 'NCC' -> dataset.NCC:NCC_Dataset
    - 'TCGA_*' -> dataset.TCGA:TCGA_Dataset
    """
    name = (dataset_name or "").strip()
    if not name:
        raise ValueError("[dataloader] cfg.data.dataset_name 이 비었습니다.")

    if name == "NCC":
        module_name, class_name = "dataset.NCC", "NCC_Dataset"
    elif name.startswith("TCGA"):
        module_name, class_name = "dataset.TCGA", "TCGA_Dataset"
    else:
        raise ValueError(f"[dataloader] 지원하지 않는 dataset_name: {name} "
                         f"(허용 예: 'NCC', 'TCGA_LUAD', 'TCGA_BRCA', ...)")

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"[dataloader] 모듈을 찾을 수 없습니다: {module_name}. "
            f"파일 경로를 확인하세요 (예: dataset/NCC.py, dataset/TCGA.py)."
        ) from e

    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"[dataloader] {module_name} 안에 클래스 {class_name} 이 없습니다."
        ) from e

    return cls

# ======================
# Utilities
# ======================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _check_required_paths(modalities: List[str], data_paths: Dict[str, str], dataset_name: str) -> None:
    """
    필요한 modality 경로가 configs에서 모두 넘어왔는지 확인.
    - NCC: WSI_coords 필수
    - TCGA_*: WSI_coords 선택 (없어도 통과)
    """
    required = []
    if "clinical" in modalities:
        required.append("clinical")
    if "WSI" in modalities:
        required.append("WSI")
        if dataset_name == "NCC":
            required.append("WSI_coords")  # NCC만 coords 필수
    if "Histogram" in modalities:
        required.append("Histogram")

    missing = [k for k in required if k not in data_paths or not data_paths[k]]
    if missing:
        raise ValueError(f"[dataloader] 필요한 데이터 경로가 빠졌습니다: {missing}. configs에 각 키를 추가하세요.")


def _build_data_paths_from_cfg(cfg) -> Dict[str, str]:
    """
    cfg.data.* 로부터 data_paths 딕셔너리를 구성.
    - cfg는 argparse.Namespace 또는 OmegaConf/DictConfig, dict 모두 허용.
    기대 키:
      cfg.data.clinical_csv
      cfg.data.wsi_dir
      cfg.data.wsi_coords_dir
      cfg.data.histogram_dir
    """
    get = _cfg_getter(cfg)

    paths = {}
    if get("data.clinical_csv"):
        paths["clinical"] = get("data.clinical_csv")
    if get("data.wsi_dir"):
        paths["WSI"] = get("data.wsi_dir")
    if get("data.wsi_coords_dir"):
        paths["WSI_coords"] = get("data.wsi_coords_dir")
    if get("data.histogram_dir"):
        paths["Histogram"] = get("data.histogram_dir")
    return paths


def _cfg_getter(cfg):
    """
    cfg를 dict처럼 '.' 경로로 조회하는 getter 팩토리.
    argparse.Namespace / dict / OmegaConf(DictConfig) 호환.
    """
    def _get(path: str, default=None):
        cur = cfg
        for key in path.split("."):
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(key, default if key == path.split(".")[-1] else None)
            else:
                # Namespace 또는 omegaconf
                if hasattr(cur, key):
                    cur = getattr(cur, key)
                else:
                    try:
                        # OmegaConf(DictConfig) 지원
                        from omegaconf import OmegaConf  # optional
                        if OmegaConf.is_config(cur):
                            return OmegaConf.select(cur, path, default=default)
                    except Exception:
                        return default
                    return default
        return cur if cur is not None else default
    return _get


# ======================
# Collate (옵션)
# ======================

def wsi_list_collate(batch):
    """
    (data_dict, label_dict)의 리스트를 받아,
    - 'WSI'는 패치 수가 가변이라 list로 유지
    - 나머지는 shape가 같으면 stack
    """
    collated_data = {}
    collated_label = {}

    for data, label in batch:
        for k, v in data.items():
            collated_data.setdefault(k, []).append(v)
        for k, v in label.items():
            collated_label.setdefault(k, []).append(v)

    # 데이터: WSI만 리스트 유지, 나머지는 stack
    for k, vals in list(collated_data.items()):
        if k == "WSI":
            continue
        try:
            collated_data[k] = torch.stack(vals, dim=0)
        except Exception:
            # shape가 다르면 그대로 리스트 유지
            collated_data[k] = vals

    # 라벨은 일반적으로 동일 shape 가정
    for k, vals in list(collated_label.items()):
        collated_label[k] = torch.stack(vals, dim=0)

    return collated_data, collated_label


# ======================
# Sampler helpers
# ======================

def _extract_labels(dataset: Dataset, indices: List[int]) -> np.ndarray:
    """
    dataset[i][1]['label'] 에서 첫 원소를 분류 라벨로 사용한다는 기존 코드를 존중.
    (필요 시 여기만 바꾸면 전체 가중치 샘플링에 반영됨)
    """
    labels = []
    for i in indices:
        _, lab = dataset[i]
        # lab['label']는 LongTensor([...]) 형태라고 가정
        val = int(lab["label"][0].item()) if torch.is_tensor(lab["label"]) else int(lab["label"][0])
        labels.append(val)
    return np.asarray(labels)


def _make_weighted_sampler(dataset: Dataset, train_indices: List[int]) -> WeightedRandomSampler:
    labels = _extract_labels(dataset, train_indices)
    class_counts = np.bincount(labels)
    # 0 count 방지
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)


# ======================
# Fold split
# ======================

def make_kfold_indices(n_samples: int, n_folds: int, fold_idx: int, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    단순 K-Fold 분할(셔플 고정). 필요하면 stratify는 상위에서 라벨로 직접 구현 가능.
    """
    if not (0 <= fold_idx < n_folds):
        raise ValueError(f"fold_idx는 [0, {n_folds-1}] 범위여야 합니다. 현재: {fold_idx}")

    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_sizes = [(n_samples // n_folds) + (1 if i < (n_samples % n_folds) else 0) for i in range(n_folds)]
    splits = []
    start = 0
    for fs in fold_sizes:
        splits.append(indices[start:start+fs])
        start += fs

    val_idx = splits[fold_idx].tolist()
    train_idx = np.concatenate([splits[i] for i in range(n_folds) if i != fold_idx]).tolist()
    return train_idx, val_idx


# ======================
# Public API
# ======================

def build_dataset_from_cfg(cfg):
    """
    configs를 받아 Dataset 인스턴스를 생성.
    """
    get = _cfg_getter(cfg)

    dataset_name = get("data.dataset_name")  # <= 추가
    if not dataset_name:
        raise ValueError("[dataloader] cfg.data.dataset_name 이 필요합니다. (예: NCC, TCGA_LUAD)")

    modalities = list(get("data.modalities", []))
    if not modalities:
        raise ValueError("[dataloader] cfg.data.modalities가 비어있습니다. 예: ['clinical','WSI','Histogram']")

    data_paths = _build_data_paths_from_cfg(cfg)
    _check_required_paths(modalities, data_paths, dataset_name)

    histogram_mode = get("data.histogram_mode", "all")

    # 동적 클래스 로딩
    DatasetCls = _resolve_dataset_class(dataset_name)

    # 공통 시그니처 시도
    kwargs = dict(
        modalities=modalities,
        data_path=data_paths,
        histogram_mode=histogram_mode,
    )
    # 혹시 데이터셋 클래스가 dataset_name을 참고하고 싶을 수 있으니 안전하게 전달
    kwargs["dataset_name"] = dataset_name

    try:
        dataset = DatasetCls(**kwargs)
    except TypeError:
        # 구식 시그니처 대응 (dataset_name 안 받는 경우)
        kwargs.pop("dataset_name", None)
        dataset = DatasetCls(**kwargs)

    return dataset


def build_dataloaders_from_cfg(
    cfg,
    dataset: Optional[Dataset] = None,
    fold_idx: Optional[int] = None,
) -> Dict[str, DataLoader]:

    # configs 기반으로 train/val DataLoader를 생성.
    # 기대 cfg 키(예시):
    #   cfg.train.batch_size
    #   cfg.train.num_workers
    #   cfg.train.seed
    #   cfg.train.n_folds
    #   cfg.train.use_weighted_sampling (bool)
    #   cfg.data.use_wsi_list_collate (bool)  # True면 wsi_list_collate 사용, 아니면 None

    # - fold_idx가 None이면 cfg.train.fold_idx를 사용.
    # - dataset을 외부에서 주면 재사용하고, 없으면 내부에서 build_dataset_from_cfg로 생성.

    get = _cfg_getter(cfg)

    if dataset is None:
        dataset = build_dataset_from_cfg(cfg)

    n_samples = len(dataset)
    n_folds = int(get("train.n_folds", 5))
    seed = int(get("train.seed", 42))
    fold = int(get("train.fold_idx", 0) if fold_idx is None else fold_idx)

    set_seed(seed)
    train_idx, val_idx = make_kfold_indices(n_samples, n_folds, fold, seed)

    batch_size = int(get("train.batch_size", 8))
    num_workers = int(get("train.num_workers", 4))
    pin_memory = bool(get("train.pin_memory", True))
    persistent_workers = bool(get("train.persistent_workers", True))
    use_weighted = bool(get("train.use_weighted_sampling", False))

    # collate 선택
    use_wsi_collate = bool(get("data.use_wsi_list_collate", False))
    collate_fn = wsi_list_collate if use_wsi_collate else None

    # Sampler
    if use_weighted:
        train_sampler = _make_weighted_sampler(dataset, train_idx)
        # WeightedRandomSampler는 인덱스를 직접 주입하지 않으므로, Subset 느낌을 맞추려면
        # worker에서 인덱스 매핑이 필요하지만 여기선 간단히 train_idx로 subset view를 만들지 않고
        # 샘플링 대상 풀을 train subset으로 제한하기 위해 작은 wrapper를 쓰는 게 일반적임.
        # 간단히 구현할 수 있도록 Subset 스타일의 DatasetView를 아래에서 사용.
        subset_train = _DatasetSubset(dataset, train_idx)
        train_loader = DataLoader(
            subset_train,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=collate_fn,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=collate_fn,
            drop_last=False,
        )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print(f"[dataloader] n_samples={n_samples} | fold {fold+1}/{n_folds} "
          f"| train={len(train_idx)} | val={len(val_idx)} | batch_size={batch_size}")
    return {"train": train_loader, "val": val_loader}


class _DatasetSubset(Dataset):     #WeightedRandomSampler를 train subset에 한정해서 쓰고 싶을 때 쓰는 간단한 뷰.

    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]




# ======================
# CLI: quick data-loading check
# ======================
def _shape_of(x):
    if torch.is_tensor(x):
        return tuple(x.shape)
    try:
        return (len(x),)  # list 등
    except Exception:
        return type(x).__name__

def _summarize_batch(batch, tag: str = "train"):
    data, label = batch
    print(f"\n[{tag}] batch summary ------------------------------")
    # data
    print("[data] keys:", list(data.keys()))
    for k, v in data.items():
        if k == "WSI":
            if isinstance(v, list):
                lens = [len(t) if torch.is_tensor(t) else (len(t) if hasattr(t, "__len__") else -1) for t in v]
                print(f"  - {k}: list(len) = {lens[:8]}{'...' if len(lens) > 8 else ''}")
                if len(v) > 0 and torch.is_tensor(v[0]):
                    print(f"    sample[0] tensor shape: {tuple(v[0].shape)}")
            else:
                print(f"  - {k}: {_shape_of(v)}")
        else:
            print(f"  - {k}: {_shape_of(v)}")

    # label
    print("[label] keys:", list(label.keys()))
    for k, v in label.items():
        if torch.is_tensor(v):
            print(f"  - {k}: tensor{tuple(v.shape)} | dtype={v.dtype}")
        else:
            print(f"  - {k}: {_shape_of(v)}")

    if "label" in label and torch.is_tensor(label["label"]):
        lbl = label["label"].detach().cpu().numpy()
        uniq, cnt = np.unique(lbl, return_counts=True)
        print(f"[label] unique: {uniq.tolist()}, counts: {cnt.tolist()}")

        # progression free month 출력
        if lbl.ndim == 2 and lbl.shape[1] >= 2:
            pfs_months = lbl[:, 1]
            print(f"[label] progression_free_month values: {pfs_months.tolist()}")

        # 원래 label 값 전체 출력
        print(f"[label] full values:\n{lbl}")

def main():
    """
    configs/configs.py의 CLI 인자를 그대로 받아
    - dataset 인스턴스 생성
    - fold split 후 dataloaders 생성
    - train/val 각각 1배치씩 뽑아 요약 정보 출력
    """
    try:
        from configs.configs import build_cfg
    except Exception as e:
        raise RuntimeError(
            "[dataloader] configs.configs.build_cfg 를 import할 수 없습니다. "
            "configs/configs.py가 존재하는지, PYTHONPATH가 올바른지 확인하세요."
        ) from e

    cfg = build_cfg()
    print("\n[dataloader] building dataset...")
    dataset = build_dataset_from_cfg(cfg)
    print(f"[dataloader] dataset built: type={type(dataset).__name__}, len={len(dataset)}")

    print("[dataloader] building dataloaders...")
    loaders = build_dataloaders_from_cfg(cfg, dataset=dataset)
    tr_loader, va_loader = loaders["train"], loaders["val"]
    print(f"[dataloader] train batches ≈ {len(tr_loader)}, val batches ≈ {len(va_loader)}")

    # 한 배치만 확인
    got = False
    for batch in tr_loader:
        _summarize_batch(batch, tag="train")
        got = True
        break
    if not got:
        print("[train] 배치를 하나도 만들지 못했습니다. 인덱스 분할/데이터를 확인하세요.")

    got = False
    for batch in va_loader:
        _summarize_batch(batch, tag="val")
        got = True
        break
    if not got:
        print("[val] 배치를 하나도 만들지 못했습니다. 인덱스 분할/데이터를 확인하세요.")

if __name__ == "__main__":
    main()