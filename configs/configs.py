from __future__ import annotations
import argparse
from types import SimpleNamespace

def _str2bool(v: str) -> bool:
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","y","1"): return True
    if v in ("no","false","f","n","0"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def _str2list(v: str):
    if v is None or v == "": return []
    return [x.strip() for x in v.split(",") if x.strip()]

def _norm_backbone(b: str):
    if not b: return ("uni", "UNI")  # default
    blo = b.strip().lower()
    mapping = {"uni":"UNI", "conch":"CONCH", "dino":"DINO", "virchow":"VIRCHOW"}
    bup = mapping.get(blo, b.upper())
    return (blo, bup)

def _default_feat_dim_for_backbone(foundation_model: str) -> int | None:
    
    """
    Safe defaults. Override with --wsi_feat_dim if different in your extractors.
    - UNI:     512
    - CONCH:   1024
    - VIRCHOW: 1024
    - DINO:    varies (ViT-S/B/etc.) -> None to force explicit override
    """
    fm = (foundation_model or "").lower()
    if fm == "uni":     return 512
    if fm == "conch":   return 1024
    if fm == "virchow": return 1024
    if fm == "dino":    return None
    
    return None


def _infer_paths_from_dataset_name(dataset_name: str, backbone: str):
    """
    dataset_name이 'NCC'거나 'TCGA_*'일 때, foundation model(uni|conch|dino)에 맞춰
    기본 경로를 구성. 사용자가 CLI로 직접 값 주면 그 값 우선.
    """
    ds = (dataset_name or "").strip()
    blo, bup = _norm_backbone(backbone)

    paths = {}
    if ds == "NCC":
        # NCC: Feature_{BACKBONE_UPPER}, Coord_{BACKBONE_UPPER}
        paths["clinical_csv"]   = "/mnt/aix22307/wsi-project/mels/MultiSurv/preprocess/preprocessed_data/PFS_NCC.csv"
        base = "/mnt/aix22307/wsi-project/extracted_features/Features_PDAC_TCGA/NCC_numpy_1.24.3"
        paths["wsi_dir"]        = f"{base}/Feature_{bup}"
        paths["wsi_coords_dir"] = f"{base}/Coord_{bup}"
        paths["histogram_dir"]  = "/mnt/aix22307/wsi-project/histogram/outputnccnew/histograms_140within"

    elif ds.startswith("TCGA_"):
        cancer = ds.split("_", 1)[1] if "_" in ds else ds.replace("TCGA", "").lstrip("_")
        cancer_up = cancer.upper()
        cancer_lo = cancer.lower()

        paths["clinical_csv"]   = f"/mnt/aix22307/wsi-project/TCGA/clinical/PFS_{cancer_up}.csv"
        # TCGA: .../<CANCER>/<backbone_lower>/pt_files
        paths["wsi_dir"]        = f"/mnt/aix22307/dataset/TCGA/{cancer_up}/{blo}/pt_files"
        # coords 규칙이 있다면 아래 경로를 맞춰 쓰고, 없으면 None로 둠 (require_wsi_coords로 제어)
        paths["wsi_coords_dir"] = f"/mnt/aix22307/dataset/TCGA/{cancer_up}/{blo}/coord_files"
        paths["histogram_dir"]  = f"/mnt/aix22307/wsi-project/histogram/{cancer_lo}_dcm_histogram/all_pt_files"

    return paths

def build_cfg():
    p = argparse.ArgumentParser(description="Training Configs")

    # -------- data --------
    p.add_argument("--dataset_name", type=str, required=True,
                   help="NCC 또는 TCGA_* 형태 (예: NCC, TCGA_LUAD, TCGA_BRCA)")
    p.add_argument("--foundation_model", type=str, default="uni",
                   choices=["uni","conch","dino", "virchow"],
                   help="경로 추론 시 사용할 백본 토큰")
    p.add_argument("--modalities", type=_str2list, default="clinical,WSI,Histogram",
                   help="콤마로 구분. 예: clinical,WSI,Histogram")

    # 자동 추론 가능하지만 사용자 입력이 있으면 그 값을 우선
    p.add_argument("--clinical_csv", type=str, default=None)
    p.add_argument("--wsi_dir", type=str, default=None)
    p.add_argument("--wsi_coords_dir", type=str, default=None)
    p.add_argument("--histogram_dir", type=str, default=None)

    p.add_argument("--histogram_mode", type=str, default="all", choices=["all","top7","top5"])
    p.add_argument("--use_wsi_list_collate", type=_str2bool, default=False)
    p.add_argument("--require_wsi_coords", type=_str2bool, default=True,
                   help="WSI 좌표가 반드시 필요하면 true (NCC/TCGA 공통 스위치)")

    # model knobs
    p.add_argument("--wsi_backbone", type=str, default="abmil",
                   choices=["abmil","deepmisl","transmil"])
    p.add_argument("--fusion", type=str, default="concat",
                   choices=["concat","bilinear","xattn"])

    # let data_loader decide this if None
    p.add_argument(
        "--wsi_feat_dim", type=int, default=None,
        help="If None: use backbone default (UNI=512, CONCH=1024, VIRCHOW=1024). For DINO, you MUST set this."
    )
    p.add_argument("--wsi_out_dim", type=int, default=256)
    p.add_argument("--hist_bins", type=int, default=90)
    p.add_argument("--hist_embed_dim", type=int, default=256)
    p.add_argument("--wsi_attn_type", type=str, default="gated", choices=["gated","dot"])

    # FTT (hist) + fusion hparams
    p.add_argument("--ftt_dim", type=int, default=256)
    p.add_argument("--ftt_depth", type=int, default=4)
    p.add_argument("--ftt_heads", type=int, default=8)
    p.add_argument("--ftt_dim_head", type=int, default=16)
    p.add_argument("--ftt_attn_dropout", type=float, default=0.3)
    p.add_argument("--ftt_ff_dropout", type=float, default=0.5)
    p.add_argument("--fuse_hidden", type=int, default=None)
    p.add_argument("--fuse_rank", type=int, default=128)
    p.add_argument("--fuse_common_dim", type=int, default=256)
    p.add_argument("--fuse_heads", type=int, default=4)
    p.add_argument("--fuse_dropout", type=float, default=0.1)
    p.add_argument("--fuse_act", type=str, default="relu")



    # -------- train --------
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--fold_idx", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=_str2bool, default=True)
    p.add_argument("--persistent_workers", type=_str2bool, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_weighted_sampling", type=_str2bool, default=False)

    args = p.parse_args()
    
    # -------- wsi_feat_dim (from backbone default if not provided) --------
    wsi_feat_dim = args.wsi_feat_dim
    if wsi_feat_dim is None:
        wsi_feat_dim = _default_feat_dim_for_backbone(args.foundation_model)
        if wsi_feat_dim is None:
            raise RuntimeError(
                f"--wsi_feat_dim is required for foundation_model='{args.foundation_model}'. "
                "Pass it explicitly (e.g., 384 or 768 for DINO)."
            )


    # dataset + backbone 기반 자동 경로
    inferred = _infer_paths_from_dataset_name(args.dataset_name, args.foundation_model)

    # 사용자 입력이 있으면 우선
    clinical_csv   = args.clinical_csv   if args.clinical_csv   else inferred.get("clinical_csv")
    wsi_dir        = args.wsi_dir        if args.wsi_dir        else inferred.get("wsi_dir")
    # coords는 None을 명시적으로 줄 수도 있으므로 is not None로 분기
    wsi_coords_dir = args.wsi_coords_dir if args.wsi_coords_dir is not None else inferred.get("wsi_coords_dir")
    histogram_dir  = args.histogram_dir  if args.histogram_dir  else inferred.get("histogram_dir")

    data = SimpleNamespace(
        dataset_name=args.dataset_name,
        foundation_model=args.foundation_model,
        modalities=args.modalities,
        clinical_csv=clinical_csv,
        wsi_dir=wsi_dir,
        wsi_coords_dir=wsi_coords_dir,
        histogram_dir=histogram_dir,
        histogram_mode=args.histogram_mode,
        use_wsi_list_collate=args.use_wsi_list_collate,
        require_wsi_coords=args.require_wsi_coords,
    )

    train = SimpleNamespace(
        n_folds=args.n_folds,
        fold_idx=args.fold_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        seed=args.seed,
        use_weighted_sampling=args.use_weighted_sampling,
    )

    cfg = SimpleNamespace(data=data, train=train)

    # 디버그 출력
    print("[cfg] dataset_name:", cfg.data.dataset_name, "| backbone:", cfg.data.foundation_model)
    print("[cfg] modalities:", cfg.data.modalities)
    print("[cfg] clinical:", cfg.data.clinical_csv)
    print("[cfg] wsi:", cfg.data.wsi_dir)
    print("[cfg] wsi_coords:", cfg.data.wsi_coords_dir)
    print("[cfg] hist:", cfg.data.histogram_dir, "| mode:", cfg.data.histogram_mode)
    print("[cfg] require_wsi_coords:", cfg.data.require_wsi_coords)
    print("[cfg] fold:", cfg.train.fold_idx, "/", cfg.train.n_folds)

    return cfg

if __name__ == "__main__":
    _ = build_cfg()
