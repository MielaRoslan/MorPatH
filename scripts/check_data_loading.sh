set -e

PY=python  # 또는 python3

# === NCC 예시 ===
$PY -m dataset.dataloader \
  --dataset_name TCGA_LUAD \
  --foundation_model uni \
  --modalities clinical,WSI,Histogram \
#   --clinical_csv /mnt/aix22307/wsi-project/mels/MultiSurv/preprocess/preprocessed_data/PFS_NCC.csv \
#   --wsi_dir /mnt/aix22307/wsi-project/extracted_features/Features_PDAC_TCGA/NCC_numpy_1.24.3/Feature_DINO \
#   --wsi_coords_dir /mnt/aix22307/wsi-project/extracted_features/Features_PDAC_TCGA/NCC_numpy_1.24.3/Coord_DINO \
#   --histogram_dir /mnt/aix22307/wsi-project/histogram/outputnccnew/histograms_140within \
  --histogram_mode all \
  --use_wsi_list_collate false \
  --n_folds 5 \
  --fold_idx 0 \
  --batch_size 1 \
  --num_workers 2 \
  --pin_memory true \
  --persistent_workers true \
  --seed 42 \
  --use_weighted_sampling false \
  --require_wsi_coords false \


