# Modified for the language-grounded-ft research workspace; see repository history.
from workspace_paths import resolve_recorded_path, workspace_path


import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange', 'force'
        ]


DATA_DIR = str(workspace_path("Genesis/data/train_04272026"))

PURE_FORCE_COLS = [
    "left_fx",
    "left_fy",
    "left_fz",
    "left_tx",
    "left_ty",
    "left_tz",
    "right_fx",
    "right_fy",
    "right_fz",
    "right_tx",
    "right_ty",
    "right_tz"
]


def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

    if params.dset == 'ettm1':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'ettm2':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'etth1':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'etth2':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    

    elif params.dset == 'electricity':
        root_path = '/data/datasets/public/electricity/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'electricity.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'traffic':
        root_path = '/data/datasets/public/traffic/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'traffic.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    
    elif params.dset == 'weather':
        root_path = '/data/datasets/public/weather/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'illness':
        root_path = '/data/datasets/public/illness/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'national_illness.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'exchange':
        root_path = '/data/datasets/public/exchange_rate/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'exchange_rate.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    
    elif params.dset == 'force':
    # ForceDataset is user-customized: designed to load all files into memory.
    # context_points = sequence length mapped to data_len.
    # target_points is not used so set to 0.
        from torch.utils.data import Dataset
        import os, pandas as pd, torch
        import torch

        def add_noise_adaptive(data, noise_factor=0.05):
            """Add adaptive Gaussian noise per channel using its standard deviation.

            Args:
                data (torch.Tensor): Time-series data (shape: [time_steps, channels] e.g. [80, 12])
                noise_factor (float): Noise ratio relative to each channel's std
            """
            # 1. Compute std per channel along the time dimension (dim=0 keeps time)
            std_per_channel = torch.std(data, dim=0, keepdim=True)
            # 2. Generate noise scaled by per-channel std
            noise = torch.randn_like(data) * std_per_channel * noise_factor
            
            return data + noise

        def scale_data(data, scale_range=(0.8, 1.2)):
            """Scale the amplitude of the time-series data using a random factor."""
            scaler = torch.rand(1, data.shape[1], device=data.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
            return data * scaler
        
        def do_nothing(data):
            return data

        class ForceDataset(Dataset):
            def __init__(self, split: str, data_len: int = params.context_points, use_cols: list = PURE_FORCE_COLS):
                super().__init__()
                self.data_len = data_len
                self.use_cols = use_cols
                self.augmentations = [add_noise_adaptive, scale_data, do_nothing]
                # Switch directories per split (val uses EVAL_DATA_DIR, test is empty dataset)
                base_dir = DATA_DIR

                index_csv = os.path.join(base_dir, 'train_04272026_thin_17pct.csv')

                df_index = pd.read_csv(index_csv)
                emb_index = "emb_index"
                required_cols = {'csv_path', 'start', emb_index}
                missing = required_cols - set(df_index.columns)
                if missing:
                    raise ValueError(f'Missing columns {missing} in {index_csv}')

                self.annotations_emb = []
                self.force_segments = []
                unique_csv_paths = df_index['csv_path'].unique()

                def resolve_csv_path(path_value: str) -> str:
                    """Support both absolute and relative csv_path entries."""
                    if not isinstance(path_value, str):
                        return None
                    candidates = []
                    # absolute path as-is
                    candidates.append(str(resolve_recorded_path(path_value)))
                    # relative to base_dir
                    candidates.append(os.path.join(base_dir, path_value))
                    # relative to base_dir/csv (current dataset layout)
                    candidates.append(os.path.join(base_dir, "csv", path_value))
                    for cand in candidates:
                        if os.path.exists(cand):
                            return cand
                    return None

                # Cache CSV contents
                data_cache = {}
                for path in unique_csv_paths:
                    resolved = resolve_csv_path(path)
                    if resolved is None:
                        continue
                    arr = pd.read_csv(resolved, usecols=self.use_cols).values.astype('float32')
                    data_cache[path] = arr

                debug_counter = 0
                for _, row in df_index.iterrows():
                    csv_path = row['csv_path']
                    if csv_path not in data_cache:
                        continue
                    start_id = int(row['start'])
                    # tmp = "bert_emb" if split == "train" else "bert_emb/label"
                    tmp = "bert_emb"
                    emb_path = os.path.join(base_dir, tmp, f"{row[emb_index]}.pt")
                    if not os.path.exists(emb_path):
                        continue
                    force_segment = data_cache[csv_path][start_id:start_id + self.data_len, :]
                    if force_segment.shape[0] != self.data_len:
                        continue
                    # quick sanity: check non-finites
                    if not np.isfinite(force_segment).all():
                        # count and basic stats
                        nonfinite = np.logical_not(np.isfinite(force_segment))
                        nf_total = int(nonfinite.sum())
                        # show a brief sample once
                        if debug_counter < 5:
                            print(f"[ForceDataset:{split}] non-finite in segment csv={csv_path} start={start_id} count={nf_total}")
                            debug_counter += 1
                        # skip non-finite segments
                        continue
                    try:
                        text_emb = torch.load(emb_path, map_location='cpu')
                    except Exception:
                        continue
                    self.force_segments.append(force_segment)
                    self.annotations_emb.append(text_emb)

            def __len__(self):
                return len(self.force_segments)

            def _apply_augmentations(self, force_tensor):
                """Apply a random sequence of augmentations to the tensor."""
                # Apply between 1 and len(augmentations) random augmentations
                num_augs_to_apply = torch.randint(1, len(self.augmentations) + 1, (1,)).item()
                
                # Select random augmentation functions
                augs = torch.randperm(len(self.augmentations))[:num_augs_to_apply]
                
                augmented_tensor = force_tensor.clone()
                for i in augs:
                    augmented_tensor = self.augmentations[i](augmented_tensor)
                    
                return augmented_tensor

            def __getitem__(self, idx):
                force_array = self.force_segments[idx]          # [seq_len, n_vars]
                force_tensor = torch.from_numpy(force_array)    # [seq_len, n_vars]
                # force_tensor = self._apply_augmentations(force_tensor)
                annotation_emb = self.annotations_emb[idx]
                # PatchTST pipeline expects (x,y); use annotation_emb as y.
                return force_tensor, annotation_emb

        dls = DataLoaders(
                datasetCls=ForceDataset,
                dataset_kwargs={
                    # 分割により今後拡張可
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    # dataset is assume to have dimension len x nvars
    if dls.train is None:
        train_len = 0
        valid_len = 0 if dls.valid is None else len(dls.valid.dataset)
        test_len = 0 if dls.test is None else len(dls.test.dataset)
        raise RuntimeError(
            f"No training samples were loaded for dset={params.dset}. "
            f"split sizes train={train_len}, val={valid_len}, test={test_len}. "
            f"Please check DATA_DIR/index csv_path resolution and emb files."
        )
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls



if __name__ == "__main__":
    class Params:
        dset= 'etth2'
        context_points= 384
        target_points= 96
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='M'
    params = Params 
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
    breakpoint()
