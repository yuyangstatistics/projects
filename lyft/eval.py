from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

import l5kit
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

import matplotlib.pyplot as plt

import os
import random
import time
from json import dumps
import warnings
warnings.filterwarnings("ignore")

import utils
from models import LyftEffnet, LyftEffnetb7, LyftDensenet
from utils import forward

### =============== Configurations ============= ###
cfg = {
    'format_version': 4,
    'data_path': "input/lyft-motion-prediction-autonomous-vehicles",
    'save_dir': "./save/",
    'name': 'effnetb7',
    'gpu_ids': [0],
    'seed': 2048,    
    'load_path': "./save/train/effnetb7-07/best.pth.tar", 
    'model_params': {
        # 'model_architecture': 'resnet50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "effnetb4",
        'lr': 1e-4,
        'train': False,
        'predict': True,
    },

    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },

    'train_data_loader': {
        'key': 'scenes/train_full.zarr',
        'batch_size': 64, # 16
        'shuffle': True,
        'num_workers': 16 # 4
    },
    
    'val_data_loader': {
        'dir': 'input/lyft-motion-prediction-autonomous-vehicles/scenes/validate_chopped_100',
        'key': 'scenes/validate_chopped_100/validate.zarr',
        'batch_size': 64,
        'shuffle': False,
        'num_workers': 16 # 4
    },

    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 64,
        'shuffle': False,
        'num_workers': 16 # 4
    },

    'train_params': {
        'max_num_steps': 12000000,
        'eval_steps': 160000,
        'max_grad_norm': 3.0,
    }
}


def main(cfg):

    # set logger, tensorboard, and devices
    cfg['save_dir'] = utils.get_save_dir(cfg['save_dir'], cfg['name'], training=False)
    log = utils.get_logger(cfg["save_dir"], cfg["name"])
    device = utils.get_devices(cfg["gpu_ids"])
    cfg["test_data_loader"]["batch_size"] *= max(1, len(cfg["gpu_ids"]))
    
    log.info(f"Cfg: {dumps(cfg, indent = 4, sort_keys = True)}")

    # get model
    log.info("Building model...")
    model = utils.init_model(cfg)
    model = nn.DataParallel(model, device_ids=cfg["gpu_ids"])
    log.info(f"Loading checkpoint from {cfg['load_path']}...")
    model = utils.load_model(model, cfg["load_path"], cfg["gpu_ids"], return_step=False)
    model = model.to(device)
    model.eval()

    # get dataloader
    DIR_INPUT = cfg["data_path"]
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager(None)

    log.info("Building validation dataset...")
    val_cfg = cfg["val_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()
    val_mask = np.load(f"{val_cfg['dir']}/mask.npz")["arr_0"]
    val_dataset = AgentDataset(cfg, val_zarr, rasterizer, agents_mask=val_mask)
    val_dataloader = DataLoader(val_dataset, shuffle=val_cfg["shuffle"], batch_size=val_cfg["batch_size"],
                                num_workers=val_cfg["num_workers"], pin_memory=True)
    log.info(str(val_dataset))

    # Predict
    log.info("Evaluating...")
    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    confidences_list = []
    agent_ids = []

    progress_bar = tqdm(val_dataloader)

    with torch.no_grad():
        for data in progress_bar:

            _, preds, confidences, _ = forward(data, model, device)

            #fix for the new environment
            preds = preds.cpu().numpy()
            world_from_agents = data["world_from_agent"].numpy()
            centroids = data["centroid"].numpy()
                
            # convert into world coordinates and compute offsets
            for idx in range(len(preds)):
                for mode in range(3):
                    preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]
            
            future_coords_offsets_pd.append(preds.copy())
            confidences_list.append(confidences.cpu().numpy().copy())  
            timestamps.append(data["timestamp"].numpy().copy())  
            agent_ids.append(data["track_id"].numpy().copy())  


    # create submission to submit to Kaggle
    pred_path = os.path.join(cfg['save_dir'], "val-pred.csv")
    log.info(f"Writing prediction for the validation set to {pred_path}.")
    write_pred_csv(pred_path,
            timestamps=np.concatenate(timestamps),  
            track_ids=np.concatenate(agent_ids),    
            coords=np.concatenate(future_coords_offsets_pd),
            confs = np.concatenate(confidences_list)
            )

if __name__ == '__main__':
    main(cfg)