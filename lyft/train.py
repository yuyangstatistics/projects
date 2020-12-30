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
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace, average_displacement_error_mean, final_displacement_error_mean
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

import matplotlib.pyplot as plt

import os
import random
import time

import warnings
warnings.filterwarnings("ignore")


from tensorboardX import SummaryWriter
from json import dumps
from adamp import AdamP

import utils
from models import LyftEffnet
from utils import forward

### =============== Configurations ============= ###
cfg = {
    'format_version': 4,
    'data_path': "input/lyft-motion-prediction-autonomous-vehicles",
    'save_dir': "./save/",
    'name': 'effnetb4_300',
    'gpu_ids': [4, 5, 6, 7],
    'seed': 2,    
    'load_path': "", 
    'model_params': {
        # 'model_architecture': 'resnet50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "effnetb4_300",
        'lr': 1e-4,
        'train': True,
        'predict': False,
    },

    'raster_params': {
        'raster_size': [300, 300],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },

    'train_data_loader': {
        'key': 'scenes/train.zarr',
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
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 16 # 4
    },

    'train_params': {
        'max_num_steps': 12000000,
        'eval_steps': 160000,
        'max_grad_norm': 5.0,
    }
}

def main(cfg):

    # set logger, tensorboard, and devices
    cfg['save_dir'] = utils.get_save_dir(cfg['save_dir'], cfg['name'], training=True)
    log = utils.get_logger(cfg["save_dir"], cfg["name"])
    tbx = SummaryWriter(cfg["save_dir"])
    device = utils.get_devices(cfg["gpu_ids"])
    cfg["train_data_loader"]["batch_size"] *= max(1, len(cfg["gpu_ids"]))
    cfg["val_data_loader"]["batch_size"] *= max(1, len(cfg["gpu_ids"]))

    log.info(f"Cfg: {dumps(cfg, indent = 4, sort_keys = True)}")


    # get model
    log.info("Building model...")
    model = LyftEffnet(cfg)
    model = nn.DataParallel(model, device_ids=cfg["gpu_ids"])
    if cfg["load_path"]:
        # reset the seed to sample some other data
        cfg["seed"] *= 2
        log.info(f"Loading checkpoint from {cfg['load_path']}...")
        model, step = utils.load_model(model, cfg["load_path"], cfg["gpu_ids"])
    else:
        step = 0
    model = model.to(device)
    model.train()


    # set random seed
    log.info(f"Using random seed {cfg['seed']}")
    utils.set_seed(cfg["seed"])

    # get saver
    saver = utils.CheckpointSaver(cfg["save_dir"], log = log)

    # get optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
    # optimizer = AdamP(model.parameters(), lr = 1e-4, weight_decay=1e-2)
    # optimizer = optim.SGD(model.parameters(), lr = 1e-4, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr = 5e-5, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr = 1e-5, momentum=0.9)

    # get dataloader
    DIR_INPUT = cfg["data_path"]
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager(None)

    log.info("Building training dataset...")
    train_cfg = cfg["train_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                                num_workers=train_cfg["num_workers"], pin_memory=True)
    log.info(str(train_dataset))

    log.info("Building validation dataset...")
    val_cfg = cfg["val_data_loader"]
    val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()
    val_mask = np.load(f"{val_cfg['dir']}/mask.npz")["arr_0"]
    val_dataset = AgentDataset(cfg, val_zarr, rasterizer, agents_mask=val_mask)
    val_dataloader = DataLoader(val_dataset, shuffle=val_cfg["shuffle"], batch_size=val_cfg["batch_size"],
                                num_workers=val_cfg["num_workers"], pin_memory=True)
    log.info(str(val_dataset))


    # Train
    log.info("Training...")
    tr_it = iter(train_dataloader)
    max_steps = cfg["train_params"]["max_num_steps"]
    losses_train = []
    steps_till_eval = cfg["train_params"]['eval_steps']
    with torch.enable_grad(), tqdm(total=max_steps) as progress_bar:
        while step < max_steps:
            try:
                data = next(tr_it)
            except StopIteration:
                tr_it = iter(train_dataloader)
                data = next(tr_it)
            
            # forward pass
            loss, _, _, batch_size = forward(data, model, device)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['train_params']['max_grad_norm'])
            optimizer.step()

            losses_train.append(loss.item())

            # log info
            step += batch_size
            progress_bar.update(batch_size)
            progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
            tbx.add_scalar('train/loss', loss, step)

            steps_till_eval -= batch_size
            if steps_till_eval <= 0:
                steps_till_eval = cfg["train_params"]["eval_steps"]

                # Evaluate and save checkpoint
                log.info(f"Evaluate at step {step}...")
                metrics = evaluate(model, val_dataloader, device)
                saver.save(step, model, metrics['neg_multi_log_likelihood'], device)

                # Log to console
                metrics_str = ', '.join(f"{k}: {v:05.2f}" for k, v in metrics.items())
                log.info(f"Validate {metrics_str}")

                # Log to tensorboard
                log.info("Visualizing in Tensorboard...")
                for k, v in metrics.items():
                    tbx.add_scalar(f"val/{k}", v, step)


def evaluate(model, data_loader, device):
    model.eval()
    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    confidences_list = []
    agent_ids = []

    progress_bar = tqdm(data_loader)

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
    
    model.train()

    pred_path = os.path.join(cfg['save_dir'], "pred.csv")

    write_pred_csv(pred_path, 
        timestamps=np.concatenate(timestamps),  
        track_ids=np.concatenate(agent_ids),    
        coords=np.concatenate(future_coords_offsets_pd),
        confs = np.concatenate(confidences_list)
        )
    val_gt_path = f"{cfg['val_data_loader']['dir']}/gt.csv"
    # metrics reference: https://github.com/lyft/l5kit/blob/380097ebd1937835d1c13ff5ec831610d42b6f73/l5kit/l5kit/evaluation/metrics.py
    metrics = compute_metrics_csv(val_gt_path, pred_path, 
        [neg_multi_log_likelihood, average_displacement_error_mean, 
        final_displacement_error_mean])
    
    return metrics

if __name__ == '__main__':
    main(cfg)