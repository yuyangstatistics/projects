"""Train an ETM on CNNDM.
"""

import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn.functional import kl_div
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from collections import OrderedDict
from json import dumps
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
from functools import partial

import util
from args import get_train_args
from data_utils import collate_mp, CNNDM
from models import ETM, PSNet


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device = util.get_devices(args.gpu_ids)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    tokenizer = RobertaTokenizer.from_pretrained(args.model_type, verbose = False)
    encoder = RobertaModel.from_pretrained(args.model_type)
    embeddings = encoder.embeddings.word_embeddings.weight  # 50265 x 768
    embed_size = embeddings.size(1)
    vocab_size = tokenizer.vocab_size

    # Get data loader
    log.info('Building dataset...')
    train_dataset = CNNDM(os.path.join(args.data_dir, "train"), args.model_type, is_test = False)
    train_collate_fn = partial(collate_mp, pad_token_id = tokenizer.pad_token_id, 
                                vocab_size = vocab_size, is_test = False)
    train_loader = DataLoader(train_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=True, 
                                num_workers=args.num_workers, 
                                collate_fn=train_collate_fn)
    val_dataset = CNNDM(os.path.join(args.data_dir, "val"), args.model_type, is_test = False)
    val_collate_fn = partial(collate_mp, pad_token_id = tokenizer.pad_token_id, 
                                vocab_size = vocab_size, is_test = False)
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.num_workers, 
                            collate_fn=val_collate_fn)

    # Get ETM
    log.info('Building ETM...')
    etm = ETM(args.num_topics, vocab_size, embed_size, args.vi_nn_hidden_size,
                args.theta_act, embeddings, args.enc_drop)
    log.info(f"ETM: {etm}")

    etm = nn.DataParallel(etm, args.gpu_ids)
    log.info(f'Loading ETM checkpoint from {args.etm_load_path}...')
    etm= util.load_model(etm, args.etm_load_path, args.gpu_ids, return_step=False)

    etm = etm.to(device)
    etm.eval()
    for param in etm.parameters():
        param.requires_grad = False

    # get PS model
    log.info('Building Propensity Neural Net Model...')
    model = PSNet(n_features=embed_size)
    log.info(f"PS Model: {model}")
    
    model = nn.DataParallel(model, args.gpu_ids)
    if args.ps_load_path:
        model, step = util.load_model(model, args.ps_load_path, args.gpu_ids)
    else:
        step = 0
    
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)


    # Get optimizer and scheduler and loss
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_wd)  # l2 weight decay
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.l2_wd)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.l2_wd)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2_wd)
    elif args.optimizer == 'asgd':
        optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.l2_wd)
    else:
        print('Defaulting to vanilla SGD')
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    scheduler = sched.LambdaLR(optimizer, lambda s: 0.999 ** s)  # Constant LR
    criterion = nn.BCELoss()

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for (i, batch) in enumerate(train_loader):
                # Setup for forward
                optimizer.zero_grad()
                model.zero_grad() # added for performance consideration

                bows = batch["src_bows"].to(device) # (batch_size x vocab_size)
                batch_size = bows.size(0)
                sums = bows.sum(1).unsqueeze(1) # (batch_size x 1)
                if args.bow_norm:
                    normalized_bows = bows / sums   # (batch_size x vocab_size)
                else:
                    normalized_bows = bows

                if torch.isnan(normalized_bows).any():
                    log.info(f"There are NaNs in bows at batch {i}")

                # Forward
                _, _, _, _, topic_features = etm(normalized_bows=normalized_bows) # (batch_size x embed_size)
                output = model(topic_features).squeeze()  # (batch_size, )
                src_input_lens = batch["src_input_lens"].to(device)
                target = 1.0 * (src_input_lens > args.doc_len_threshold) # (batch_size, )
                loss = criterion(output, target)
                
                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, 
                                         loss=loss.item())
                tbx.add_scalar('train/loss', loss.item(), step)
                tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'], step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results = evaluate(args, etm, model, val_loader, device)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Val {results_str}')
                    
                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'val/{k}', v, step)


def evaluate(args, etm, model, data_loader, device):
    bce_meter = util.AverageMeter()
    acc_meter = util.AverageMeter()
    model.eval()
    criterion = nn.BCELoss()
    with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
        for batch in data_loader:
            bows = batch["src_bows"].to(device) # (batch_size x vocab_size)
            batch_size = bows.size(0)
            sums = bows.sum(1).unsqueeze(1) # (batch_size x 1)
            if args.bow_norm:
                normalized_bows = bows / sums   # (batch_size x vocab_size)
            else:
                normalized_bows = bows
            
            _, _, _, _, topic_features = etm(normalized_bows=normalized_bows) # (batch_size x embed_size)
            output = model(topic_features).squeeze()  # (batch_size, )
            src_input_lens = batch["src_input_lens"].to(device)
            target = 1.0 * (src_input_lens > args.doc_len_threshold) # (batch_size, ), 1.0 to make it to float
            loss = criterion(output, target)
            pred = 1.0 * (output > 0.5)
            accuracy = (1.0 * (pred == target)).mean()
            
            bce_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy.item(), batch_size)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(BCE=bce_meter.avg, 
                                     ACC=acc_meter.avg)
        
        results_list = [('BCE', bce_meter.avg), ('ACC', acc_meter.avg)]
        results = OrderedDict(results_list)
    model.train()
    return results

if __name__ == '__main__':
    main(get_train_args())