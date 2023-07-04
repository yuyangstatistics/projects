"""Train an ETM on CNNDM.
"""

import numpy as np
import random
import torch
import torch.nn as nn
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
from models import ETM


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

    # Get model
    log.info('Building model...')
    model = ETM(args.num_topics, vocab_size, embed_size, args.vi_nn_hidden_size,
                args.theta_act, embeddings, args.enc_drop)
    log.info(f"model: {model}")

    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
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

    # Get optimizer and scheduler
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
    val_dataset = CNNDM(os.path.join(args.data_dir, "val"), args.model_type, is_test = True)
    val_collate_fn = partial(collate_mp, pad_token_id = tokenizer.pad_token_id, 
                                vocab_size = vocab_size, is_test = False)
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.num_workers, 
                            collate_fn=val_collate_fn)

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
                _, _, kl_divergence, preds = model(normalized_bows=normalized_bows)
                recon_loss = -(preds * bows).sum(1).mean()
                kl_divergence = kl_divergence.mean()
                loss = recon_loss + kl_divergence
                loss_val = loss.item()
                
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
                                         recon_loss=recon_loss.item(), 
                                         kl_div=kl_divergence.item(), 
                                         NELBO=loss_val)
                tbx.add_scalar('train/recon_loss', recon_loss.item(), step)
                tbx.add_scalar('train/kl_div', kl_divergence.item(), step)
                tbx.add_scalar('train/NELBO', loss_val, step)
                tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'], step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_topics = evaluate(args, model, val_loader, tokenizer, device)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Val {results_str}')
                    for k, topic_words in pred_topics.items():
                        if k < 50:
                            log.info(f"Topic {k}: {topic_words}")
                    
                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'val/{k}', v, step)
                    util.visualize(tbx,
                                   pred_topics=pred_topics,
                                   step=step,
                                   split='val',
                                   vis_num_topics=args.vis_num_topics)


def evaluate(args, model, data_loader, tokenizer, device):
    ppl_meter = util.AverageMeter()
    model.eval()
    with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
        beta = model(beta_only=True)[:args.num_topics, :]  # (K x vocab_size)
        top_10_indices = beta.argsort(dim = -1, descending=True)[:, :10]  # (K x 10)
        frequency = torch.zeros(beta.size(0), 55).to(device)  # (K x 55), occurrence and cooccurrence

        for batch in data_loader:
            bows = batch['src_bows'].to(device)
            batch_size = bows.size(0)
            half_batch_size = batch_size // 2
            bows_1 = bows[:half_batch_size, :]
            bows_2 = bows[half_batch_size: 2 * half_batch_size, :]
            
            # get theta from the first half of the documents
            sums_1 = bows_1.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_bows_1 = bows_1 / sums_1  # (half_batch_size x vocab_size)
            else:
                normalized_bows_1 = bows_1
            _, _, _, preds = model(normalized_bows=normalized_bows_1)
            # get prediction on the second half of the documents
            recon_loss = -(preds * bows_2).sum(1) # (half_batch_size, )
            sums_2 = bows_2.sum(1)
            loss = recon_loss / sums_2  # (half_batch_size, )
            ppl_meter.update(loss.mean().item(), half_batch_size)

            # get word-doc occurrence frequency: used for topic coherence calculation
            occurrence = 1 * (torch.stack([bows[i, :][top_10_indices] \
                for i in range(batch_size)]) != 0)  # (batch_size x K x 10)
            cooccurence = torch.stack([torch.stack([util.get_outer_triu_values(occurrence[i, j, :]) \
                for j in range(occurrence.size(1))]) for i in range(occurrence.size(0))])  # (batch_size x K x 55)
            frequency += cooccurence.sum(0)  # (K x 55)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(PPL=ppl_meter.avg)
        
        results_list = [('PPL', ppl_meter.avg), 
                        ('TC', util.get_topic_coherence(frequency.data.cpu().numpy(), \
                            len(data_loader.dataset))), 
                        ('TD', util.get_topic_diversity(beta, args.td_topnum))]
        results = OrderedDict(results_list)
        pred_topics = util.get_topics(beta, tokenizer, args.vis_num_words)
    model.train()
    return results, pred_topics

if __name__ == '__main__':
    main(get_train_args())