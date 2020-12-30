""" Run threshold-based answerability verification.

Author:
    Yu Yang (yang6367@umn.edu)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_retro_test_args
from collections import OrderedDict
from json import dumps
from models import SketchyReader, IntensiveReader
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device = util.get_devices(args.gpu_ids)
    args.batch_size *= max(1, len(args.gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    # Get model
    log.info('Building model...')
    sreader = SketchyReader(word_vectors=word_vectors,
                  hidden_size=args.hidden_size)
    ireader = IntensiveReader(word_vectors=word_vectors,
                  hidden_size=args.hidden_size)
    sreader = nn.DataParallel(sreader, args.gpu_ids)
    ireader = nn.DataParallel(ireader, args.gpu_ids)
    log.info(f'Loading Sketchy Reader checkpoint from {args.load_sketchy_path}...')
    sreader = util.load_model(sreader, args.load_sketchy_path, args.gpu_ids, return_step=False)
    sreader = sreader.to(device)
    sreader.eval()
    log.info(f'Loading Intensive Reader checkpoint from {args.load_intensive_path}...')
    ireader = util.load_model(ireader, args.load_intensive_path, args.gpu_ids, return_step=False)
    ireader = ireader.to(device)
    ireader.eval()
    

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    bce_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            slogits = sreader(cw_idxs, qw_idxs)
            log_p1, log_p2, ilogits = ireader(cw_idxs, qw_idxs)
            y1, y2, y = y1.to(device), y2.to(device), (y1 > 0).float().to(device)
            v = args.beta1 * slogits + args.beta2 * ilogits
            nll_loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            bce_loss = nn.BCELoss()(v, y)
            nll_meter.update(nll_loss.item(), batch_size)
            bce_meter.update(bce_loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            # use TAV
            starts, ends = util.tav(p1, p2, v, args.lambda1, args.lambda2, 
                                    args.threshold, args.max_ans_len)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg, 
                                         BCE=bce_meter.avg)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('NLL', nll_meter.avg),
                        ('BCE', bce_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        log.info('Visualizing in TensorBoard...')
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    main(get_retro_test_args())
