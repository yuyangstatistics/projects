"""Test the Propensity Score model on CNNDM.
Based on ps2 model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from collections import OrderedDict
from json import dumps
from tqdm import tqdm
import os
from functools import partial

import util
from args import get_test_args
from data_utils import collate_mp, CNNDM
from models import ETM, PSNet
import json
import csv


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    device = util.get_devices(args.gpu_ids)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))
    args.sim_data_save_path = os.path.join(args.save_dir, 'sim_data.csv')

    # Get embeddings
    log.info('Loading embeddings...')
    tokenizer = RobertaTokenizer.from_pretrained(args.model_type, verbose = False)
    encoder = RobertaModel.from_pretrained(args.model_type)
    embeddings = encoder.embeddings.word_embeddings.weight  # 50265 x 768
    embed_size = embeddings.size(1)
    vocab_size = tokenizer.vocab_size

    # Get data loader
    log.info('Building dataset...')
    dataset = CNNDM(os.path.join(args.data_dir, args.split), args.model_type, is_test = False)
    collate_fn = partial(collate_mp, pad_token_id = tokenizer.pad_token_id, 
                                vocab_size = vocab_size, is_test = False)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, collate_fn=collate_fn)

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
    model = PSNet(n_features=args.num_topics)
    model = nn.DataParallel(model, args.gpu_ids)
    log.info(f'Loading PSNet checkpoint from {args.ps_load_path}...')
    model = util.load_model(model, args.ps_load_path, args.gpu_ids, return_step=False)
    log.info(f"PS Model: {model}")
    model = model.to(device)
    model.eval()

    ## Set up simulation
    # load propensity score dictionary
    with open(args.ps_path) as fp:
        ps_dict = json.load(fp)
    log.info(f"Propensity Score Dictionary: {ps_dict}")

    # write the header of the simulation file
    with open(args.sim_data_save_path, 'w+', newline ='') as fp:
        write = csv.writer(fp)
        write.writerow(["Treat", "True_PS", "Response", "Est_PS"])

    # Test
    log.info(f'Evaluating on {args.split} split...')
    bce_meter = util.AverageMeter()
    acc_meter = util.AverageMeter()
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
            
            _, theta, _, _, _ = etm(normalized_bows=normalized_bows) # (batch_size x K)
            output = model(theta).squeeze()  # (batch_size, )
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

            # Simulate data
            propensity_scores = torch.FloatTensor([ps_dict[f"{typ}"] for typ in batch['src_keywords_inclusion_types'].tolist()]).to(device)
            responses = torch.bernoulli(torch.sigmoid(args.alpha * target + args.beta * propensity_scores + args.gamma)) # (batch_size, )
            sim_data = torch.stack([target, propensity_scores, responses, output]).transpose(0, 1) # (batch_size x 4)
            with open(args.sim_data_save_path, 'a+', newline ='') as fp:
                write = csv.writer(fp)
                write.writerows(sim_data.tolist())
        
        results_list = [('BCE', bce_meter.avg), ('ACC', acc_meter.avg)]
        results = OrderedDict(results_list)
    
    # Log to console
    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
    log.info(f'{args.split.title()} {results_str}')
    log.info(f"The simulated data has been saved to '{args.sim_data_save_path}'.")
    

if __name__ == '__main__':
    main(get_test_args())