"""Command-line arguments.
Adapted from SQuAD code.
"""

import argparse

def get_train_args():
    """Get arguments needed in train.py."""
    parser = argparse.ArgumentParser('Train a ETM on CNNDM')

    add_train_test_args(parser)
    add_etm_args(parser)
    add_psmodel_args(parser)

    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=10,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='BCE',
                        choices=('BCE', 'ACC'), 
                        help='Name of validation metric to determine best checkpoint.')
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        choices=('adam', 'adagrad', 'adadelta', 'rmsprop', 'asgd', 'sgd'),
                        help='Optimizer used in training.')
    parser.add_argument('--eval_steps',
                        type=int,
                        default=50000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=5,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.02,
                        help='Learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=1.2e-6,
                        help='L2 weight decay.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dropout layers.')

    args = parser.parse_args()  # can only be added at the last parser round

    if args.metric_name in ('BCE'):
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('ACC'):
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    return args


def get_test_args():
    """Get arguments needed in test.py."""
    parser = argparse.ArgumentParser('Test a trained PS model on CNNDM')

    print("Loading train_test args...")
    add_train_test_args(parser)
    print("Loading etm args...")
    add_etm_args(parser)
    print("Loading psmode args...")
    add_psmodel_args(parser)
    print("Loding ps sim args...")
    add_ps_sim_args(parser)

    parser.add_argument('--split',
                        type=str,
                        default='test',
                        choices=('train', 'val', 'test'),
                        help='Split to use for testing.')

    # Require ps_load_path and etm_load_pathfor test.py
    args = parser.parse_args()
    if not args.ps_load_path:
        raise argparse.ArgumentError('Missing required argument --ps_load_path')
    if not args.etm_load_path:
        raise argparse.ArgumentError('Missing required argument --etm_load_path')

    return args


def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify the model.')
    parser.add_argument('--gpu_ids',
                        nargs='+',
                        type=int,
                        default=0, 
                        required=True,
                        help='GPU ids. (ex. 3 4 5 6)')                
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')
    parser.add_argument('--data_dir',
                        type=str,
                        default='/home/yang6367/summarizer/cnn-dailymail/processed', 
                        help='The processed CNNDM data directory.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')


def add_etm_args(parser):
    parser.add_argument('--model_type',
                        type=str,
                        default='roberta-base', 
                        help='The type of Roberta model.')                             
    parser.add_argument('--num_topics',
                        type=int,
                        default=300,   # when larger than vi_nn_hidden_size, there will be nans
                        help='Number of topics for the ETM.')   
    parser.add_argument('--vi_nn_hidden_size',
                        type=int,
                        default=800, 
                        help='Hidden layer size in the VI Neural Networks.')     
    parser.add_argument('--theta_act',
                        type=str,
                        default='relu',
                        choices=('relu', 'rrelu', 'leakyrelu', 'elu', 'selu', 'glu', 'tanh', 'softplus'),
                        help='Activation function in the VI Nerual Networks.')     
    parser.add_argument('--enc_drop',
                        type=float,
                        default=0.0, 
                        help='Dropout rate for the Encoder.')  
    parser.add_argument('--etm_load_path',
                        type=str,
                        default="/home/yang6367/summarizer/etm/save/train/etm-01/best.pth.tar",
                        help='The trained ETM checkpoint.')  
    parser.add_argument('--bow_norm',
                        type=int,
                        default=1, 
                        help='Whether to normalize the BOWs.')                           


def add_psmodel_args(parser):
    parser.add_argument('--doc_len_threshold',
                        type=int,
                        default=800,
                        help='The document length threshold for treatment assignment.')             
    parser.add_argument('--ps_load_path',
                        type=str,
                        default="",
                        help='The trained PSNet checkpoint.')         
    parser.add_argument('--sim_data_path',
                        type=str,
                        default="",
                        help='The trained PSNet checkpoint.')                             

def add_ps_sim_args(parser):
    parser.add_argument('--alpha',
                        type=float,
                        default=-0.25,
                        help='Alpha parameter for simulation.')             
    parser.add_argument('--beta',
                        type=float,
                        default=5.0,
                        help='Beta parameter for simulation.')         
    parser.add_argument('--gamma',
                        type=float,
                        default=-5.0,
                        help='Gamma parameter for simulation.')     
    parser.add_argument('--ps_path',
                        type=str,
                        default="/home/yang6367/text-causal/etm/save/propensity_scores.json",
                        help='True propensity score path.')                                             