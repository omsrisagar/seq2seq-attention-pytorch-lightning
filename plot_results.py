import glob
import multiprocessing
import os
import pickle
import time
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hmean
from myutils.utils import plot_figures

if __name__ == "__main__":
    parser = ArgumentParser()

    # add PROGRAM level args
    # parser.add_argument("--N_samples", type=int, default=256 * 10)
    parser.add_argument("--data_dir", type=str, default="data/240712_Experimenal_Data/varying_all_noise", help="path to the directory containing training files")
    # parser.add_argument("--cmds_to_run_file", type=str, default="", help="path to the file containing list of commands to run")
    # parser.add_argument("--num_workers", type=int, default=0, help="number of parallel workers; give 0 to use all 16")
    parser.add_argument("--debug", action='store_true', help='adds --debug flag to runs')
    # parser.add_argument("--resume_checkpoint", action='store_true', help='loads the latest checkpoint')
    args = parser.parse_args()

    exp_name = os.path.basename(args.data_dir)
    model_names = ['no_pla', 'pla', 'base_model']
    legend_model_names = ['Order-Aware', 'Order-Agnostic', 'Multi-label']

    training_files = sorted(glob.glob(os.path.join(args.data_dir, "ar-training-*")))
    training_files = training_files[:6] if args.debug else training_files

    # epoch based metrics
    metrics = ['train_loss_epoch', 'train_sequence_acc_epoch', 'train_precision_acc_epoch', 'train_recall_acc_epoch',
               'test_loss', 'test_sequence_acc', 'test_precision_acc', 'test_recall_acc']

    metrics_desc = ['Train Loss', 'Train Accuracy (Sequence/Label)', 'Train Precision', 'Train Recall',
                    'Test Loss', 'Test Accuracy (Sequence/Label)', 'Test Precision', 'Test Recall']

    metrics_dict = {key : np.zeros((len(model_names), len(training_files))) for key in metrics}
    metrics_desc_dict = dict(zip(metrics, metrics_desc))

    for i in range(len(model_names)):
        for j in range(len(training_files)):
            logdir = Path('train', model_names[i], exp_name)
            ckpt_dir = Path(logdir, Path(training_files[j]).stem, 'csv_logs')
            last_version = sorted(os.listdir(ckpt_dir), reverse=True)[0]
            csv_file = Path(ckpt_dir, last_version, 'metrics.csv')
            df = pd.read_csv(csv_file, usecols=metrics)
            for key, value in metrics_dict.items():
                indx = -2 if 'train' in key else -1
                value[i][j] = df[key].iloc[indx]

    # Calculate F1 scores and store in dict
    prec_recall_array = np.concatenate((np.expand_dims(metrics_dict['train_precision_acc_epoch'], axis=-1),
                                        np.expand_dims(metrics_dict['train_recall_acc_epoch'], axis=-1)), axis=-1)
    metrics_dict['train_f1_acc_epoch'] = hmean(prec_recall_array, axis=-1)
    metrics_desc_dict['train_f1_acc_epoch'] = 'Train F1 Score'
    prec_recall_array = np.concatenate((np.expand_dims(metrics_dict['test_precision_acc'], axis=-1),
                                        np.expand_dims(metrics_dict['test_recall_acc'], axis=-1)), axis=-1)
    metrics_dict['test_f1_acc'] = hmean(prec_recall_array, axis=-1)
    metrics_desc_dict['test_f1_acc'] = 'Test F1 Score'

    for key, value in metrics_dict.items():
        plot_figures(
            output_path=Path('train', 'results', exp_name),
            desc=key,
            y=value,
            xlabel='Probability of noise added',
            ylabel=metrics_desc_dict[key],
            x=np.arange(0, 0.91, 0.05),
            legend=legend_model_names,
            show_plot=True,
            gen_pkl=True,
            save_pdf=True,
        )

