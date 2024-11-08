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
    parser.add_argument("--data_dir", type=str, default="data/240712_Experimental_Data/varying_all_noise", help="path to the directory containing training files")
    parser.add_argument("--base_folder", type=str, default="train", help="path to the root training folder where pla no_pla and bm train results are stored")
    parser.add_argument("--ckpt_to_use", type=str, default="best", help="best or last checkpoint to use")
    parser.add_argument("--last_epoch", type=int, default=100, help="Used for assertion - to make sure training ran this many epochs")
    parser.add_argument("--write_results", type=int, default=1, help="Whether to write results to excel file.")
    # parser.add_argument("--debug", action='store_true', help='adds --debug flag to runs')
    args = parser.parse_args()

    exp_name = os.path.basename(args.data_dir)
    model_names = ['no_pla', 'pla', 'base_model']
    legend_model_names = ['Order-Aware', 'Order-Agnostic', 'Multi-label']
    noise_levels = [0, 0.2, 0.4, 0.6, 0.8]
    data_sizes = [10, 50, 250, 1250, 6250, 31250]

    def get_correct_order(filepath: str):
        filename = Path(filepath).stem # ar-training-data_05000000_31250
        nums = filename.split('_')[1:]
        return '{:0>8}_{:0>5}'.format(*nums)

    training_files = sorted(glob.glob(os.path.join(args.data_dir, "ar-training-*")), key=get_correct_order)
    # training_files = training_files[:12] if args.debug else training_files
    len_dim2 = 6 # sizes: 10, 50, ..., 31250
    len_dim1 = len(training_files) // len_dim2 # noise levels

    # epoch based metrics
    metrics = ['train_loss_epoch', 'train_sequence_acc_epoch',
               'train_precision_acc_epoch', 'train_recall_acc_epoch', 'train_f1_acc_epoch',
               'test_loss', 'test_sequence_acc',
               'test_precision_acc', 'test_recall_acc', 'test_f1_acc']

    # to get data from train csv files
    usecols = metrics + ['epoch',
               'train_precision_acc_err_epoch', 'train_recall_acc_err_epoch', 'train_f1_acc_err_epoch',
               'test_precision_acc_err', 'test_recall_acc_err', 'test_f1_acc_err']

    metrics_desc = ['Train Loss', 'Train Accuracy (Sequence or Label)',
                    'Train Precision', 'Train Recall', 'Train F1 Score',
                    'Test Loss', 'Test Accuracy (Sequence or Label)',
                    'Test Precision', 'Test Recall', 'Test F1 Score']

    # can be deleted once we rename err columns such that err is appened to existing name
    err_dict = {'train_precision_acc_epoch': 'train_precision_acc_err_epoch',
                'train_recall_acc_epoch': 'train_recall_acc_err_epoch',
                'train_f1_acc_epoch': 'train_f1_acc_err_epoch',
                'test_precision_acc': 'test_precision_acc_err',
                'test_recall_acc': 'test_recall_acc_err',
                'test_f1_acc': 'test_f1_acc_err'
                }

    usecols_dict = {key : np.zeros((len(model_names), len_dim1, len_dim2)) for key in usecols if key != 'epoch'}
    metrics_desc_dict = dict(zip(metrics, metrics_desc))

    # for gathering data use usecols
    for i in range(len(model_names)):
        for j in range(len_dim1):
            for k in range(len_dim2):
                logdir = Path(args.base_folder, model_names[i], exp_name)
                ckpt_dir = Path(logdir, Path(training_files[j * len_dim2 + k]).stem, 'csv_logs')
                last_version = sorted(os.listdir(ckpt_dir), reverse=True)[0]
                csv_file = Path(ckpt_dir, last_version, 'metrics.csv')
                df = pd.read_csv(csv_file, usecols=usecols)
                for key, value in usecols_dict.items():
                    indx = -3 if 'train' in key else -2 if args.ckpt_to_use == 'best' else -1
                    if 'train' in key:
                        assert df['epoch'].iloc[indx] == args.last_epoch - 1 # make sure training ran till this point
                    else:
                        assert df['epoch'].iloc[indx] == args.last_epoch
                    value[i][j][k] = df[key].iloc[indx]


    # for plotting use metrics
    for i in range(len_dim1):
        output_folder = Path(args.base_folder, 'results', exp_name, f'{args.ckpt_to_use}_checkpoint',
                             f'p={noise_levels[i]}')
        output_folder.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(f'{output_folder}/p={noise_levels[i]}.xlsx') as writer:
            for key in metrics:
                if args.write_results:
                    df = pd.DataFrame(usecols_dict[key][:,i,:], columns=data_sizes, index=legend_model_names)
                    # df.to_csv(f'{output_folder}/{key}.csv', index=True, header=True)
                    df.to_excel(writer, sheet_name=metrics_desc_dict[key], index=True, header=True)
                    if 'loss' not in key and 'sequence' not in key: # write error information as well
                        df = pd.DataFrame(usecols_dict[err_dict[key]][:, i, :], columns=data_sizes, index=legend_model_names)
                        # df.to_csv(f'{output_folder}/{key}_err.csv', sep='\t', index=True, header=True)
                        df.to_excel(writer, sheet_name=metrics_desc_dict[key] + ' Error', index=True, header=True)
                plot_figures(
                    output_path=output_folder,
                    desc=key,
                    y=usecols_dict[key][:, i, :],
                    yerr=usecols_dict[err_dict[key]][:, i, :] if 'loss' not in key and 'sequence' not in key else None,
                    xlabel='Data Size (x 14)',
                    # xlabel='Probability of noise added',
                    # xlabel='Varying Seed Deviation',
                    ylabel=metrics_desc_dict[key],
                    # x=np.arange(0, 0.91, 0.05),
                    xticklabels=data_sizes,
                    legend=legend_model_names,
                    show_plot=True,
                    gen_pkl=True,
                    save_pdf=True,
                )

