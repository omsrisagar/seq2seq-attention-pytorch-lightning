import glob
import multiprocessing
import os
import pickle
import time
import subprocess
from argparse import ArgumentParser
from pathlib import Path


if __name__ == "__main__":
    parser = ArgumentParser()

    # add PROGRAM level args
    # parser.add_argument("--N_samples", type=int, default=256 * 10)
    parser.add_argument("--data_dir", type=str, default="data/240712_Experimenal_Data/varying_all_noise", help="path to the directory containing training files")
    parser.add_argument("--base_folder", type=str, default="train", help="path to the root training folder where pla no_pla and bm train results are stored")
    parser.add_argument("--cmds_to_run_file", type=str, default="", help="path to the file containing list of commands to run")
    parser.add_argument("--num_workers", type=int, default=0, help="number of parallel workers; give 0 to use all 16")
    parser.add_argument("--gpus", type=str, default="0,", help="Which gpus to train on e.g., '1,4'; use -1 to train on all--> might use DDP!")
    parser.add_argument("--max_epochs", type=int, default=70, help="Number of epochs to run")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--debug", action='store_true', help='adds --debug flag to runs')
    parser.add_argument("--resume_checkpoint", action='store_true', help='loads the latest checkpoint')
    args = parser.parse_args()

    exp_name = os.path.basename(args.data_dir)
    os.makedirs(args.base_folder, exist_ok=True)
    model_names = ['no_pla', 'pla', 'base_model']
    # model_names = [model_names[-1]]

    # Regular Seq2Seq model
    nopla_base_cmd = f"python seq2seq_trainer_activity_recg.py --gpus {args.gpus} --batch_size {args.batch_size} --max_epochs {args.max_epochs} --N_valid_size 0.2 --exclude_eos 1 --use_pred_eos 0 --use_pla 0 --teacher_forcing_ratio 1 --use_base_model 0 --use_max_seq_len 0 --same_vocab_in_out 0"

    # PLA based seq2seq model
    pla_base_cmd = f"python seq2seq_trainer_activity_recg.py --gpus {args.gpus} --batch_size {args.batch_size} --max_epochs {args.max_epochs} --N_valid_size 0.2 --exclude_eos 1 --use_pred_eos 0 --use_pla 1 --teacher_forcing_ratio 0 --use_base_model 0 --use_max_seq_len 0 --same_vocab_in_out 0"

    # # Multi-label classification (base model)
    bm_base_cmd = f"python seq2seq_trainer_activity_recg.py --gpus {args.gpus} --batch_size {args.batch_size} --max_epochs {args.max_epochs} --N_valid_size 0.2 --exclude_eos 1 --use_pred_eos 0 --use_pla 0 --teacher_forcing_ratio 0 --use_base_model 1 --same_vocab_in_out 0"

    base_cmds = [nopla_base_cmd, pla_base_cmd, bm_base_cmd]
    # base_cmds = [base_cmds[-1]]

    training_files = glob.glob(os.path.join(args.data_dir, "ar-training-*"))
    training_files = training_files[:6] if args.debug else training_files

    if args.cmds_to_run_file:
        assert os.path.isfile(args.cmds_to_run_file), "Unable to open provided cmds_to_run file"
        with open(args.cmds_to_run_file, 'rb') as f:
            commands_to_run = pickle.load(f)
    else:
        commands_to_run = []
        for file in sorted(training_files):
            # Add model independent run args here
            addon_str = " --debug" if args.debug else ""
            addon_str += " --train_data_path " + file
            for i in range(len(model_names)):
                # Add model dependent run args here
                logdir = Path(args.base_folder, model_names[i], exp_name)
                model_addon_str = " --log_dir " + str(logdir)
                if args.resume_checkpoint:
                    ckpt_dir = Path(logdir, Path(file).stem, 'csv_logs')
                    last_version = sorted(os.listdir(ckpt_dir), reverse=True)[0]
                    ckpt = str(next(Path(ckpt_dir, last_version, 'checkpoints').iterdir()))
                    assert ckpt.endswith('.ckpt'), "Not a checkpoint file"
                    assert 'epoch=99' in ckpt, "ckpt not trained till epoch 100" # hard coded for now, check!
                    model_addon_str += " --resume_checkpoint " + ckpt
                commands_to_run.append(base_cmds[i] + addon_str + model_addon_str)

    print(f"Total number of commands to run: {len(commands_to_run)}")

    def run_cmd(i_cmd):
        i, cmd = i_cmd
        print(f"Running command {i+1}/{len(commands_to_run)}: {cmd}")
        return subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    start_time = time.time()
    pool = multiprocessing.Pool(None if args.num_workers == 0 else args.num_workers)
    result = pool.map(run_cmd, list(enumerate(commands_to_run)))

    pool.close()
    pool.join()
    if sum(result) != 0:
        failed_runs = [commands_to_run[i] for i in range(len(result)) if result[i]]
        print("\nFollowing runs failed!\n")
        print(failed_runs)
        print(f"\nNumber of failed runs: {len(failed_runs)}\n Saving list to train/failed_runs.pkl")
        with open(Path(args.base_folder, 'failed_runs.pkl'), 'wb') as file:
            pickle.dump(failed_runs, file)
    print(f"Finished running in {(time.time() - start_time)/3600} hours")

