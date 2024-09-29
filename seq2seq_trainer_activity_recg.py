import argparse
import os
import sys
from argparse import ArgumentParser
import random
import pprint as pp
import numpy as np
from csv import writer
import logging
from contextlib import redirect_stdout

# # python.dataScience.notebookFileRoot=${fileDirname}
# wdir = os.path.abspath(os.getcwd() + "/../../")
# sys.path.append(wdir)

# print(sys.path)
# print(wdir)


import text_loaders as tl
import rnn_encoder_decoder as encdec

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics.classification.accuracy import Accuracy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger

from text_loaders import PAD_token, UNK_token, SOS_token, EOS_token
from munkres import Munkres
m = Munkres()


class Seq2SeqTrainer(pl.LightningModule):
    """Encoder decoder pytorch module for trainning seq2seq model with teacher forcing

    Module try to learn mapping from one sequence to antother. This implementation try to learn to reverse string of chars
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--emb_dim", type=int, default=32)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.1)
        return parser

    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        padding_index=0,
        emb_dim=8,
        hidden_dim=32,
        dropout=0.1,
        teacher_forcing_ratio=0.5,
        output_file="",
        exclude_eos=False, # for loss and seq_accuracy calculation
        use_pred_eos=True, # if True, tokens till EOS is predicted are taken as output, else till ground truth EOS
        use_pla=True, # whether to include orderless PLA for reordering targets; if False default order of tgts is used
        use_base_model=False,
        use_max_seq_len=False, # whether to use max. sequence len for predicting
        **kwargs,
    ) -> None:
        super().__init__()

        # dynamic, based on tokenizer vocab size defined in datamodule
        self.input_dim = input_vocab_size
        self.output_dim = output_vocab_size

        self.enc_emb_dim = emb_dim  # ENC_EMB_DIM
        self.dec_emb_dim = emb_dim  # DEC_EMB_DIM

        self.enc_hid_dim = hidden_dim  # ENC_HID_DIM
        self.dec_hid_dim = hidden_dim  # DEC_HID_DIM

        self.enc_dropout = dropout  # ENC_DROPOUT
        self.dec_dropout = dropout  # DEC_DROPOUT

        self.teacher_forcing_ratio = teacher_forcing_ratio # used during training
        self.exclude_eos = exclude_eos
        self.use_pred_eos = use_pred_eos
        self.use_pla = use_pla

        self.use_base_model = use_base_model
        self.use_max_seq_len = use_max_seq_len

        self.pad_idx = padding_index

        self.output_file = output_file

        self.save_hyperparameters()

        self.max_epochs = kwargs["max_epochs"]

        self.learning_rate = 0.0005

        # self.input_src = torch.LongTensor(1).to(self.device)
        # self.input_src_len = torch.LongTensor(1).to(self.device)
        # self.input_trg = torch.LongTensor(1).to(self.device)

        # todo: remove it this blocks loading state_dict from checkpoints
        # Error(s) in loading state_dict for Seq2SeqCorrector:
        # size mismatch for input_src: copying a param with shape
        # torch.Size([201, 18]) from checkpoint,
        # the shape in current model is torch.Size([1]).
        # self.register_buffer("input_src", torch.LongTensor(1))
        # self.register_buffer("input_src_len", torch.LongTensor(1))
        # self.register_buffer("input_trg", torch.LongTensor(1))

        if self.use_base_model:
            self.fc1 = nn.Linear(self.enc_hid_dim * 2, self.enc_hid_dim)
            self.fc2 = nn.Linear(self.enc_hid_dim, self.output_dim)
            self._loss = nn.BCEWithLogitsLoss()
        else:
            # self._loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
            self._loss = nn.CrossEntropyLoss()

        self.attention = encdec.Attention(self.enc_hid_dim, self.dec_hid_dim)

        #    INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT
        self.encoder = encdec.Encoder(
            self.input_dim,
            self.enc_emb_dim,
            self.enc_hid_dim,
            self.dec_hid_dim,
            self.enc_dropout,
        )

        self.decoder = encdec.Decoder(
            self.output_dim,  # OUTPUT_DIM,
            self.dec_emb_dim,  # DEC_EMB_DIM,
            self.enc_hid_dim,  # ENC_HID_DIM,
            self.dec_hid_dim,  # DEC_HID_DIM,
            self.dec_dropout,  # DEC_DROPOUT,
            self.attention,
        )

        self._init_weights()

    def _init_weights(self):

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):

        # src = [src len, batch size]
        # src_len = [batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs TODO: change to registered buffer in pyLightning
        decoder_outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(
            self.device
        )

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        mask = self.create_mask(src)
        # mask = [batch size, src len]
        # without sos token at the beginning and eos token at the end

        if self.use_base_model:
            # return self.fc2(self.fc1(hidden))
            layer1 = self.fc1(encoder_outputs).transpose(0, 1) # batch_size x time_dim x hidden_dim*2
            # return self.fc2(layer1).sum(dim=1)
            layer1_true = layer1 * mask.unsqueeze(2) # multiply by mask to zero out padded data
            layer1_true = torch.sum(layer1_true, dim=1) # sum across time dimension
            return self.fc2(layer1_true)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        # starting with input=<sos> (trg[0]) token and try to predict next token trg[1] so loop starts from 1 range(1, trg_len)
        for t in range(1, trg_len): # Here we know src_len=trg_len, that's why we are runnig till trg_len, otherwise, we need to run max_seq_len

            # insert input token (will be embedded internally), previous hidden state, all encoder hidden states
            #  and mask
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)

            # place predictions in a tensor holding predictions for each token
            decoder_outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return decoder_outputs

    def loss(self, logits, target):

        return self._loss(logits, target)

    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=5e-4)

        # optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = optim.LambdaLR(optimizer, ...)
        # return [optimizer], [scheduler]

        # optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        # scheduler = optim.lr_scheduler.InverseSquareRootLR(optimizer, self.lr_warmup_steps)
        # return (
        #     [optimizer],
        #     [
        #         {
        #             "scheduler": scheduler,
        #             "interval": "step",
        #             "frequency": 1,
        #             "reduce_on_plateau": False,
        #             "monitor": "val_loss",
        #         }
        #     ],
        # )
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                # steps_per_epoch=int(len(self.train_dataloader())),
                steps_per_epoch=int(len(self.trainer._data_connector._train_dataloader_source.dataloader())),
                epochs=self.max_epochs,
                anneal_strategy="linear",
                final_div_factor=1000,
                pct_start=0.01,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]
    def calc_loss_and_metrics(self, batch, mode='train'):
        """
        Calculates loss, sequential accuracy, precision, recall
        """
        src_batch, trg_batch = batch

        src_seq = src_batch["src_ids"]
        # change from [batch, seq_len] -> to [seq_len, batch]
        src_seq = src_seq.transpose(0, 1)
        src_lengths = src_batch["src_lengths"]

        trg_seq = trg_batch["trg_ids"]
        # change from [batch, seq_len] -> to [seq_len, batch]
        trg_seq = trg_seq.transpose(0, 1)
        trg_lengths = trg_batch["trg_lengths"]

        # resize input buffers, should speed up training and help
        # with memory leaks https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741

        # self.input_src.resize_(src_seq.shape).copy_(src_seq)
        # self.input_src_len.resize_(src_lengths.shape).copy_(src_lengths)
        # self.input_trg.resize_(trg_seq.shape).copy_(trg_seq)

        # just for testing lr scheduler
        # output = torch.randn((*trg_seq.size(), self.output_dim), requires_grad=True, device=trg_seq.device)

        # output = self.forward(self.input_src, self.input_src_len, self.input_trg)
        # old version of forward, with tensors from dataloader
        outputs = self.forward(src_seq, src_lengths, trg_seq, teacher_forcing_ratio=self.teacher_forcing_ratio if mode == 'train' else 0)
        if self.use_base_model: # multi-label
            without_sos = trg_seq[1:].transpose(0, 1)
            _, eos_indices = self.create_eos_mask(without_sos, from_eos=True)
            bs = trg_seq.shape[1]
            trg_bm = torch.zeros((bs, self.output_dim), device=trg_seq.device)
            for i in range(bs):
                trg_bm[i, without_sos[i][:eos_indices[i]]] = 1
            logits_bm = outputs
            pred_probs_bm = F.sigmoid(logits_bm)
            pred_bm = (pred_probs_bm > 0.5).int()
        else:
            # do not know if this is a problem, loss will be computed with sos token

            # without zeros at the beginning i.e.,indx=0 (note that output is initially all zeros and updated only index=1 onwards)
            logits = outputs[1:].transpose(0, 1)

            # without sos at the beginning (as this is the ground truth seq. we append sos ourselves in the beginning)
            trg = trg_seq[1:].transpose(0, 1)

            # Now sort trg and output by trg_len before feeding to pla function
            if self.use_max_seq_len:
                trg_sorted, logits_sorted, trg_lengths_sorted = trg, logits, trg_lengths-1
            else:
                trg_sorted, logits_sorted, trg_lengths_sorted = self.sort_labels(logits, trg, trg_lengths)
            pred_sorted = logits_sorted.argmax(dim=2)
            pred_probs_sorted = F.softmax(logits_sorted, dim=2)

            # accumulate across batch
            # trg = [(trg len - 1) * batch size]
            # logits = [(trg len - 1) * batch size, output dim]

            if self.exclude_eos:
                # Make EOS tokens also to zero for loss calculation & also accuracy as below
                trg_from_eos_mask, trg_len_till_eos = self.create_eos_mask(trg_sorted, from_eos=True)
                trg_sorted = trg_sorted * trg_from_eos_mask
            # else:
            #     # Get mask for tokens after EOS token
            #     trg_after_eos_mask, trg_len_till_eos = self.create_eos_mask(trg_sorted, from_eos=False)

            if self.use_pred_eos: # keep only labels till EOS; zeros after that. # we may have to include_eos in this case otherwise network won't learn to predict eos (instead it will be penalized); for now ignore use_pred_eos
                assert not self.exclude_eos, "Should include EOS in ground truth for guidance and learning"
                # Make EOS tokens also to zero for loss calculation & also accuracy as below
                pred_eos_mask, pred_len_till_eos = self.create_eos_mask(pred_sorted, from_eos=self.exclude_eos)
                pred_sorted *= pred_eos_mask
                logits_sorted *= pred_eos_mask.unsqueeze(-1) # anyways there is 0 mask for loss - so wherever there is 0 in trg, those logits are not used
            else: # we want keep all the labels predicted
                pred_len_till_eos = torch.full_like(trg_lengths_sorted, trg_lengths_sorted.max()) # allow max. possible generation;
            # elif self.use_max_seq_len: # we want keep all the labels predicted
            #     pred_len_till_eos = trg_lengths_sorted # pred_sorted remains same;
            # else:
            #     mask_to_use = trg_from_eos_mask if self.exclude_eos else trg_after_eos_mask
            #     pred_sorted *= mask_to_use
            #     logits_sorted *= mask_to_use.unsqueeze(-1) # anyways there is 0 mask for loss - so wherever there is 0 in trg, those logits are not used
            #     pred_len_till_eos = trg_len_till_eos

            if self.use_pla:
                assert self.exclude_eos, "If PLA tyring to predict EOS does not make sense" # prob is that you will have trg rearrange like [4, 5,0,0,2,0,7] -->which may not be correct guidance
                # Now reorder targets for position loss alignment (PLA)
                trg_sorted = self.order_the_targets_pla(logits_sorted, trg_sorted, trg_lengths_sorted)

            logits = logits_sorted.reshape(-1, self.output_dim) # check notes for validation step here
            trg = trg_sorted.reshape(-1)

        if self.use_base_model:
            # Calculate binary cross entropy loss
            loss = self.loss(logits_bm, trg_bm)

            # sequence accuracy: compare list of predicted ids for all sequences in a batch to targets
            device = logits_bm.device
            accuracy = Accuracy(task="multilabel", num_labels=self.output_dim).to(device)
            acc = accuracy(pred_bm, trg_bm)
        else:
            # Calculate cross entropy loss ignoring 0 trg indices
            loss = self.loss(logits, trg)  # all '0' trg indices are ignored for loss calculation internally

            # sequence accuracy: compare list of predicted ids for all sequences in a batch to targets
            device = pred_sorted.device
            accuracy = Accuracy(task="multiclass", num_classes=self.output_dim, ignore_index=self.pad_idx).to(device)
            acc = accuracy(pred_sorted.reshape(-1), trg)

        # compute precision, recall accuracy and write to file if needed
        if self.use_base_model:
            precision, recall = self.calc_accu(pred=pred_bm,
                                               tgt=without_sos,
                                               pred_probs=pred_probs_bm.detach())
        else:
            precision, recall = self.calc_accu(pred_sorted,
                                            trg_sorted,
                                            pred_len_till_eos,
                                            pred_probs_sorted.detach())

        return loss, acc, precision, recall

    def sort_labels(self, pred, tgt, tgt_len):
        """
        Sorts labels (tgt) in decreasing order to label length (tgt_len) and also aligns tgt and pred to that.
        """
        tgt_len, sort_ind = tgt_len.sort(dim=0, descending=True)
        # reduce 1 from tgt_len as we need to exclude sos
        tgt_len -= 1
        return tgt[sort_ind], pred[sort_ind], tgt_len

    def order_the_targets_pla(self, scores, targets, label_lengths_sorted):
        device = targets.device
        scores_tensor = scores.clone()
        scores = scores.data.cpu().numpy()
        targets = targets.data.cpu().numpy()
        targets_new = targets.copy()
        targets_newest = targets.copy()
        N = scores.shape[0]
        time_steps = scores.shape[1]
        indexes = np.argmax(scores, axis=2)
        changed_batch_indexes = []
        for i in range(N):
            len_wo_eos = label_lengths_sorted[i] - 1 # originally considered by authors without including EOS
            len_with_eos = label_lengths_sorted[i]
            len_to_use = len_wo_eos if self.use_pred_eos else len_with_eos
            # len_to_use = len_with_eos # trg anyways will not have 2 at the end (because of exclude_eos assertion), so we can use full length
            common_indexes = set(targets[i][0:len_to_use]).intersection(set(indexes[i])).difference({0})
            diff_indexes = set(targets[i][0:len_to_use]).difference(set(indexes[i])).difference({0})
            diff_indexes_list = list(diff_indexes)
            common_indexes_copy = common_indexes.copy()
            index_array = np.zeros((len(diff_indexes), len(diff_indexes)))
            if common_indexes != set():
                changed_batch_indexes.append(i)
                for j in range(len_to_use): # again without considering the last predicted value
                    if indexes[i][j] in common_indexes:
                        if indexes[i][j] != targets_new[i][j].item(): # just swaps the values
                            old_value = targets_new[i][j]
                            new_value = indexes[i][j]
                            new_value_index = np.where(
                                targets_new[i] == new_value)[0][0]
                            targets_new[i][j] = new_value
                            targets_new[i][new_value_index] = old_value
                        common_indexes.remove(indexes[i][j].item())
                        if common_indexes == set():
                            break

            targets_newest[i] = targets_new[i]
            n_different = len(diff_indexes)
            if n_different > 1:
                diff_indexes_tuples = [[count, elem]
                                       for count, elem in enumerate(
                        targets_new[i][0:len_to_use]) # even here we are considering ground truth without eos
                                       if elem in diff_indexes]
                diff_indexes_locations, diff_indexes_ordered = zip(
                    *diff_indexes_tuples)
                cost_matrix = np.zeros((n_different, n_different),
                                       dtype=np.float32)
                for diff_count, diff_index_location in enumerate(
                        diff_indexes_locations):
                    losses = -F.log_softmax(
                        scores_tensor[i][diff_index_location], dim=0)
                    temp = losses[torch.LongTensor(diff_indexes_ordered)]
                    cost_matrix[diff_count, :] = temp.data.cpu().numpy()
                cost_matrix_orig = cost_matrix.copy()
                indexes2 = m.compute(cost_matrix) # modified cost_matrix also! even though docstring says otherwise
                for new_label_count, new_label in indexes2:
                    targets_newest[i][diff_indexes_locations[new_label_count]] = diff_indexes_ordered[new_label]

        targets_newest = torch.LongTensor(targets_newest).to(device)
        return targets_newest

    def filter(self, data, rem_elem):
        mask = torch.zeros_like(data)
        for elem in rem_elem:
            mask_i = data == elem
            mask += mask_i
        mask = ~mask.type(torch.bool)
        return mask

    def create_eos_mask(self, pred: torch.Tensor, from_eos: bool = True) -> torch.Tensor:
        """
        Takes prediction tensor BxT and returns BxT mask where elements from EOS token are marked False
        """
        mask = torch.ones_like(pred)
        eos_mask = (pred == EOS_token)*1
        no_eos_present = eos_mask.sum(1) == 0
        eos_indices = eos_mask.argmax(1) # First occurrence of EOS
        eos_indices = torch.where(no_eos_present, pred.shape[1]-1, eos_indices) # # if EOS token is not detected, then put max seq len
        # eos_indices[eos_indices == 0] = pred.shape[1]-1
        # mask[eos_indices:] = 0 # not working
        if not from_eos:
            eos_indices += 1 # indices after EOS are made to 0; else from EOS are made to 0
        for batch_indx in range(len(pred)):
            mask[batch_indx, eos_indices[batch_indx]:] = 0 # indices from EOS are made 0

        return mask, eos_indices

    def calc_accu(self,
                  pred: torch.Tensor,
                  tgt: torch.Tensor,
                  pred_len: torch.Tensor = None,
                  pred_probs: torch.Tensor = None) -> tuple[float, float]:
        """
        Accuracy in terms of precision - percentage of correct (those in tgt) items predicted
        Accuracy in terms of recall - percentage of items predicted that are in tgt
        pred_probs: B x T x output_dim (sequential); B x output_dim (base_model)
        write_to_file: write pred_prob, ground_truth to csv for each predicted/mispredicted (supposed to predict, but didnt)element
        """
        precision_accuracy = []
        recall_accuracy = []
        preds_TP = []
        preds_FP = []
        preds_FN = []
        for batch_indx in range(len(pred)):
            all_tgts = tgt[batch_indx, :]
            nonzero_tgts = all_tgts[all_tgts != 0]
            # always remove EOS from accuracy/prec/recall calculations
            if not self.exclude_eos:
                assert nonzero_tgts.pop() == EOS_token # remove EOS from prec/recall calculations
            tgt_filter = nonzero_tgts.cpu().tolist()
            if self.use_base_model:
                assert tgt_filter.pop() == EOS_token # remove the last EOS entry
                pred_filter = pred[batch_indx, :].nonzero().squeeze(1).cpu().tolist()
                pred_prob_filter = pred_probs[batch_indx, :].cpu().numpy() # e.g., 25 array
            else:
                preds = pred[batch_indx, :][:pred_len[batch_indx]].cpu()
                pred_filter = preds[preds != 0].tolist()
                pred_prob_filter = pred_probs[batch_indx, :, :][:pred_len[batch_indx]].cpu().numpy()  # e.g., 3x25 array
            TP = set(np.intersect1d(pred_filter, tgt_filter))
            FP = set(np.setdiff1d(pred_filter, tgt_filter))
            diff_elem = set(np.setxor1d(pred_filter, tgt_filter).astype(int)) # elements only in one of the arrays but not both
            FN = diff_elem - FP # TN --> corresponds to all elements not in prediction or ground truth for this input (check whether MW is including this or not)
            if self.output_file:
                if self.use_base_model:
                    preds_TP.extend([[pred_prob_filter[elem], 1] for elem in TP])
                    preds_FP.extend([[pred_prob_filter[elem], 0] for elem in FP])
                    preds_FN.extend([[pred_prob_filter[elem], 1] for elem in FN])
                else:
                    preds_TP.extend([[pred_prob_filter[preds.tolist().index(elem), elem],1] for elem in TP])
                    preds_FP.extend([[pred_prob_filter[preds.tolist().index(elem), elem],0] for elem in FP])
                    try:
                        if len(pred_prob_filter):
                            preds_FN.extend([[pred_prob_filter[:, elem].max(),1] for elem in FN]) # max. prob. of this element across predicted timesteps if anything is predicted at all.
                        else:
                            preds_FN.extend([[0, 1] for _ in FN]) # if nothing is predicted, then prob. is 0
                    except IndexError:
                        logging.info("index error!")
            try:
                precision_accuracy.append(len(TP) / len(pred_filter) if pred_filter else 0) # no entries predicted if 0
                recall_accuracy.append(len(TP)/len(tgt_filter))
            except TypeError:
                print("hello")
        if self.output_file:
            with open(self.output_file, 'a') as f_object:
                write_obj = writer(f_object)
                write_obj.writerows(preds_TP)
                write_obj.writerows(preds_FP)
                write_obj.writerows(preds_FN)
                f_object.close()
        return np.mean(precision_accuracy), np.mean(recall_accuracy) # precision, recall

    def training_step(self, batch, batch_idx):

        loss, acc, precision, recall = self.calc_loss_and_metrics(batch)

        # need to cast to list of predicted sequences (as list of token ids)   [ [seq1_tok1, seq1_tok2, ...seq1_tokN],..., [seqK_tok1, seqK_tok2, ...seqK_tokZ]]
        # predicted_ids = pred_seq.tolist()

        # need to add additional dim to each target reference sequence in order to
        # convert to format needed by bleu_score function [ seq1=[ [reference1], [reference2] ], seq2=[ [reference1] ] ]
        # target_ids = torch.unsqueeze(trg_batch, 1).tolist()

        # bleu score needs two arguments
        # first: predicted_ids - list of predicted sequences as a list of predicted ids
        # second: target_ids - list of references (can be many, list)
        # bleu_score = plfunc.nlp.bleu_score(predicted_ids, target_ids, n_gram=3)
        # torch.unsqueeze(trg_batchT,1).tolist())

        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_sequence_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_precision_acc",
            precision,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_recall_acc",
            recall,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        # self.log(
        #     "val_bleu_idx",
        #     bleu_score,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     sync_dist=True,
        # )

        # return loss
        return {'loss': loss, 'acc': acc, 'precision': precision, 'recall': recall}

    def validation_step(self, batch, batch_idx):
        """validation is in eval mode so we do not have to use
        placeholder input tensors
        """
        loss, acc, precision, recall = self.calc_loss_and_metrics(batch, mode='val')

        # need to cast to list of predicted sequences (as list of token ids)   [ [seq1_tok1, seq1_tok2, ...seq1_tokN],..., [seqK_tok1, seqK_tok2, ...seqK_tokZ]]
        # predicted_ids = pred_seq.tolist()

        # need to add additional dim to each target reference sequence in order to
        # convert to format needed by bleu_score function [ seq1=[ [reference1], [reference2] ], seq2=[ [reference1] ] ]
        # target_ids = torch.unsqueeze(trg_batch, 1).tolist()

        # bleu score needs two arguments
        # first: predicted_ids - list of predicted sequences as a list of predicted ids
        # second: target_ids - list of references (can be many, list)
        # bleu_score = plfunc.nlp.bleu_score(predicted_ids, target_ids, n_gram=3)
        # torch.unsqueeze(trg_batchT,1).tolist())

        self.log(
            "val_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_sequence_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_precision_acc",
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_recall_acc",
            recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        # self.log(
        #     "val_bleu_idx",
        #     bleu_score,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     sync_dist=True,
        # )

        return loss, acc, precision, recall

    def test_step(self, batch, batch_idx):
        """validation is in eval mode so we do not have to use
        placeholder input tensors
        """
        loss, acc, precision, recall = self.calc_loss_and_metrics(batch, mode='test')

        # need to cast to list of predicted sequences (as list of token ids)   [ [seq1_tok1, seq1_tok2, ...seq1_tokN],..., [seqK_tok1, seqK_tok2, ...seqK_tokZ]]
        # predicted_ids = pred_seq.tolist()

        # need to add additional dim to each target reference sequence in order to
        # convert to format needed by bleu_score function [ seq1=[ [reference1], [reference2] ], seq2=[ [reference1] ] ]
        # target_ids = torch.unsqueeze(trg_batch, 1).tolist()

        # bleu score needs two arguments
        # first: predicted_ids - list of predicted sequences as a list of predicted ids
        # second: target_ids - list of references (can be many, list)
        # bleu_score = plfunc.nlp.bleu_score(predicted_ids, target_ids, n_gram=3)
        # torch.unsqueeze(trg_batchT,1).tolist())

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test_sequence_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test_precision_acc",
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test_recall_acc",
            recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        # self.log(
        #     "test_bleu_idx",
        #     bleu_score,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     sync_dist=True,
        # )

        return loss, acc, precision, recall


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)

    # look to .vscode/launch.json file - there are set some args
    parser = ArgumentParser()

    # add PROGRAM level args
    # parser.add_argument("--N_samples", type=int, default=256 * 10)
    parser.add_argument("--N_valid_size", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--exclude_eos", type=int, default=0, help="Whether to exclude EOS token for loss and accuracy calculation")
    parser.add_argument("--use_pred_eos", type=int, default=1, help="Whether to use predicted EOS as the last token")
    parser.add_argument("--use_pla", type=int, default=1, help="Whether to use PLA for orderless learning")
    parser.add_argument("--use_base_model", type=int, default=0, help="Whether to use multi-label classification instead of sequence model")
    parser.add_argument("--use_max_seq_len", type=int, default=0, help="Whether to use maximum possible sequence length for output prediction")
    parser.add_argument("--same_vocab_in_out", type=int, default=1, help="Whether to use same vocab for both input and output tokens")
    parser.add_argument("--train_data_path", type=str, default="./data/ar-training-data_050505_100.txt")
    parser.add_argument("--test_data_path", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="./results")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--debug", action='store_true')

    # add model specific args
    parser = Seq2SeqTrainer.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # change to debug settings if debug mode is passed
    args.max_epochs = 2 if args.debug else args.max_epochs
    args.log_dir = "train/debug" if args.debug else args.log_dir
    args.batch_size = 16 if args.debug else args.batch_size

    # configure logging on module level, redirect to file
    train_filename = os.path.basename(args.train_data_path)
    logdir = args.log_dir+'/'+train_filename.split('.')[0]
    os.makedirs(logdir, exist_ok=True)
    # Configure logging to save to a file
    log_file = os.path.join(logdir, "output.log")
    # logging.basicConfig(filename=log_file, level=logging.INFO)
    # logger = logging.getLogger("my_logger")
    # logging.getLogger("trainer.lightning.pytorch").setLevel(logging.DEBUG)
    # logger = logging.getLogger("trainer.lightning.pytorch")
    # logger.addHandler(logging.FileHandler(log_file))
    # logger.addHandler(logging.StreamHandler())

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w"),
        ],
        level=logging.INFO,
        # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        # datefmt="%Y-%m-%d %H:%M:%S",
    )
    # log_fh = open(log_file, 'a')
    logger = logging.getLogger(__name__)

    # get eval filename from training one if not provided
    args.test_data_path = args.test_data_path if args.test_data_path else args.train_data_path.replace('training', 'evaluation')

    dm = tl.ActivityRecDataModule(
        batch_size=args.batch_size,
        N_valid_size=args.N_valid_size,
        num_workers=args.num_workers,
        use_max_seq_len=args.use_max_seq_len,
        same_vocab_in_out=args.same_vocab_in_out,
        train_filename=args.train_data_path,
        test_filename=args.test_data_path,
        debug_mode=args.debug
    )

    # # dm = tl.SeqPairJsonDataModule(
    # #     path=args.dataset_path,
    # #     batch_size=args.batch_size,
    # #     n_samples=args.N_samples,
    # #     n_valid_size=args.N_valid_size,
    # #     num_workers=args.num_workers,
    # # )
    #
    # # with redirect_stdout(log_fh):
    # #     dm.prepare_data()

    # this will be run twice; once by us to keep vocab dim to be passed to model and once by trainer itself.
    dm.setup("fit")

    # to see results run in console
    # tensorboard --logdir tb_logs/
    # then open browser http://localhost:6006/

    # log_desc = f"RNN with attention model vocab_size={dm.vocab_size} data_size={dm.dims}, emb_dim={args.emb_dim} hidden_dim={args.hidden_dim}"
    input_dim = dm.input_vocab.n_words
    output_dim = dm.output_vocab.n_words
    log_desc = f"RNN with attention model input vocab_size={input_dim} output vocab siz={output_dim} emb_dim={args.emb_dim} hidden_dim={args.hidden_dim}"
    logging.info(log_desc)

    tb_logger = TensorBoardLogger(logdir, name="pl_tensorboard_logs", comment=log_desc )
    csv_logger = CSVLogger(logdir, name="csv_logs")

    from pytorch_lightning.callbacks import LearningRateMonitor

    lr_monitor = LearningRateMonitor(logging_interval="step")
    from pytorch_lightning.callbacks import ModelCheckpoint

    # saves top-K checkpoints based on "val_loss" metric
    best_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        # dirpath="my/path/",
        filename="best-{epoch:02d}-{val_loss:.2f}",
    )

    # saves last-K checkpoints based on "global_step" metric
    # make sure you log it inside your LightningModule
    last_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="step",
        mode="max",
        # dirpath="my/path/",
        filename="last-{epoch:02d}-{step}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=[csv_logger, tb_logger], callbacks=[lr_monitor, best_checkpoint, last_checkpoint]
    )  # , distributed_backend='ddp_cpu')

    model_args = vars(args)
    logging.info(pp.PrettyPrinter().pprint(model_args))
    model = Seq2SeqTrainer(input_vocab_size=input_dim, output_vocab_size=output_dim, padding_index=tl.PAD_token, **model_args)

    if args.resume_checkpoint:
        model = model.load_from_checkpoint(args.resume_checkpoint)

    # if an output file is provided in this run, update it in the model object.
    if args.output_file:
        model.output_file = os.path.join(logdir, args.output_file)

    # # most basic trainer, uses good defaults (1 gpu)
    trainer.fit(model, datamodule=dm)

    # Test the performance
    logging.info("\n-------Performing Testing on Provided Evaluation/Test File---------\n")
    dm.setup('test')
    print(f"Testing with the best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    trainer.test(model, ckpt_path='best', datamodule=dm)

# sample cmd

# python seq2seq_trainer.py --dataset_path /data/10k_sent_typos_wikipedia.jsonl \
# --gpus=2 --max_epoch=5 --batch_size=16 --num_workers=4 \
# --emb_dim=128 --hidden_dim=512 \
# --log_gpu_memory=True --weights_summary=full \
# --N_samples=1000000 --N_valid_size=10000 --distributed_backend=ddp --precision=16 --accumulate_grad_batches=4 --val_check_interval=640 --gradient_clip_val=2.0 --track_grad_norm=2

# tensorboard dev --logdir model_corrector/pl_tensorboard_logs/version??

