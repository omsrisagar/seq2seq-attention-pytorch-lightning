import os
import sys
import datetime as dt
import json

#import dill # pickle lambda https://stackoverflow.com/a/25353243/75037
import pickle
import itertools as itr
import random


import pytorch_lightning as pl
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler

from torchnlp.samplers import BucketBatchSampler, SortedSampler
from torchnlp.encoders.text import CharacterEncoder, SubwordEncoder
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors


from torchnlp.encoders.text import StaticTokenizerEncoder
from tokenize_utils import (
    uni_bi_grams_vocab_gen,
    bigrams_tokenize,
)

PAD_token = 0
UNK_token = 1
EOS_token = 2
SOS_token = 3

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "pad_indx", 1: "unk", 2: "EOS", 3: "SOS"}
        self.n_words = 4  # Count above
        self.max_seq_len = 0 # count max. seq. length; need to add 2 to this!

    def filter_word(self, word):
        "Removes HISTORY in the beginning if exists"
        if '_' not in word:
            return word
        first, second = word.split('_', 1)
        return second if 'HISTORY' in first else word

    def addSequence(self, sequence):
        words = sequence.split(',')
        words = [self.filter_word(word) for word in words if word and not word.isspace()]
        if len(words) > self.max_seq_len:
            self.max_seq_len = len(words)
        for word in words:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class SequenceDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        # Sanity check
        for pair in self.pairs:
            assert pair[0][0].data == 3 # start index should be SOS
            assert pair[0][-1].data == 2 # end index needs to be EOS
            assert pair[1][0].data == 3 # same
            assert pair[1][-1].data == 2 # same
            # there should be no 0s
            pair0_indices = torch.nonzero(pair[0] == 0)
            assert len(pair0_indices) == 0
            pair1_indices = torch.nonzero(pair[1] == 0)
            assert len(pair1_indices) == 0
            # there should be no 1s
            pair0_indices = torch.nonzero(pair[0] == 1)
            assert len(pair0_indices) == 0
            pair1_indices = torch.nonzero(pair[1] == 1)
            assert len(pair1_indices) == 0

        self.transform = transform

    def __getitem__(self, index):
        # This method should return only 1 sample and label
        # (according to "index"), not the whole dataset
        # So probably something like this for you:

        input_seq, output_seq = self.pairs[index]
        return {'in_seq': input_seq, 'out_seq': output_seq}

    def __len__(self):
        return len(self.pairs)

class ActivityRecDataModule(pl.LightningDataModule):
    '''simple pytroch-lightning data module witch generates artificial data,
    generate simple translation task from permuted alphabet to normal alphabet as
    list of dicts
     [
         { "correct" : "...", "incorrect": "..."}
         { "correct" : "...", "incorrect": "..."}
     ]
    '''

    def __init__(
            self, batch_size=4, N_valid_size=200, num_workers=1, train_filename=None, test_filename=None
    ):
        super().__init__()

        self.batch_size = batch_size
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.N_valid_size = N_valid_size
        self.num_workers = num_workers


    def read_file(self, filename):
        print(f"Reading lines from {filename}...")

        # Read the file and split into lines
        lines = open(filename).read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [l.split('#') for l in lines]
        print("Read %s sequence pairs" % len(pairs))
        input_vocab = Vocab('activity_recognition_vocab')
        # output_vocab = Vocab('activity_recognition_output')
        output_vocab = input_vocab # since both input and output have activities

        return input_vocab, output_vocab, pairs

    def gen_sequences(self, filename):
        """generates input and output sequences from file
        """
        input_vocab, output_vocab, pairs = self.read_file(filename)
        print(f"Counting words from {filename}...")
        for pair in pairs:
            input_vocab.addSequence(pair[0])
            output_vocab.addSequence(pair[1])
        print("Counted words:")
        print(input_vocab.name, input_vocab.n_words)
        print(output_vocab.name, output_vocab.n_words)
        return input_vocab, output_vocab, pairs

    def indexes_from_sequence(self, vocab, sequence):
        words = sequence.split(',')
        words = [vocab.filter_word(word) for word in words if word and not word.isspace()]
        return [SOS_token] + [vocab.word2index[word] for word in words]

    def tensor_from_sequence(self, vocab, sequence):
        indexes = self.indexes_from_sequence(vocab, sequence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long)

    def tensors_from_pair(self, pair, input_vocab, output_vocab):
        input_tensor = self.tensor_from_sequence(input_vocab, pair[0])
        target_tensor = self.tensor_from_sequence(output_vocab, pair[1])
        return (input_tensor, target_tensor)

    def gen_tensors(self, pairs):
        "Generates indexes and then tensors from data"
        return [self.tensors_from_pair(pair, self.input_vocab, self.output_vocab) for pair in pairs]

        # n = len(pairs)
        # # MAX_LENGTH = max(input_vocab.max_seq_len, output_vocab.max_seq_len)
        # input_ids = np.zeros((n, input_vocab.max_seq_len), dtype=np.int32)
        # target_ids = np.zeros((n, output_vocab.max_seq_len), dtype=np.int32)
        #
        # for idx, (inp, tgt) in enumerate(pairs):
        #     inp_ids = self.indexes_from_sequence(input_vocab, inp)
        #     tgt_ids = self.indexes_from_sequence(output_vocab, tgt)
        #     inp_ids.append(EOS_token)
        #     tgt_ids.append(EOS_token)
        #     input_ids[idx, :len(inp_ids)] = inp_ids
        #     target_ids[idx, :len(tgt_ids)] = tgt_ids
        #
        # return TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(target_ids))

    def prepare_data(self):
        # stuff here is done once at the very beginning of training
        # before any distributed training starts

        # https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html#prepare-data

        pass

    def setup(self, stage):
        # https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html#setup

        # self.train_full_dataset = self.gen_tensors(train_pairs, self.input_vocab, self.output_vocab)
        # self.test_dataset = self.gen_tensors(val_pairs, self.input_vocab, self.output_vocab)
        if stage == 'fit':
            self.input_vocab, self.output_vocab, train_pairs = self.gen_sequences(self.train_filename) # input and output vocab are same objects
            print("Example train pair: raw and tensor formats")
            print(random.choice(train_pairs))
            tensor_train_pairs = self.gen_tensors(train_pairs)
            print(random.choice(tensor_train_pairs))
            train_dataset = SequenceDataset(tensor_train_pairs)
            self.train_data, self.val_data = random_split(train_dataset, [1-self.N_valid_size, self.N_valid_size])

        if stage == 'test':
            _, _, test_pairs = self.gen_sequences(self.test_filename)
            print("Example test pair: raw and tensor formats")
            print(random.choice(test_pairs))
            tensor_test_pairs = self.gen_tensors(test_pairs)
            print(random.choice(tensor_test_pairs))
            self.test_data = SequenceDataset(tensor_test_pairs)

    def __collate_fn(self, sample: list, prepare_target=True):
        """
        torch.utils.Dataloader collate_fn

        change layout of data from list of dicts to dict of tensors
         [
           {text: 'a', label:'0'}
           {text: 'b', label:'1'}
           {text: 'c', label:'2'}
         ]
         to
         { text: ['a', 'b', 'c'], label:[0,1,2] }

         and encode tokens to its ids in vocab, do also 0 padding
        """

        # sort in reverse order, need for packed sequence

        sorted_sample = sorted(sample, key=lambda x: -len(x["in_seq"]))

        collate_sample = collate_tensors(
            sorted_sample, stack_tensors=stack_and_pad_tensors
        )

        src_tokens, src_lengths = collate_sample['in_seq'].tensor, collate_sample['in_seq'].lengths

        # cant change layout here, becaure when use distributeddataloader (multi-gpu) it will
        # divide first dim by the number of gpus,
        # change from [batch, seq_len] -> to [seq_len, batch]
        # src_tokens = src_tokens.transpose(0, 1)

        inputs = {"src_ids": src_tokens, "src_lengths": src_lengths}

        ### encode tokens based on vocab
        trg_tokens, trg_lengths = collate_sample['out_seq'].tensor, collate_sample['out_seq'].lengths

        # change from [batch, seq_len] -> to [seq_len, batch]
        # trg_tokens = trg_tokens.transpose(0, 1)
        targets = {"trg_ids": trg_tokens, "trg_lengths": trg_lengths}

        return inputs, targets

    def train_dataloader(self):

        # dataloader with sampler for distributed training trainer should have set replace_sampler_ddp=False
        self._train_dl = DataLoader(
            dataset=self.train_data,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.__collate_fn,
            batch_size=self.batch_size,
        )

        return self._train_dl

    def val_dataloader(self):

        # with normal sampler
        self._val_dl = DataLoader(
            dataset=self.val_data,
            collate_fn=self.__collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return self._val_dl

    def test_dataloader(self):

        # with normal sampler
        self._test_dl = DataLoader(
            dataset=self.test_data,
            collate_fn=self.__collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return self._test_dl

class ABCSec2SeqDataModule(pl.LightningDataModule):
    '''simple pytroch-lightning data module witch generates artificial data,
    generate simple translation task from permuted alphabet to normal alphabet as 
    list of dicts 
     [
         { "correct" : "...", "incorrect": "..."}
         { "correct" : "...", "incorrect": "..."}
     ]
    '''
    def __init__(
        self, batch_size=4, N_random_samples=1000, N_valid_size=200, num_workers=1
    ):
        super().__init__()

        self.batch_size = batch_size

        self.vocab_size = -1

        self.padding_index = -1

        assert N_random_samples > N_valid_size

        self.N_random_samples = N_random_samples
        self.N_valid_size = N_valid_size
        self.num_workers = num_workers

    def prepare_data(self):
        # stuff here is done once at the very beginning of training
        # before any distributed training starts

        # https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html#prepare-data

        pass

    def _setup_task1(self, N_random_samples):
        """generate simple translation task from permuted alphabet to normal alphabet.
        Fixed length only permuted characters
        """

        # If you want to play with it:
        # - you can try shorter vocab for faster trainning eg. 'abcdefghij' (ony 10 chars)
        init_string = "abcdefghijklmnopqrstuwxyz"  

        dataset = []
        for i in range(N_random_samples):

            l = list(init_string)
            random.shuffle(l)
            dataset.append(
                {"correct": f"{i}-{init_string}", "incorrect": f'{i}-{"".join(l)}'}
            )

        return dataset

    def _setup_task2(self, N_random_samples):
        """generates simple translation task random length, random letters"""

        init_string = "abcdefghijklmnoprstuwxyz"
        str_len = len(init_string)

        dataset = []
        for i in range(N_random_samples):
            # random seq len [3, str_len]
            rnd_len = random.randint(3, str_len)
            # random characters choosen with replacement
            t = random.choices(init_string, k=rnd_len)
            t_sort = sorted(t)

            dataset.append({"correct": "".join(t_sort), "incorrect": "".join(t)})

        return dataset

    def _bucket_train_sort_func(self, i):
        """defines sort key for bucketBatchSampler, should be defined in top scope because
        in distributed mode lambda can't be pickled
        """
        return -len(self.train_ds[i]["incorrect"])

    def _bucket_val_sort_func(self, i):
        """defines sort key for bucketBatchSampler, should be defined in top scope because
        in distributed mode lambda can't be pickled
        """
        return -len(self.valid_ds[i]["incorrect"])

    def _sampler_sort_func(self, x):
        """
        function for SortedSampler, sort in reverse orderby incorrect sequence lenght,
        added random value for hashing the rows in distributed scenario, each epoch get slight different
        set of sentences
        sequences arent sorted exacly, but it does not matter much,
        """

        return -len(x["incorrect"]) + random.randint(0, 4)

    def setup(self, stage):
        # https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html#setup

        N_valid_size = self.N_valid_size

        # dataset = self._setup_task1(self.N_random_samples)
        dataset = self._setup_task2(self.N_random_samples)

        # list of dicts
        self.train_ds = dataset[0:-N_valid_size]

        self.valid_ds = dataset[-N_valid_size:]

        # load dataset build vocab and numericalize

        # todo: change it bad design! only for prototyping and learning
        dataset_example_gen = (ex["correct"] + " " + ex["incorrect"] for ex in dataset)

        
        self.tokenizer = CharacterEncoder(
            dataset_example_gen, append_eos=True, append_sos=True
        )
        pickle.dump(
            self.tokenizer,
            open(f"./abc_data_character_encoder.p", "wb"),
        )

        self.train_sampler = SortedSampler(
            self.train_ds, sort_key=self._sampler_sort_func
        )

        self.val_sampler = SortedSampler(
            self.valid_ds, sort_key=self._sampler_sort_func
        )

        # #samplers from torchnlp, did not work with distibutedDataParallel
        # self.train_sampler = BucketBatchSampler(
        #     sampler=SequentialSampler(self.train_ds),
        #     # bucket_size_multiplier=1000,
        #     batch_size=self.batch_size,
        #     drop_last=True,
        #     sort_key=self._bucket_train_sort_func,
        #     #sort_key=lambda i: -len(self.train_ds[i]["incorrect"]),
        # )

        # self.val_sampler = BucketBatchSampler(
        #     sampler=SequentialSampler(self.valid_ds),
        #     batch_size=self.batch_size,
        #     drop_last=True,
        #     sort_key = self._bucket_val_sort_func,
        #     #sort_key=lambda i: -len(self.valid_ds[i]["incorrect"]),
        # )

        # samplers from catalyst
        # DistributedWrapperSampler
        # DynamicBatchLensampler
        # https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py

        # DynamicLenBatchSampler, DistributedSamplerWrapper

        # train_sampler = RandomSampler(self.train_ds)
        # train_sampler = DynamicLenBatchSampler(train_sampler, self.batch_size, drop_last=True)

        # self.train_sampler = train_sampler
        # self.train_sampler = DistributedSamplerWrapper(train_sampler)

        # valid_sampler = RandomSampler(self.valid_ds)
        # valid_sampler = DynamicLenBatchSampler(valid_sampler, self.batch_size, drop_last=True)
        # self.val_sampler = valid_sampler
        # self.valid_sampler = DistributedSamplerWrapper(valid_sampler)

        ### todo: do wymiany
        self.vocab_size = self.tokenizer.vocab_size
        self.padding_index = self.tokenizer.padding_index  # =0

    def __collate_fn(self, sample: list, prepare_target=True):
        """
        torch.utils.Dataloader collate_fn

        change layout of data from list of dicts to dict of tensors
         [
           {text: 'a', label:'0'}
           {text: 'b', label:'1'}
           {text: 'c', label:'2'}
         ]
         to
         { text: ['a', 'b', 'c'], label:[0,1,2] }

         and encode tokens to its ids in vocab, do also 0 padding
        """

        # sort in reverse order, need for packed sequence

        sorted_sample = sorted(sample, key=lambda x: -len(x["incorrect"]))

        collate_sample = collate_tensors(
            sorted_sample, stack_tensors=stack_and_pad_tensors
        )

        ### todo: do wymiany
        src_tokens, src_lengths = self.tokenizer.batch_encode(
            collate_sample["incorrect"]
        )

        # cant change layout here, becaure when use distributeddataloader (multi-gpu) it will
        # divide first dim by the number of gpus,
        # change from [batch, seq_len] -> to [seq_len, batch]
        # src_tokens = src_tokens.transpose(0, 1)

        inputs = {"src_ids": src_tokens, "src_lengths": src_lengths}

        ### todo: do wymiany
        ### encode tokens based on vocab
        trg_tokens, trg_lengths = self.tokenizer.batch_encode(collate_sample["correct"])

        # change from [batch, seq_len] -> to [seq_len, batch]
        # trg_tokens = trg_tokens.transpose(0, 1)
        targets = {"trg_ids": trg_tokens, "trg_lengths": trg_lengths}

        return inputs, targets

    def train_dataloader(self):

        # dataloader with sampler for distributed training trainer should have set replace_sampler_ddp=False
        self._train_dl = DataLoader(
            dataset=self.train_ds,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=self.train_sampler,
            collate_fn=self.__collate_fn,
            batch_size=self.batch_size,
        )

        return self._train_dl

    def val_dataloader(self):

        # with normal sampler
        self._val_dl = DataLoader(
            dataset=self.valid_ds,
            collate_fn=self.__collate_fn,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return self._val_dl


class SeqPairJsonDataModule(pl.LightningDataModule):
    '''
    Data module for reading json line file in format

    { "correct" : "...", "incorrect": "..."}
    { "correct" : "...", "incorrect": "..."}
    { "correct" : "...", "incorrect": "..."}

    '''
    def __init__(
        self, path, batch_size=4, num_workers=2, n_samples=-1, n_valid_size=10000
    ):
        super().__init__()

        self.batch_size = batch_size
        self.json_path = path

        self.vocab_size = -1
        self.padding_index = -1

        self.num_workers = num_workers
        self.n_samples = n_samples if n_samples > 0 else None

        # default last 10k data for validation
        self.valid_split_size = n_valid_size

        # sequences longer then this will be ignored
        self._max_seq_len = 400


        # this is the place where you can plug in your tokenizer
        self.tokenizer = CharTokenizerEncoder()
        #self.tokenizer = BiGramTokenizerEncoder()

    def prepare_data(self):
        # stuff here is done once at the very beginning of training
        # before any distributed training starts
        # https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html#prepare-data
        pass

    def _sampler_sort_func(self, x):
        """
        function for SortedSampler, sort in reverse orderby incorrect sequence lenght,
        added random value for hashing the rows in distributed scenario, each epoch get slight different
        set of sentences
        sequences arent sorted exacly, but it does not matter much,
        """
        # sort data by "incorrect" sequence lenght, add some noise to sorting, little hack :)
        return -len(x["incorrect"]) + random.randint(0, 5)

    def setup(self, stage):
        # https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html#setup

        self._dataset = []
        max_seq_len = 0
        skipped_examples = 0
        with open(self.json_path) as json_file:

            for line in itr.islice(json_file, self.n_samples):
                json_obj = json.loads(line)
                del json_obj["errors"]

                if len(json_obj["incorrect"]) > self._max_seq_len:
                    # omit to long sequence
                    skipped_examples += 1
                    continue

                self._dataset.append(json_obj)
                # compute some dataset stats
                max_seq_len = max(
                    max_seq_len, len(json_obj["correct"]), len(json_obj["incorrect"])
                )

        ds_len = len(self._dataset)
        self.dims = (ds_len, max_seq_len)

        stats = {
            "dataset_len": ds_len,
            "max_seq_len": max_seq_len,
            "skiped_examples": skipped_examples,
        }

        self.tokenizer.train(
            self._dataset, append_eos=True, append_sos=True, min_occurrences=1000
        )

        # bad, bad, hardcoded path, possible pull request
        self.tokenizer.save_vocab("model_corrector/")

        self.vocab_size = self.tokenizer.vocab_size
        self.padding_index = self.tokenizer.padding_index  # =0

        dataset_len = len(self._dataset)

        assert_msg = (
            f"lenght of all gathered dataset examples is {dataset_len}, it is less than validation split."
            + f"Try to increase self._max_seq_len={self._max_seq_len} or decrease self.valid_split_size={self.valid_split_size}"
        )
        assert dataset_len > self.valid_split_size, assert_msg

        last_idx = dataset_len - self.valid_split_size

        # list of dicts
        self.valid_ds = self._dataset[last_idx:]

        self.train_ds = self._dataset[0:last_idx]
        # random.shuffle(self.train_ds)

        self.train_sampler = SortedSampler(
            self.train_ds, sort_key=self._sampler_sort_func
        )

        self.val_sampler = SortedSampler(
            self.valid_ds, sort_key=self._sampler_sort_func
        )

        return stats

    def __collate_fn(self, sample: list, prepare_target=True):
        """
        torch.utils.Dataloader collate_fn

        change layout of data from list of dicts to dict of tensors
         [
           {correct: 'a', incorrect:'b'}
           {correct: 'b', incorrect:'a'}
           {correct: 'c', incorrect:'b'}
         ]
         to
         { correct: ['a', 'b', 'c'], incorrect:['b', 'a', 'b'] }

         and encode tokens to its ids in vocab, do also 0 padding
        """

        # sort in reverse order, need for packed sequence

        sorted_sample = sorted(sample, key=lambda x: -len(x["incorrect"]))

        collate_sample = collate_tensors(
            sorted_sample, stack_tensors=stack_and_pad_tensors
        )
        src_tokens, src_lengths = self.tokenizer.encode_batch(
            collate_sample["incorrect"]
        )

        # cant change layout here, because when use distributeddataloader (multi-gpu) it will
        # divide first dim by the number of gpus,
        # change from [batch, seq_len] -> to [seq_len, batch]
        # src_tokens = src_tokens.transpose(0, 1)

        inputs = {"src_ids": src_tokens, "src_lengths": src_lengths}

        ### encode tokens based on vocab
        trg_tokens, trg_lengths = self.tokenizer.encode_batch(collate_sample["correct"])

        # change from [batch, seq_len] -> to [seq_len, batch]
        # trg_tokens = trg_tokens.transpose(0, 1)
        targets = {"trg_ids": trg_tokens, "trg_lengths": trg_lengths}

        return inputs, targets

    def train_dataloader(self):

        # dataloader with sampler for distributed training trainer should have set replace_sampler_ddp=False
        self._train_dl = DataLoader(
            dataset=self.train_ds,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=self.train_sampler,
            collate_fn=self.__collate_fn,
            batch_size=self.batch_size,
        )

        return self._train_dl

    def val_dataloader(self):

        # with normal sampler
        self._val_dl = DataLoader(
            dataset=self.valid_ds,
            collate_fn=self.__collate_fn,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return self._val_dl


from abc import ABC, abstractmethod


class TokenizerEncoder(ABC):
    def __init__(self):
        self.tokenizer_name = "abstract_corrector_tokenizer"
        self.tokenizer = None

    @abstractmethod
    def train(
        self,
        pair_dataset,
        append_eos=True,
        append_sos=True,
        min_occurrences=1000,
        **kwargs,
    ):

        pass

    @abstractmethod
    def encode_batch(self, samples):
        pass

    def save_vocab(self, folder_path):
        """pickle dump tokenizer"""

        pickle.dump(
            self.tokenizer,
            open(os.path.join(folder_path, f"{self.tokenizer_name}.p"), "wb+"),
        )


class CharTokenizerEncoder(TokenizerEncoder):
    def __init__(self):

        self.vocab_size = None
        self.padding_index = None

        # determine how many sequences we take to build the vocabulary
        self._tokenizer_max_seq = 3 * 10 ** 5

        # pickle tokenizer file name
        self.tokenizer_name = "char_corrector_tokenizer"

    def train(
        self, pair_dataset, append_eos=True, append_sos=True, min_occurrences=1000
    ):
        """train a tokenizer"""

        # create generator based on slice of data (3*10^5 sentences)
        dataset_example_gen = (
            ex["correct"] + " " + ex["incorrect"]
            for ex in itr.islice(pair_dataset, self._tokenizer_max_seq)
        )

        self.tokenizer = CharacterEncoder(
            dataset_example_gen, append_eos=True, append_sos=True, min_occurrences=1000
        )

        # after training set the variables
        self.vocab_size = self.tokenizer.vocab_size
        self.padding_index = self.tokenizer.padding_index  # =0

    def encode(self, text):
        pass

    def encode_batch(self, samples):
        """
        Encodes list of strings

        Args:
        -----------
        samples: list of strings
        """

        # it is compatible with pytrochNLP
        tokens, lengths = self.tokenizer.batch_encode(samples)

        return tokens, lengths

    def decode(self, text):
        pass





class BiGramTokenizerEncoder(TokenizerEncoder):
    def __init__(self):

        self.vocab_size = None
        self.padding_index = None
        self.ngrams = 2

        # determine how many sequences we take to build the vocabulary
        self._tokenizer_max_seq = 3 * 10 ** 5
        self.tokenizer_name = "bigram_corrector_tokenizer"

    def train(
        self, pair_dataset, append_eos=True, append_sos=True, min_occurrences=300
    ):
        """train a tokenizer"""

        # create genartor for incorrect data only
        dataset_example_gen = (
            ex["incorrect"]
            for ex in itr.islice(pair_dataset, self._tokenizer_max_seq)
        )

        self.tokenizer = StaticTokenizerEncoder(
            dataset_example_gen,
            min_occurrences=min_occurrences,
            append_eos=append_eos,
            append_sos=append_sos,
            tokenize=uni_bi_grams_vocab_gen,
            detokenize=self._detokenize #lambda x: "".join(x),  # concat all tokens
        )

        self.tokenizer.tokenize =bigrams_tokenize #ngram_tokenizer(self.ngrams)

        # after training set the variables
        self.vocab_size = self.tokenizer.vocab_size
        self.padding_index = self.tokenizer.padding_index  # =0

    def _detokenize(self, tokens):
        return "".join(tokens)



    def encode(self, text):
        pass

    def encode_batch(self, samples):
        """
        Encodes list of strings

        Args:
        -----------
        samples: list of strings
        """

        # it is compatible with pytrochNLP
        tokens, lengths = self.tokenizer.batch_encode(samples)

        return tokens, lengths

    def decode(self, text):
        pass
