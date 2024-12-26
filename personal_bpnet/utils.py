"""
Data loader for chunked data for CLIPNET.
"""

import random

import numpy as np
import pyfastx
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset, Sampler


def twohot_encode(seq):
    """
    Allows you to access id, seq, and twohot(seq) as attributes. Handles IUPAC ambiguity
    codes for heterozygotes.
    """
    seq_list = list(seq.upper())
    encoding = {
        "A": np.array([2, 0, 0, 0]),
        "C": np.array([0, 2, 0, 0]),
        "G": np.array([0, 0, 2, 0]),
        "T": np.array([0, 0, 0, 2]),
        "N": np.array([0, 0, 0, 0]),
        "M": np.array([1, 1, 0, 0]),
        "R": np.array([1, 0, 1, 0]),
        "W": np.array([1, 0, 0, 1]),
        "S": np.array([0, 1, 1, 0]),
        "Y": np.array([0, 1, 0, 1]),
        "K": np.array([0, 0, 1, 1]),
    }
    twohot = [encoding.get(seq, seq) for seq in seq_list]
    return np.array(twohot).swapaxes(0, 1) / 2


def reverse_complement_twohot(seq_twohot):
    """
    Computes reverse-complement twohot. Handles heterozygotes encoded via IUPAC
    ambiguity codes.

    seqs_twohot should be (4, n) where n is the length of the sequence.
    """
    # inverting each sequence in rc along both axes takes the reverse complement.
    # Except for the at and cg heterozygotes, which need to be complemented by masks.
    rc = seq_twohot[::-1, ::-1].swapaxes(0, 1)
    # Get mask of all at and cg heterozygotes
    at = np.all(rc == [0.5, 0, 0, 0.5], axis=1)
    cg = np.all(rc == [0, 0.5, 0.5, 0], axis=1)
    # Complement at and cg heterozygotes
    rc[at] = [0, 0.5, 0.5, 0]
    rc[cg] = [0.5, 0, 0, 0.5]
    return rc.swapaxes(0, 1)


class ChunkedDataset(Dataset):
    def __init__(
        self,
        seq_chunks,
        signal_chunks,
        in_window=2114,
        out_window=1000,
        reverse_complement=True,
        jitter=True,
    ):
        self.seq_chunks = seq_chunks
        self.signal_chunks = signal_chunks
        if len(self.seq_chunks) != len(self.signal_chunks):
            raise ValueError(
                "len(seq_chunks) != len(signal_chunks) (%d, %d)"
                % (len(self.seq_chunks), len(self.signal_chunks))
            )
        self.in_window = in_window
        self.out_window = out_window
        self.jitter = jitter
        self.reverse_complement = reverse_complement
        self.chunk_indices = list(range(len(self.seq_chunks)))
        self.chunk_lengths = [
            len(pyfastx.Fasta(chunk))
            for chunk in tqdm.tqdm(seq_chunks, desc="Calculating dataset length")
        ]
        self.chunk_data_indices = None
        self.current_chunk_data = None
        self.current_chunk_signals = None
        self.current_chunk_index = -1

    def __len__(self):
        return sum(self.chunk_lengths)

    def __getitem__(self, idx):
        # Load the chunk if it is not already loaded
        chunk_idx, data_idx = self.chunk_data_indices[idx]
        if chunk_idx != self.current_chunk_index:
            self.current_chunk_seq, self.current_chunk_signal = self._load_chunk(
                chunk_idx
            )
            self.current_chunk_index = chunk_idx
        # Get the data. We do the twohot encoding here to save memory.
        X = twohot_encode(self.current_chunk_seq[data_idx])
        y = self.current_chunk_signal[data_idx, :, :]

        # If self.jitter == True, randomly select a region of length self.in/out_window
        # else, take the middle subsequence of length self.in/out_window
        pad = (X.shape[-1] - self.in_window) // 2
        j = random.randint(0, pad * 2) if self.jitter else pad
        X = X[:, j : j + self.in_window]
        y = y[:, j : j + self.out_window]

        # If self.reverse_complement == True, randomly reverse complement the sequence
        if self.reverse_complement and random.random() > 0.5:
            X = reverse_complement_twohot(X)
            y = y[::-1, ::-1]

        # Convert to tensors
        X_torch = torch.from_numpy(X.copy()).to(torch.float)
        y_torch = torch.from_numpy(y.copy()).to(torch.float)
        return X_torch, y_torch

    def _load_chunk(self, idx):
        seq_file = self.seq_chunks[idx]
        signal_file = self.signal_chunks[idx]
        seqs = [rec.seq for rec in pyfastx.Fasta(seq_file)]
        signals = np.load(signal_file)["arr_0"]
        return seqs, signals

    def set_chunk_data_indices(self, chunk_data_indices):
        self.chunk_data_indices = chunk_data_indices


class ChunkSampler(Sampler):
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset

    def __iter__(self):
        # shuffle chunk order
        random.shuffle(self.dataset.chunk_indices)

        # shuffle data order within chunks
        chunk_data_indices = []
        for chunk_idx in self.dataset.chunk_indices:
            chunk_length = self.dataset.chunk_lengths[chunk_idx]
            shuffled_indices = list(range(chunk_length))
            random.shuffle(shuffled_indices)
            chunk_data_indices.extend([(chunk_idx, i) for i in shuffled_indices])

        self.dataset.set_chunk_data_indices(chunk_data_indices)
        return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)


class ChunkedDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=64,
        batch_sampler=None,
        # num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        # multiprocessing_context=None,
        generator=None,
        # prefetch_factor=None,
        persistent_workers=False,
    ):
        sampler = ChunkSampler(dataset, batch_size=batch_size)
        super().__init__(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            batch_sampler=batch_sampler,
            # num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            # multiprocessing_context=multiprocessing_context,
            generator=generator,
            # prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )


class BedGenerator(torch.utils.data.Dataset):
    """A data generator for BPNet inputs.

    This generator takes in an extracted set of sequences, output signals,
    and control signals, and will return a single element with random
    jitter and reverse-complement augmentation applied. Jitter is implemented
    efficiently by taking in data that is wider than the in/out windows by
    two times the maximum jitter and windows are extracted from that.
    Essentially, if an input window is 1000 and the maximum jitter is 128, one
    would pass in data with a length of 1256 and a length 1000 window would be
    extracted starting between position 0 and 256. This  generator must be
    wrapped by a PyTorch generator object.

    Parameters
    ----------
    sequences: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
            A one-hot encoded tensor of `n` example sequences, each of input
            length `in_window`. See description above for connection with jitter.

    signals: torch.tensor, shape=(n,)
            The signals to predict. A single scalar

    in_window: int, optional
            The input window size. Default is 2114.

    max_jitter: int, optional
            The maximum amount of jitter to add, in either direction, to the
            midpoints that are passed in. Default is 0.

    reverse_complement: bool, optional
            Whether to reverse complement-augment half of the data. Default is False.

    random_state: int or None, optional
            Whether to use a deterministic seed or not.
    """

    def __init__(
        self,
        sequences,
        signals,
        in_window=2114,
        max_jitter=0,
        reverse_complement=False,
        random_state=None,
    ):
        self.in_window = in_window
        self.max_jitter = max_jitter

        self.reverse_complement = reverse_complement
        self.random_state = np.random.RandomState(random_state)

        self.signals = signals
        self.sequences = sequences
        if signals.shape[0] != sequences.shape[0]:
            raise ValueError(
                f"({sequences.shape[0]}) sequences and ({signals.shape[0]}) signals"
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        i = self.random_state.choice(len(self.sequences))
        j = (
            0
            if self.max_jitter == 0
            else self.random_state.randint(self.max_jitter * 2)
        )

        X = self.sequences[i][:, j : j + self.in_window]
        y = self.signals[i]

        if self.reverse_complement and self.random_state.choice(2) == 1:
            X = torch.flip(X, [0, 1])

        return X, y
