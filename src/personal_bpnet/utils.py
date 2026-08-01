# utils.py
# Author: Adam He <adamyhe@gmail.com>

"""
A bunch of utility functions for working with personal genome sequences.
Notably includes a data loader for loading chunked data to train CLIPNET.
"""

import random

import matplotlib.pyplot as plt
import numba
import numpy as np
import pyfaidx
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset, Sampler

# IUPAC ambiguity codes for heterozygotes, in (A, C, G, T) channel order. Values
# are pre-halved so `twohot_encode` needs no division at call time.
_TWOHOT_CODES = {
    "A": (1.0, 0, 0, 0),
    "C": (0, 1.0, 0, 0),
    "G": (0, 0, 1.0, 0),
    "T": (0, 0, 0, 1.0),
    "N": (0, 0, 0, 0),
    "M": (0.5, 0.5, 0, 0),
    "R": (0.5, 0, 0.5, 0),
    "W": (0.5, 0, 0, 0.5),
    "S": (0, 0.5, 0.5, 0),
    "Y": (0, 0.5, 0, 0.5),
    "K": (0, 0, 0.5, 0.5),
}
# Byte-indexed lookup table (0-255) so encoding is a gather from a small table
# instead of per-character dict lookups; unrecognized bytes (e.g. "-" gaps or the
# rest of the IUPAC alphabet) fall back to the all-zero row, same as "N".
_TWOHOT_LOOKUP = np.zeros((256, 4), dtype=np.float64)
for _base, _code in _TWOHOT_CODES.items():
    _TWOHOT_LOOKUP[ord(_base)] = _code
    _TWOHOT_LOOKUP[ord(_base.lower())] = _code


@numba.njit(cache=True)
def _twohot_gather(codes, lookup, out):
    """Gathers `lookup[codes[n, i]]` into `out[n, :, i]`.

    `codes` is (n_seqs, seq_len) ASCII byte codes and `out` is the
    (n_seqs, 4, seq_len) output buffer to fill in place. A compiled loop beats
    `lookup[codes]`-style numpy fancy-indexing here (~5x faster), and lets a
    single call encode an entire batch of equal-length sequences at once
    instead of looping in Python over one sequence at a time.
    """
    for n in range(codes.shape[0]):
        for i in range(codes.shape[1]):
            for j in range(4):
                out[n, j, i] = lookup[codes[n, i], j]


def twohot_encode(seq):
    """
    Calculates a two-hot encoding of a given DNA sequence. Handles IUPAC ambiguity
    codes for heterozygotes. IMPORTANT: Note that this script halves the encoding
    for compatibility with models/methods that require one-hot encoded sequences.
    Heterozygous positions are represented as 0.5 pairs. This differs from the
    original implementation used in CLIPNET tensorflow, which has double values.
    """
    codes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8).reshape(1, -1)
    out = np.empty((1, 4, codes.shape[1]), dtype=_TWOHOT_LOOKUP.dtype)
    _twohot_gather(codes, _TWOHOT_LOOKUP, out)
    return out[0]


def reverse_complement_twohot(seq_twohot):
    """
    Computes reverse-complement twohot. Handles heterozygotes encoded via IUPAC
    ambiguity codes.

    seqs_twohot should be (4, n) where n is the length of the sequence.
    """
    # Reversing the channel axis (order A, C, G, T) swaps A<->T and C<->G, i.e.
    # complementation; reversing the position axis reverses the strand direction.
    # This also correctly self-complements the A/T and C/G heterozygote codes
    # (their two channels are symmetric around the reversal), so no special-casing
    # is needed. `.copy()` avoids returning a view aliased to the input, which
    # would let an in-place assignment on the result silently mutate the input.
    return seq_twohot[::-1, ::-1].copy()


def get_twohot_fasta_sequences(fasta_fp):
    """
    Given a fasta file where every record is the same length, returns a
    twohot-encoded array of shape (n, 4, len) of all sequences.

    All records are encoded in a single batched call to `_twohot_gather` rather
    than looping over sequences one at a time. Note: this function used to
    optionally parallelize per-sequence encoding across processes, back when
    each call to `twohot_encode` did a Python-level per-character loop. Now that
    encoding is a single compiled gather, a single sequence encodes in a few
    microseconds — well below the per-task IPC/pickling overhead of a process
    pool, and batching removes the per-sequence Python overhead entirely, so
    multiprocessing (measured to be over an order of magnitude *slower* here)
    was removed rather than reimplemented.
    """
    fa = pyfaidx.Fasta(fasta_fp)
    seqs = [str(rec) for rec in fa]
    lengths = {len(seq) for seq in seqs}
    if len(lengths) != 1:
        raise ValueError(
            "All sequences must be the same length to be stacked into one "
            f"array; found lengths {sorted(lengths)}."
        )

    length = lengths.pop()
    codes = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(
        len(seqs), length
    )
    out = np.empty((len(seqs), 4, length), dtype=_TWOHOT_LOOKUP.dtype)
    _twohot_gather(codes, _TWOHOT_LOOKUP, out)
    return out


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
            # len(pyfaidx.Fasta(...)) counts total bases, not records; use the
            # number of keys in the index for the number of sequences instead.
            len(pyfaidx.Fasta(chunk).keys())
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
        seqs = [str(rec) for rec in pyfaidx.Fasta(seq_file)]
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


class ScalarLoader(torch.utils.data.Dataset):
    """A data generator for sequence networks w/ single scalar outputs.

    This generator takes in an extracted set of sequences and output signal
    (single scalar/sequence) and will return a single element with random
    jitter and reverse-complement augmentation applied. Note that unlike with
    BPNet's data generator, the output is not jittered/rev comped. Jitter is
    implemented efficiently by taking in data that is wider than the in windows
    by two times the maximum jitter and windows are extracted from that.
    Essentially, if an input window is 1000 and the maximum jitter is 128, one
    would pass in data with a length of 1256 and a length 1000 window would be
    extracted starting between position 0 and 256. This generator must be
    wrapped by a PyTorch generator object.

    Parameters
    ----------
    sequences: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
            A one-hot encoded tensor of `n` example sequences, each of input
            length `in_window`. See description above for connection with jitter.

    signals: torch.tensor, shape=(n,)
            The signals to predict. A single scalar per sequence.

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
            (self.sequences.shape[-1] - self.in_window) // 2
            if self.max_jitter == 0
            else self.random_state.randint(self.max_jitter * 2)
        )

        X = self.sequences[i, :, j : j + self.in_window]
        y = self.signals[i]

        if self.reverse_complement and self.random_state.choice(2) == 1:
            X = torch.stack([reverse_complement_twohot(x.numpy().copy()) for x in X])

        return X, y


class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps=10, initial_lr=0.0001, target_lr=0.0005):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.target_lr = target_lr
        self.current_steps = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def step(self):
        self.current_steps += 1
        if self.current_steps <= self.warmup_steps:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (
                self.current_steps / self.warmup_steps
            )
            self.current_lr = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr


def plot_side(arr, ylim=[-2, 2.5], yticks=[0, 2], xticks=[], pic_name=None):
    """
    Adapted from APARENT code (Bogard et al. 2019)
    """
    if arr.shape[0] % 2 != 0:
        raise ValueError("arr must have even length.")
    midpoint = int(arr.shape[0] / 2)
    pl = arr[:midpoint]
    mn = arr[midpoint:]
    plt.bar(
        range(pl.shape[0]),
        pl,
        width=-2,
        color="r",
    )
    plt.bar(range(mn.shape[0]), -mn, width=-2, color="b")
    axes = plt.gca()
    axes.set_ylim(ylim)
    axes.set_yticks(yticks)
    axes.set_xticks(xticks)
    axes.spines[["right", "top", "bottom"]].set_visible(False)
    plt.xlim(-0.5, pl.shape[0] - 0.5)
    axes.tick_params(labelleft=False)

    if pic_name is None:
        plt.show()
    else:
        plt.savefig(pic_name, transparent=True)
        plt.close()
