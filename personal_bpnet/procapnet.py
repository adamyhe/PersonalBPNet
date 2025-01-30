# procapnet.py
# Kelly Cochran
# Copied from the ENCODE accession:
# https://www.encodeproject.org/files/ENCFF976FHE/@@download/ENCFF976FHE.tar.gz
# This class is included to make loading ProCapNet models easier.

import numpy as np
import torch


class ProCapNet(torch.nn.Module):
    def __init__(self, n_filters=512, n_layers=8, n_outputs=2):
        super(ProCapNet, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.trimming = 557
        self.deconv_kernel_size = 75

        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
        self.rconvs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    n_filters, n_filters, kernel_size=3, padding=2**i, dilation=2**i
                )
                for i in range(1, self.n_layers + 1)
            ]
        )
        self.fconv = torch.nn.Conv1d(
            n_filters, n_outputs, kernel_size=self.deconv_kernel_size
        )

        self.relus = torch.nn.ModuleList(
            [torch.nn.ReLU() for _ in range(0, self.n_layers + 1)]
        )
        self.linear = torch.nn.Linear(n_filters, 1)

    def forward(self, X):
        start, end = self.trimming, X.shape[2] - self.trimming

        X = self.relus[0](self.iconv(X))
        for i in range(self.n_layers):
            X_conv = self.relus[i + 1](self.rconvs[i](X))
            X = torch.add(X, X_conv)

        X = X[
            :,
            :,
            start - self.deconv_kernel_size // 2 : end + self.deconv_kernel_size // 2,
        ]

        y_profile = self.fconv(X)

        X = torch.mean(X, axis=2)
        y_counts = self.linear(X).reshape(X.shape[0], 1)

        return y_profile, y_counts

    def predict(self, X, batch_size=64, logits=False):
        with torch.no_grad():
            starts = np.arange(0, X.shape[0], batch_size)
            ends = starts + batch_size

            y_profiles, y_counts = [], []
            for start, end in zip(starts, ends):
                X_batch = X[start:end]

                y_profiles_, y_counts_ = self(X_batch)
                if not logits:  # apply softmax
                    y_profiles_ = self.log_softmax(y_profiles_)
                y_profiles.append(y_profiles_.cpu().detach().numpy())
                y_counts.append(y_counts_.cpu().detach().numpy())

            y_profiles = np.concatenate(y_profiles)
            y_counts = np.concatenate(y_counts)
            return y_profiles, y_counts

    def log_softmax(self, y_profile):
        y_profile = y_profile.reshape(y_profile.shape[0], -1)
        y_profile = torch.nn.LogSoftmax(dim=-1)(y_profile)
        y_profile = y_profile.reshape(y_profile.shape[0], self.n_outputs, -1)
        return y_profile
