# clipnet_pytorch.py
# Author: Adam He <adamyhe@gmail.com>

"""
CLIPNET: a subclass of PersonalBPNet (itself a subclass of bpnetlite.bpnet.BPNet)
that adds batch normalization after each convolutional and linear layer, which was
used in the original CLIPNET TensorFlow implementation and appears to improve
performance. Inherits PersonalBPNet's DataLoader-based fit() unchanged.

Also includes PauseNet, a transfer learning wrapper around CLIPNET (or any
BPNet-like model) for fine-tuning to a single scalar phenotype per locus.
"""

import time

import numpy as np
import torch
from bpnetlite.bpnet import CountWrapper
from bpnetlite.logging import Logger
from bpnetlite.losses import log1pMSELoss
from bpnetlite.performance import pearson_corr
from tangermeme.predict import predict

from .personal_bpnet import PersonalBPNet


class CLIPNET(PersonalBPNet):
    """
    A basic BPNet model with stranded profile and total count prediction.

    Identical to `PersonalBPNet` (and therefore `bpnetlite.bpnet.BPNet`), except that
    a batch normalization layer follows each convolutional and linear layer. See
    `PersonalBPNet`/`bpnetlite.bpnet.BPNet` for the model architecture, the meaning
    of the constructor parameters, and the DataLoader-based `fit()` method, which is
    inherited unchanged. Defaults for `n_filters`, `n_outputs`, and `trimming` are
    set to match the pretrained CLIPNET weights (2114 bp in, 1000 bp out).
    """

    def __init__(
        self, *args, n_filters=512, n_outputs=2, trimming=(2114 - 1000) // 2, **kwargs
    ):
        super().__init__(
            *args, n_filters=n_filters, n_outputs=n_outputs, trimming=trimming, **kwargs
        )
        self.ibn = torch.nn.BatchNorm1d(self.n_filters)
        self.rbn = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(self.n_filters) for _ in range(self.n_layers)]
        )
        self.pbn = torch.nn.BatchNorm1d(self.n_outputs)
        self.cbn = torch.nn.BatchNorm1d(1)

    def forward(self, X, X_ctl=None):
        """A forward pass of the model.

        This method takes in a nucleotide sequence X, a corresponding
        per-position value from a control track, and a per-locus value
        from the control track and makes predictions for the profile
        and for the counts. This per-locus value is usually the
        log(sum(X_ctl_profile)+1) when the control is an experimental
        read track but can also be the output from another model.

        Parameters
        ----------
        X: torch.tensor, shape=(batch_size, 4, length)
                The one-hot encoded batch of sequences.

        X_ctl: torch.tensor or None, shape=(batch_size, n_strands, length)
                A value representing the signal of the control at each position in
                the sequence. If no controls, pass in None. Default is None.

        Returns
        -------
        y_profile: torch.tensor, shape=(batch_size, n_strands, out_length)
                The output predictions for each strand trimmed to the output
                length.

        y_counts: torch.tensor, shape=(batch_size, 1)
                The output predictions for the total counts.
        """

        start, end = self.trimming, X.shape[2] - self.trimming

        X = self.irelu(self.ibn(self.iconv(X)))
        for i in range(self.n_layers):
            X_conv = self.rrelus[i](self.rbn[i](self.rconvs[i](X)))
            X = torch.add(X, X_conv)

        if X_ctl is None:
            X_w_ctl = X
        else:
            X_w_ctl = torch.cat([X, X_ctl], dim=1)

        # profile prediction
        y_profile = self.pbn(self.fconv(X_w_ctl))[:, :, start:end]

        # counts prediction
        X = torch.mean(X[:, :, start - 37 : end + 37], dim=2)
        if X_ctl is not None:
            X_ctl = torch.sum(X_ctl[:, :, start - 37 : end + 37], dim=(1, 2))
            X_ctl = X_ctl.unsqueeze(-1)
            X = torch.cat([X, torch.log(X_ctl + 1)], dim=-1)

        y_counts = self.cbn(self.linear(X).reshape(X.shape[0], 1))

        return y_profile, y_counts


class PauseNet(CountWrapper):
    """
    A class for transfer learning a CLIPNET model (or any BPNet-like model) to a
    single scalar phenotype per input. The base model's counts-head linear layer
    (and batch normalization layer, if present) are replaced with newly initialized
    ones, which remain trainable regardless of `base_trainable`. By default, the
    rest of the network is also trainable; set `base_trainable=False` to freeze it.
    """

    def __init__(
        self,
        base_model,
        base_trainable=True,
        n_filters=None,
        output_bias=True,
        name=None,
        verbose=True,
    ):
        super().__init__(base_model)
        self.name = name or "pausenet"
        self.base_trainable = base_trainable

        if not self.base_trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the base model's counts head with a freshly initialized one.
        # Defaults to the base model's existing linear layer width (which already
        # accounts for control tracks, if any) unless explicitly overridden.
        if n_filters is None:
            n_filters = self.model.linear.in_features
        self.model.linear = torch.nn.Linear(n_filters, 1, bias=output_bias)
        for param in self.model.linear.parameters():
            param.requires_grad = True

        if hasattr(self.model, "cbn"):
            self.model.cbn = torch.nn.BatchNorm1d(1)
            for param in self.model.cbn.parameters():
                param.requires_grad = True

        self.logger = Logger(
            [
                "Epoch",
                "Iteration",
                "Training Time",
                "Validation Time",
                "Training MSE",
                "Validation Pearson",
                "Validation MSE",
                "Saved?",
            ],
            verbose=verbose,
        )

    def fit(
        self,
        training_data,
        optimizer,
        scheduler=None,
        valid_data=None,
        max_epochs=100,
        batch_size=64,
        validation_iter=100,
        early_stopping=None,
        device="cuda",
        verbose=True,
    ):
        iteration = 0
        early_stop_count = 0
        best_loss = float("inf")
        self.logger.start()

        for epoch in range(max_epochs):
            tic = time.time()

            for data in training_data:
                if len(data) == 4:
                    X, X_ctl, y, _ = data
                    X_ctl = X_ctl.to(device)
                else:
                    X, y, _ = data
                    X_ctl = None

                X, y = X.to(device), y.to(device)

                # Clear the optimizer and set the model to training mode
                optimizer.zero_grad()
                self.train()

                # Run forward pass
                y_pred = self(X, X_ctl)

                # Calculate loss
                loss = log1pMSELoss(y_pred, torch.abs(y)).mean()

                # Extract the loss for logging
                loss_ = loss.item()

                # Update the model
                loss.backward()
                optimizer.step()

                # Report measures if desired
                if verbose and iteration % validation_iter == 0:
                    train_time = time.time() - tic

                    with torch.no_grad():
                        self.eval()

                        tic = time.time()

                        # Initialize lists to store validation statistics
                        valid_mse = []
                        pred_val = []
                        obs_val = []

                        # Loop over the validation data
                        for data in valid_data:
                            if len(data) == 4:
                                X_val, X_ctl_val, y_val, _ = data
                                X_ctl_val = (X_ctl_val,)
                            else:
                                X_val, y_val, _ = data
                                X_ctl_val = None

                            y_val = torch.abs(y_val)
                            y_pred = predict(
                                self,
                                X_val,
                                args=X_ctl_val,
                                batch_size=batch_size,
                                device=device,
                            )
                            obs_val.append(y_val)
                            pred_val.append(y_pred)
                            valid_mse.append(log1pMSELoss(y_val, y_pred))

                        val_corr = pearson_corr(
                            torch.cat(pred_val).squeeze(),
                            torch.log(torch.cat(obs_val).squeeze() + 1),
                        )
                        valid_loss = torch.cat(valid_mse).mean()

                        valid_time = time.time() - tic

                        self.logger.add(
                            [
                                epoch,
                                iteration,
                                train_time,
                                valid_time,
                                loss_,
                                np.nan_to_num(val_corr).item(),
                                valid_loss.item(),
                                (valid_loss < best_loss).item(),
                            ]
                        )

                        self.logger.save(f"{self.name}.log")

                        # Save the model if it is the best so far
                        if valid_loss < best_loss:
                            torch.save(self.state_dict(), f"{self.name}.torch")
                            torch.save(
                                {
                                    "early_stop_count": early_stop_count,
                                    "epoch": epoch,
                                    "optimizer_state_dict": optimizer.state_dict(),
                                },
                                f"{self.name}.checkpoint.torch",
                            )
                            best_loss = valid_loss
                            early_stop_count = 0
                        else:
                            early_stop_count += 1

                if early_stopping is not None and early_stop_count >= early_stopping:
                    break

                iteration += 1

            if early_stopping is not None and early_stop_count >= early_stopping:
                break
            if scheduler is not None:
                scheduler.step()

        torch.save(self, f"{self.name}.final.torch")
