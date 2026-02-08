# bnbpnet.py
# Author: Adam He <adamyhe@gmail.com>

"""
Adapted from bpnetlite
Original license: https://github.com/jmschrei/bpnet-lite/blob/master/LICENSE

Redid the validation loop to work with a PyTorch DataLoader, rather than
having to load the whole validation set into memory at once. Also added a bunch of
batch normalization layers, which were used in CLIPNET and appear to improve
performance.

Also, the model checkpoints save the optimizer state dict and epoch number
in addition to the model state dict, so that training can be resumed from
a checkpoint.
"""

import time

import numpy as np
import torch
from bpnetlite.logging import Logger
from bpnetlite.performance import calculate_performance_measures, pearson_corr
from tangermeme.predict import predict

from .losses import _mixture_loss_masked

torch.backends.cudnn.benchmark = True


class BNBPNet(torch.nn.Module):
    """
    Batch Normalized BPNet. Adapted from the implementation of BPNet in bpnetlite.

    Contains additional batch normalization layers not found in BPNet.

        The model takes in one-hot encoded sequence, runs it through:

        (1) a single wide convolution operation

        THEN

        (2) a user-defined number of dilated residual convolutions

        THEN

        (3a) profile predictions done using a very wide convolution layer
        that also takes in stranded control tracks

        AND

        (3b) total count prediction done using an average pooling on the output
        from 2 followed by concatenation with the log1p of the sum of the
        stranded control tracks and then run through a dense layer.

        Batch normalization is added after each (de)convolution and before the
        activation function.

        Parameters
        ----------
        n_filters: int, optional
                The number of filters to use per convolution. Default is 64.

        n_layers: int, optional
                The number of dilated residual layers to include in the model.
                Default is 8.

        n_outputs: int, optional
                The number of profile outputs from the model. Generally either 1 or 2
                depending on if the data is unstranded or stranded. Default is 2.

        n_control_tracks: int, optional
                The number of control tracks to feed into the model. When predicting
                TFs, this is usually 2. When predicting accessibility, this is usualy
                0. When 0, this input is removed from the model. Default is 2.

        alpha: float, optional
                The weight to put on the count loss.

        profile_output_bias: bool, optional
                Whether to include a bias term in the final profile convolution.
                Removing this term can help with attribution stability and will usually
                not affect performance. Default is True.

        count_output_bias: bool, optional
                Whether to include a bias term in the linear layer used to predict
                counts. Removing this term can help with attribution stability but
                may affect performance. Default is True.

        name: str or None, optional
                The name to save the model to during training.

        trimming: int or None, optional
                The amount to trim from both sides of the input window to get the
                output window. This value is removed from both sides, so the total
                number of positions removed is 2*trimming.

        verbose: bool, optional
                Whether to display statistics during training. Setting this to False
                will still save the file at the end, but does not print anything to
                screen during training. Default is True.
    """

    def __init__(
        self,
        n_filters=512,
        n_layers=8,
        n_outputs=2,
        n_control_tracks=0,
        alpha=1,
        profile_output_bias=True,
        count_output_bias=True,
        name=None,
        trimming=(2114 - 1000) // 2,
        verbose=True,
    ):
        # We need to define all the layers in the __init__ method
        super(CLIPNET, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.n_control_tracks = n_control_tracks

        self.alpha = alpha
        self.name = name or "clipnet.{}.{}".format(n_filters, n_layers)
        self.trimming = trimming or 2**n_layers

        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
        self.ibn = torch.nn.BatchNorm1d(n_filters)
        self.irelu = torch.nn.ReLU()

        self.rconvs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    n_filters, n_filters, kernel_size=3, padding=2**i, dilation=2**i
                )
                for i in range(1, self.n_layers + 1)
            ]
        )
        self.rbn = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(n_filters) for i in range(1, self.n_layers + 1)]
        )
        self.rrelus = torch.nn.ModuleList(
            [torch.nn.ReLU() for i in range(1, self.n_layers + 1)]
        )

        self.fconv = torch.nn.Conv1d(
            n_filters + n_control_tracks,
            n_outputs,
            kernel_size=75,
            padding=37,
            bias=profile_output_bias,
        )
        self.pbn = torch.nn.BatchNorm1d(n_outputs)

        n_count_control = 1 if n_control_tracks > 0 else 0
        self.linear = torch.nn.Linear(
            n_filters + n_count_control, 1, bias=count_output_bias
        )
        self.cbn = torch.nn.BatchNorm1d(1)

        self.logger = Logger(
            [
                "Epoch",
                "Training Time",
                "Validation Time",
                "Training MNLL",
                "Training Count MSE",
                "Validation MNLL",
                "Validation Profile Pearson",
                "Validation Count Pearson",
                "Validation Count MSE",
                "Saved?",
            ],
            verbose=verbose,
        )

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

        # return
        return y_profile, y_counts

    def fit(
        self,
        training_data,
        optimizer,
        valid_data,
        ctl_has_mask=True,
        scheduler=None,
        max_epochs=100,
        batch_size=64,
        early_stopping=None,
        dtype=torch.bfloat16,
        device="cuda",
        verbose=True,
    ):
        """Fit the model to data and validate it periodically.

        This method controls the training of a BPNet model. It will fit the
        model to examples generated by the `training_data` DataLoader object
        and, if validation data is provided, will periodically validate the
        model against it and return those values. The periodicity can be
        controlled using the `validation_iter` parameter.

        Two versions of the model will be saved: the best model found during
        training according to the validation measures, and the final model
        at the end of training. Additionally, a log will be saved of the
        training and validation statistics, e.g. time and performance.

        Parameters
        ----------
        training_data: torch.utils.data.DataLoader
                A generator that produces examples to train on. If n_control_tracks
                is greater than 0, must product two inputs, otherwise must produce
                only one input.

        optimizer: torch.optim.Optimizer
                An optimizer to control the training of the model.

        valid_data: torch.utils.data.DataLoader
                A generator that produces examples to earlystop on. If n_control_tracks
                is greater than 0, must product two inputs, otherwise must produce
                only one input.

        ctl_has_mask: bool
                Whether the control tracks will contain a boolean mask for excluding
                positions from profile loss calculations. The final control track
                will be treated as a boolean mask, whereas the other tracks will be
                processed as normal. Default is True.

        scheduler: torch.optim.lr_scheduler._LRScheduler or None
                A scheduler to control the learning rate of the optimizer. Optional.

        max_epochs: int
                The maximum number of epochs to train for, as measured by the
                number of times that `training_data` is exhausted. Default is 100.

        batch_size: int
                The number of examples to include in each batch. Default is 64.

        early_stopping: int or None
                Whether to stop training early. If None, continue training until
                max_epochs is reached. If an integer, continue training until that
                number of `validation_iter` ticks has been hit without improvement
                in performance. Default is None.

        verbose: bool
                Whether to print out the training and evaluation statistics during
                training. Default is True.
        """
        early_stop_count = 0
        best_loss = float("inf")
        self.logger.start()

        for epoch in range(max_epochs):
            tic = time.time()

            training_profile_loss_log = 0
            training_count_loss_log = 0
            for data in training_data:
                if len(data) == 4:
                    X, X_ctl, y, labels = data
                    X_ctl = torch.abs(X_ctl).to(device).float()
                    if ctl_has_mask:
                        mask = X_ctl[:, :, -1].bool()
                        if self.n_control_tracks > 0:
                            X_ctl = X_ctl[:, :, :-1]
                        else:
                            X_ctl = None
                    else:
                        mask = None
                else:
                    X, y, labels = data
                    X_ctl = None
                    mask = None

                X = X.to(device).float()
                y = torch.abs(y).to(device)

                # Clear the optimizer and set the model to training mode
                optimizer.zero_grad()
                self.train()

                # Run forward pass
                with torch.autocast(device_type=device, dtype=dtype):
                    y_hat_logits, y_hat_logcounts = self(X, X_ctl)
                    training_profile_loss, training_count_loss, loss = (
                        _mixture_loss_masked(
                            y, y_hat_logits, y_hat_logcounts, self.alpha, labels, mask
                        )
                    )
                    # Log training statistics
                    training_profile_loss_log += training_profile_loss.item()
                    training_count_loss_log += training_count_loss.item()

                    # Backpropagate
                    loss.backward()
                    optimizer.step()

            # Update the learning rate if a scheduler is provided
            if scheduler is not None:
                scheduler.step()

            # Measure the training time
            train_time = time.time() - tic

            # Evaluate the model on the validation data
            with torch.no_grad():
                self.eval()

                tic = time.time()

                # Initialize lists to store validation statistics
                profile_corr = []
                valid_mnll = []
                valid_mse = []
                pred_counts = []
                obs_counts = []

                # Loop over the validation data
                for data in valid_data:
                    if len(data) == 3:
                        X_val, X_ctl_val, y_val = data
                        X_ctl_val = (torch.abs(X_ctl_val),)
                    else:
                        X_val, y_val = data
                        X_ctl_val = None

                    y_val = torch.abs(y_val)
                    y_profile, y_counts = predict(
                        self,
                        X_val,
                        args=X_ctl_val,
                        batch_size=batch_size,
                        device=device,
                        dtype=dtype,
                    )
                    obs_counts.append(y_val.sum(dim=(-2, -1)).reshape(-1, 1))
                    pred_counts.append(y_counts)

                    z = y_profile.shape
                    y_profile = y_profile.reshape(y_profile.shape[0], -1)
                    y_profile = torch.nn.functional.log_softmax(y_profile, dim=-1)
                    y_profile = y_profile.reshape(*z)

                    measures = calculate_performance_measures(
                        y_profile,
                        y_val,
                        y_counts,
                        kernel_sigma=7,
                        kernel_width=81,
                        measures=[
                            "profile_mnll",
                            "profile_pearson",
                            "count_mse",
                        ],
                    )
                    profile_corr.append(measures["profile_pearson"])
                    valid_mnll.append(measures["profile_mnll"])
                    valid_mse.append(measures["count_mse"])

                # Other metrics can be calculated in the loop, but
                # count_corr needs to be calculated by storing the
                # counts and then calculating the correlation at the end
                count_corr = pearson_corr(
                    torch.cat(pred_counts).squeeze(),
                    torch.log(torch.cat(obs_counts).squeeze() + 1),
                )

                # Concatenate the lists of validation measures
                profile_corr = torch.cat(profile_corr)
                valid_mnll = torch.cat(valid_mnll)
                valid_mse = torch.cat(valid_mse)
                valid_loss = valid_mnll.mean() + self.alpha * valid_mse.mean()

                valid_time = time.time() - tic

                self.logger.add(
                    [
                        epoch,
                        train_time,
                        valid_time,
                        training_profile_loss_log / len(training_data),
                        training_count_loss_log / len(training_data),
                        measures["profile_mnll"].mean().item(),
                        np.nan_to_num(profile_corr).mean(),
                        np.nan_to_num(count_corr).mean(),
                        measures["count_mse"].mean().item(),
                        (valid_loss < best_loss).item(),
                    ]
                )

                self.logger.save("{}.log".format(self.name))

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

            # Check for early stopping after finishing training and validation
            if early_stopping is not None and early_stop_count >= early_stopping:
                break

        torch.save(self, "{}.final.torch".format(self.name))


# Create CLIPNET alias for BNBPNet
CLIPNET = BNBPNet
ProCapNet = BNBPNet
