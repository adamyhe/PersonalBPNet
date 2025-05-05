# clipnet_tensorflow.py
# Author: Adam He <adamyhe@gmail.com>

"""
A class for importing CLIPNET models trained in TensorFlow into PyTorch.
"""

import h5py
import torch


def _namer(prefix, suffix):
    """
    A small helper function to make fetching names from h5 files easier.
    """
    return f"{prefix}{suffix}/{prefix}{suffix}"


def _convert_w(x):
    """
    A small helper function to make loading weight parameters easier.
    """
    return torch.nn.Parameter(torch.tensor(x[:]).permute(2, 1, 0))


def _convert_b(x):
    """
    A small helper function to make loading bias parameters easier.
    """
    return torch.nn.Parameter(torch.tensor(x[:]))


def _get_tf_same_padding(kernel_size, dilation=1):
    """
    Returns the left and right padding for a given kernel size and dilation.
    Necessary to reproduce TensorFlow's "same" padding in PyTorch.
    """
    total_pad = (kernel_size - 1) * dilation
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    return pad_left, pad_right


class AsymmetricPad(torch.nn.Module):
    """
    This layer manually injects asymmetric padding into a convolutional layer.
    Set the padding on the convolutional layer to "0" and use this layer to
    inject asymmetric padding. This reproduces TensorFlow's "same" padding.
    """

    def __init__(self, kernel_size, dilation=1):
        super(AsymmetricPad, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        pad_left, pad_right = _get_tf_same_padding(self.kernel_size, self.dilation)
        pad_layer = torch.nn.ConstantPad1d((pad_left, pad_right), 0)
        return pad_layer(x)


class CLIPNET_TF(torch.nn.Module):
    """
    This is a reference implementation for CLIPNET models. It exactly matches the
    architecture in the main CLIPNET TensorFlow repository and contains a method
    for converting the TensorFlow weights to PyTorch. Note that this architecture
    differs from the one found in clipnet_pytorch.py, which is a complete rewrite
    in PyTorch and takes a lot more cues from ProCapNet/BPNet.
    """

    def __init__(self):
        super(CLIPNET_TF, self).__init__()
        self.n_layers = 9

        self.input = torch.nn.Identity()
        self.ibnorm = torch.nn.BatchNorm1d(4)
        self.iconv = torch.nn.Conv1d(4, 64, kernel_size=8, padding=0)
        self.ibn = torch.nn.BatchNorm1d(64)
        self.irelu = torch.nn.ELU(alpha=1)
        self.imp = torch.nn.MaxPool1d(2)

        self.sconv = torch.nn.Conv1d(32, 128, kernel_size=4, padding=0)
        self.sbnorm = torch.nn.BatchNorm1d(64)
        self.srelu = torch.nn.ReLU()
        self.smp = torch.nn.MaxPool1d(2)

        self.rconv = torch.nn.Conv1d(64, 64, kernel_size=1, padding=0)
        self.rpads = torch.nn.ModuleList(
            [AsymmetricPad(kernel_size=3, dilation=2**i) for i in range(self.n_layers)]
        )
        self.rconvs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(64, 64, kernel_size=3, padding=0, dilation=2**i)
                for i in range(self.n_layers)
            ]
        )
        self.rbnorms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(64) for i in range(self.n_layers)]
        )
        self.rrelus = torch.nn.ModuleList(
            [torch.nn.ReLU() for i in range(self.n_layers)]
        )
        self.skipbnorms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(64) for i in range(self.n_layers)]
        )
        self.skiprelus = torch.nn.ModuleList(
            [torch.nn.ReLU() for i in range(self.n_layers)]
        )
        self.skipmp = torch.nn.MaxPool1d(2)

        self.plinear = torch.nn.Linear(7872, 1000)
        self.pbnorm = torch.nn.BatchNorm1d(1000)
        self.prelu = torch.nn.ReLU()

        self.cavgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.clinear = torch.nn.Linear(64, 1)
        self.cbnorm = torch.nn.BatchNorm1d(1)
        self.crelu = torch.nn.ReLU()

    def forward(self, X):
        """
        A forward pass of the model.

        This method takes in a nucleotide sequence X and makes predictions
        for the profile and for the counts.

        Parameters
        ----------
        X: torch.tensor, shape=(batch_size, 4, length)
                The two-hot encoded batch of sequences.

        Returns
        -------
        y_profile: torch.tensor, shape=(batch_size, out_length * n_strands)
                The output predictions for each strand.
        y_counts: torch.tensor, shape=(batch_size, 1)
                The output predictions for the counts (in RPM scale).
        """
        # input
        X = self.input(X)
        X = self.ibnorm(X)
        # Apply first convolutional layer
        X = self.iconv(X)
        X = self.ibn(X)
        X = self.irelu(X)
        X = self.imp(X)
        # Apply second convolutional layer
        X = self.sconv(X)
        X = self.sbnorm(X)
        X = self.srelu(X)
        X = self.smp(X)
        # Apply third convolutional layer (for shape compatibility)
        X = self.rconv(X)
        # Apply dilated residual layers
        for i in range(self.n_layers):
            # Dilated convolution
            X_conv = self.rpads[i](X)
            X_conv = self.rconvs[i](X_conv)
            X_conv = self.rbnorms[i](X_conv)
            X_conv = self.rrelus[i](X_conv)
            # Skip connection
            X = torch.add(X, X_conv)
            X = self.skipbnorms[i](X)
            X = self.skiprelus[i](X)
        # Max pooling after residual layers
        X = self.skipmp(X)
        # Profile prediction
        # We need to manually permute and flatten since the pytorch
        # conv layers output a tensor of shape (batch, channels, steps), but
        # the linear weights expect a tensor of shape (batch, steps, channels)
        y_profile = X.permute(0, 2, 1).reshape(X.shape[0], -1)
        y_profile = self.plinear(y_profile)
        y_profile = self.pbnorm(y_profile)
        y_profile = self.prelu(y_profile)
        # Count prediction
        y_counts = self.cavgpool(X).squeeze(-1)
        y_counts = self.clinear(y_counts)
        y_counts = self.cbnorm(y_counts)
        y_counts = self.crelu(y_counts)

        return y_profile, y_counts

    @classmethod
    def from_tf(cls, filename):
        """Loads a model from the CLIPNET TensorFlow format.

        Note that this method does not require the installation of TensorFlow and
        differs in architecture from the PyTorch implementation in this repository.
        Notably, CLIPNET_TF models take in 1000 bp sequence inputs and yield 500 bp
        predictions.

        See https://github.com/Danko-Lab/clipnet for more details.

        Parameters
        ----------
        filename: str
                The name of the h5 file that stores the trained model parameters.

        Returns
        -------
        model: CLIPNET_TF
                A CLIPNET tensorflow model compatible with this repository in PyTorch.
        """
        h5 = h5py.File(filename, "r")
        w = h5["model_weights"]
        k, b, mm, mv = "kernel:0", "bias:0", "moving_mean:0", "moving_variance:0"

        model = CLIPNET_TF()

        ibnorm = _namer("batch_normalization", "")
        model.ibnorm.weight.data = _convert_b(w[ibnorm]["gamma:0"])
        model.ibnorm.bias.data = _convert_b(w[ibnorm]["beta:0"])
        model.ibnorm.running_mean = _convert_b(w[ibnorm][mm])
        model.ibnorm.running_var = _convert_b(w[ibnorm][mv])
        model.ibnorm.eps = 0.001
        model.ibnorm.momentum = 0.99

        iconv = _namer("conv1d", "")
        model.iconv.weight.data = _convert_w(w[iconv][k])
        model.iconv.bias.data = _convert_b(w[iconv][b])

        ibn = _namer("batch_normalization", "_1")
        model.ibn.weight.data = _convert_b(w[ibn]["gamma:0"])
        model.ibn.bias.data = _convert_b(w[ibn]["beta:0"])
        model.ibn.running_mean = _convert_b(w[ibn][mm])
        model.ibn.running_var = _convert_b(w[ibn][mv])
        model.ibn.eps = 0.001
        model.ibn.momentum = 0.99

        sconv = _namer("conv1d", "_1")
        model.sconv.weight.data = _convert_w(w[sconv][k])
        model.sconv.bias.data = _convert_b(w[sconv][b])

        sbn = _namer("batch_normalization", "_2")
        model.sbnorm.weight.data = _convert_b(w[sbn]["gamma:0"])
        model.sbnorm.bias.data = _convert_b(w[sbn]["beta:0"])
        model.sbnorm.running_mean = _convert_b(w[sbn][mm])
        model.sbnorm.running_var = _convert_b(w[sbn][mv])
        model.sbnorm.eps = 0.001
        model.sbnorm.momentum = 0.99

        rconv = _namer("conv1d", "_2")
        model.rconv.weight.data = _convert_w(w[rconv][k])
        model.rconv.bias.data = _convert_b(w[rconv][b])

        for i in range(1, 9 + 1):
            rconv = _namer("conv1d", f"_{i + 2}")
            model.rconvs[i - 1].weight.data = _convert_w(w[rconv][k])
            model.rconvs[i - 1].bias.data = _convert_b(w[rconv][b])

            rbn = _namer("batch_normalization", f"_{(2 * i - 1) + 2}")
            model.rbnorms[i - 1].weight.data = _convert_b(w[rbn]["gamma:0"])
            model.rbnorms[i - 1].bias.data = _convert_b(w[rbn]["beta:0"])
            model.rbnorms[i - 1].running_mean = _convert_b(w[rbn][mm])
            model.rbnorms[i - 1].running_var = _convert_b(w[rbn][mv])
            model.rbnorms[i - 1].eps = 0.001
            model.rbnorms[i - 1].momentum = 0.99

            skipbn = _namer("batch_normalization", f"_{(2 * i - 1) + 3}")
            model.skipbnorms[i - 1].weight.data = _convert_b(w[skipbn]["gamma:0"])
            model.skipbnorms[i - 1].bias.data = _convert_b(w[skipbn]["beta:0"])
            model.skipbnorms[i - 1].running_mean = _convert_b(w[skipbn][mm])
            model.skipbnorms[i - 1].running_var = _convert_b(w[skipbn][mv])
            model.skipbnorms[i - 1].eps = 0.001
            model.skipbnorms[i - 1].momentum = 0.99

        plinear = _namer("dense", "")
        model.plinear.weight.data = torch.nn.Parameter(torch.tensor(w[plinear][k][:].T))
        model.plinear.bias.data = _convert_b(w[plinear][b])

        pbn = _namer("batch_normalization", "_21")
        model.pbnorm.weight.data = _convert_b(w[pbn]["gamma:0"])
        model.pbnorm.bias.data = _convert_b(w[pbn]["beta:0"])
        model.pbnorm.running_mean = _convert_b(w[pbn][mm])
        model.pbnorm.running_var = _convert_b(w[pbn][mv])
        model.pbnorm.eps = 0.001
        model.pbnorm.momentum = 0.99

        clinear = _namer("dense", "_1")
        model.clinear.weight.data = torch.nn.Parameter(torch.tensor(w[clinear][k][:].T))
        model.clinear.bias.data = _convert_b(w[clinear][b])

        cbn = _namer("batch_normalization", "_22")
        model.cbnorm.weight.data = _convert_b(w[cbn]["gamma:0"])
        model.cbnorm.bias.data = _convert_b(w[cbn]["beta:0"])
        model.cbnorm.running_mean = _convert_b(w[cbn][mm])
        model.cbnorm.running_var = _convert_b(w[cbn][mv])
        model.cbnorm.eps = 0.001
        model.cbnorm.momentum = 0.99

        return model
