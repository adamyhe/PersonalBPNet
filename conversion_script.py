import torch

from personal_bpnet.clipnet_tensorflow import CLIPNET_TF

for i in range(1, 10):
    model = CLIPNET_TF.from_tf(f"fold_{i}.h5")
    torch.save(model, f"fold_{i}.torch")

models_torch = []
for i in range(1, 10):
    models_torch.append(torch.load(f"fold_{i}.torch"))
