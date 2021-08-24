from .train import LitNNSimulator
from torchvision.io import read_image
import torch as th
import numpy as np
import matplotlib.pyplot as plt

model = LitNNSimulator.load_from_checkpoint(
    "lightning_logs/version_1/checkpoints/epoch=219-step=54999.ckpt"
)
model.eval()

image = read_image("8210_11.png").unsqueeze(0)/255
thickness = th.eye(16)[10].float().unsqueeze(0)

y = model(image, thickness).detach().numpy()[0]

plt.figure()
plt.plot(np.abs(y[0]), 'r-', label='n0')
plt.plot(np.abs(y[1]), 'b-', label='n90')
plt.plot(np.abs(y[0]) - np.abs(y[1]), 'g-', label='n90')
# plt.plot(y[2], 'r--', label='k0')
# plt.plot(y[3], 'b--', label='k90')
plt.xlabel("Freq (THz)")
plt.ylabel("n")
plt.legend()
plt.savefig('n.png')
plt.close()

# y = np.load("dataset/spectrums/8210.npy")

y = model.spectrum(image, thickness).detach().numpy()[0]

S11_0 = y[0] + 1j*y[1]
S11_90 = y[2] + 1j*y[3]
S21_0 = y[4] + 1j*y[5]
S21_90 = y[6] + 1j*y[7]

plt.figure()
plt.plot(np.abs(S21_0) ** 2, 'r-', label='tran0')
plt.plot(np.abs(S11_0) ** 2, 'b-', label='refl0')
plt.plot(np.abs(S21_90) ** 2, 'r--', label='tran90')
plt.plot(np.abs(S11_90) ** 2, 'b--', label='refl90')
plt.xlabel("Freq (THz)")
plt.ylabel("Transmittance/Reflectance (-)")
plt.legend()
plt.savefig('amp.png')
plt.close()

plt.figure()
plt.plot(np.angle(S21_0), 'r-', label='tran0')
plt.plot(np.angle(S11_0), 'b-', label='refl0')
plt.plot(np.angle(S21_90), 'r--', label='tran90')
plt.plot(np.angle(S11_90), 'b--', label='refl90')
plt.xlabel("Freq (THz)")
plt.ylabel("Transmittance/Reflectance (-)")
plt.legend()
plt.savefig('angle.png')
plt.close()
