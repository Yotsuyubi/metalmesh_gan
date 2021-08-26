from .train import LitGAN, LitNNSimulator
from torchvision.io import read_image
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import torchvision


nnsim = LitNNSimulator.load_from_checkpoint(
    "model/nnsim.ckpt"
)
model = LitGAN(nnsim=nnsim).load_from_checkpoint(
    "model/gan.ckpt"
)
model.eval()

# spectrum = th.tensor(
#     np.load("dataset/spectrums/10.npy")
# ).float().unsqueeze(0)
spectrum = th.rand(1, 8, 250).requires_grad_(True)

# y = spectrum[0]
# S11_0 = y[0] + 1j*y[1]
# S11_90 = y[2] + 1j*y[3]
# S21_0 = y[4] + 1j*y[5]
# S21_90 = y[6] + 1j*y[7]

# plt.figure()
# plt.plot(np.abs(S21_0) ** 2, 'r-', label='tran0')
# plt.plot(np.abs(S11_0) ** 2, 'b-', label='refl0')
# plt.plot(np.abs(S21_90) ** 2, 'r--', label='tran90')
# plt.plot(np.abs(S11_90) ** 2, 'b--', label='refl90')
# plt.xlabel("Freq (THz)")
# plt.ylabel("Transmittance/Reflectance (-)")
# plt.legend()
# plt.savefig('_amp.png')
# plt.close()

# plt.figure()
# plt.plot(np.angle(S21_0), 'r-', label='tran0')
# plt.plot(np.angle(S11_0), 'b-', label='refl0')
# plt.plot(np.angle(S21_90), 'r--', label='tran90')
# plt.plot(np.angle(S11_90), 'b--', label='refl90')
# plt.xlabel("Freq (THz)")
# plt.ylabel("Transmittance/Reflectance (-)")
# plt.legend()
# plt.savefig('_angle.png')
# plt.close()

optim = th.optim.Adam([spectrum], lr=1e-2)
for _ in range(100):
    optim.zero_grad()
    image, thickness = model(spectrum)
    # thickness = th.eye(16)[3].unsqueeze(0).type_as(thickness)
    y = nnsim(image, thickness)[0]
    loss = -(th.abs(th.abs(y[0]) - th.abs(y[1])).max())
    print(-loss)
    loss.backward()
    optim.step()

image, thickness = model(spectrum)
# thickness = th.eye(16)[3].unsqueeze(0).type_as(thickness)
image = th.where(image >= 0.5, 1.0, 0.0)
torchvision.utils.save_image(image, "test.png")
print(th.max(thickness, dim=1)[1]+1)

image, thickness = model(spectrum)
image = th.where(image >= 0.5, 1.0, 0.0)
# thickness = th.eye(16)[3].unsqueeze(0).type_as(thickness)
y = nnsim(image, thickness).detach().numpy()[0]

plt.figure()
plt.plot(np.abs(y[0]), 'r-', label='n0')
plt.plot(np.abs(y[1]), 'b-', label='n90')
plt.plot(np.abs(y[0]) - np.abs(y[1]), 'g-', label='n90')
plt.xlabel("Freq (THz)")
plt.ylabel("n")
plt.legend()
plt.savefig('n.png')
plt.close()

y = nnsim.spectrum(image, thickness).detach().numpy()[0]

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
