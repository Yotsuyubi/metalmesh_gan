from .train import LitGAN
from nn_simulator import nn_simulator
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import torchvision


nnsim = nn_simulator("model/nnsim.ckpt", "freq.txt")
model = LitGAN(nnsim=nnsim).load_from_checkpoint(
    "checkpoints/last.ckpt"
)
model.eval()

z = th.randn(1, 512).requires_grad_(True)
thickness_weight = th.eye(16)[0].unsqueeze(0).requires_grad_(True)

optim = th.optim.Adam([
    {'params': z},
    {'params': thickness_weight, 'lr': 1e-2}
], lr=1e-2)
best = 0
best_img = None
best_thick = None
best_r = 0
best_T = 0
for _ in range(5000):
    optim.zero_grad()

    image, _ = model(z)
    thickness = th.softmax(thickness_weight, dim=-1)

    y = nnsim(image, thickness)[0]
    S = nnsim.spectrum(image, thickness)[0]
    S21_0 = th.abs(y[4] + 1j*y[5])
    S21_90 = th.abs(y[6] + 1j*y[7])

    n, index = th.max((th.abs(y[0]) - th.abs(y[1])).abs(), dim=0)
    r = 2 * np.pi * n * (th.max(thickness, dim=1)[1]+1)*8e-6 / (
        299792458/nnsim.freq[index])
    T = S21_0[index] + S21_90[index]

    loss = - (r + T*1)
    loss.backward()
    print(T.item(), r.item())

    if -loss > best:
        best_img = image
        best = -loss
        best_thick = thickness
        best_r = r
        best_T = T

    optim.step()
print(best, best_r, best_T)

image, _ = model(z)
thickness = th.softmax(thickness_weight, dim=-1)
image = th.where(image >= 0.5, 1.0, 0.0)
y = nnsim(image, thickness)[0]
S = nnsim.spectrum(image, thickness)[0]
S21_0 = th.abs(y[4] + 1j*y[5])
S21_90 = th.abs(y[6] + 1j*y[7])
index = 150
n, _ = th.max((th.abs(y[0, index]) - th.abs(y[1, index])).abs(), dim=0)
r = 2 * np.pi * n * (th.max(thickness, dim=1)[1]+1)*8e-6 / (
    299792458/th.linspace(0.5e12, 1e12, steps=250)[index])
print(r, n)
torchvision.utils.save_image(image, "test.png")
print(th.max(thickness, dim=1)[1]+1)
y = nnsim(image, thickness).detach().cpu().numpy()[0]

freq = nnsim.fresnel.freq / 1e12

plt.figure()
plt.plot(freq, np.abs(y[0]), 'r-', label='n0')
plt.plot(freq, np.abs(y[1]), 'b-', label='n90')
plt.plot(freq, np.abs(np.abs(y[0]) - np.abs(y[1])), 'g-', label='$\Delta n$')
plt.xlabel("Freq (THz)")
plt.ylabel("n")
plt.legend()
plt.savefig('n.png')
plt.close()

y = nnsim.spectrum(image, thickness).detach().cpu().numpy()[0]

S11_0 = y[0] + 1j*y[1]
S11_90 = y[2] + 1j*y[3]
S21_0 = y[4] + 1j*y[5]
S21_90 = y[6] + 1j*y[7]

plt.figure()
plt.plot(freq, np.abs(S21_0) ** 2, 'r-', label='tran0')
plt.plot(freq, np.abs(S11_0) ** 2, 'b-', label='refl0')
plt.plot(freq, np.abs(S21_90) ** 2, 'r--', label='tran90')
plt.plot(freq, np.abs(S11_90) ** 2, 'b--', label='refl90')
plt.xlabel("Freq (THz)")
plt.ylabel("Transmittance/Reflectance (-)")
plt.legend()
plt.savefig('amp.png')
plt.close()

plt.figure()
plt.plot(freq, np.angle(S21_0), 'r-', label='tran0')
plt.plot(freq, np.angle(S11_0), 'b-', label='refl0')
plt.plot(freq, np.angle(S21_90), 'r--', label='tran90')
plt.plot(freq, np.angle(S11_90), 'b--', label='refl90')
plt.xlabel("Freq (THz)")
plt.ylabel("Transmittance/Reflectance (-)")
plt.legend()
plt.savefig('angle.png')
plt.close()
