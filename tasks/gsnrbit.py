import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcinv
from core.math_utils import *
import math

Rs = 32e9
Bn = 12.5e9  # Noise bandwidth (Hz)
BERt = 1e-3  # Target bit error rate

th_100 = 2 * (erfcinv(2 * BERt)) ** 2 * (Rs / Bn)
th_200 = (14 / 3) * (erfcinv(BERt * (3 / 2))) ** 2 * (Rs / Bn)
th_400 = 10 * (erfcinv(BERt * (8 / 3))) ** 2 * (Rs / Bn)
print(th_100)
print(th_200)
print(th_400)
gsnr_values = np.linspace(0, 40, 100)
fixed_bit_rate = []
shannon_bit_rate=[]
flex_bit_rate=[]

for GSNR in gsnr_values:

    GSNR_l=db2lin(GSNR)
    print(f"GSNR(dB): {GSNR}, GSNR(lineare): {GSNR_l}")
    if GSNR_l < th_100:
        flex_bit_rate.append(0)
    elif th_100 <= GSNR_l < th_200:
        flex_bit_rate.append(100)
    elif th_200 <= GSNR_l < th_400:
        flex_bit_rate.append(200)
    elif GSNR_l >= th_400:
        flex_bit_rate.append(400)

    fixed_bit_rate.append(100 if GSNR_l >= th_100 else 0)
    shannon_bit_rate.append((2 * Rs * math.log2(1 + GSNR_l * (Rs / Bn)))*10**(-9))

print(shannon_bit_rate)
plt.figure(figsize=(10, 6))

plt.plot(gsnr_values, shannon_bit_rate, label="shannon", color="blue")
plt.plot(gsnr_values, fixed_bit_rate, label="fixed-rate", color="red")
plt.plot(gsnr_values, flex_bit_rate, label="flex-rate", color="cyan")

plt.xlabel("GSNR (dB)", fontsize=12)
plt.ylabel("Bit Rate (Gbps)", fontsize=12)
plt.title("Bit Rate vs. GSNR for Different Strategies", fontsize=14)

plt.legend()
plt.grid(True)
plt.show()
plt.close()
plt.savefig("gsnrbit.png", dpi=600)












