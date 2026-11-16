from scipy import fft
import matplotlib.pyplot as plt
import numpy as np

def f(t:np.ndarray):
    return t % 2*np.pi

fs = 100
t = np.linspace(0, 2*np.pi, fs, endpoint=False)
signal = f(t)

fft_values = fft.fft(signal)
frequencies = fft.fftfreq(signal.size, d=1 / fs)

frequencies = fft.fftshift(frequencies)
fft_values = fft.fftshift(fft_values)

fig, axes = plt.subplots(3, 1)
axes[0].plot(frequencies, fft_values.real)
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Real part")
axes[0].set_title("Real part of the FFT")
axes[0].grid(True)

axes[1].plot(frequencies, fft_values.imag)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Imaginary part")
axes[1].set_title("Imaginary part of the FFT")
axes[1].grid(True)

reconstructed_signal = fft.ifft(fft.ifftshift(fft_values))
axes[2].plot(t, reconstructed_signal.real)
axes[2].set_xlabel("Time (radians)")
axes[2].set_ylabel("Amplitude")
axes[2].set_title("Inverse FFT")
axes[2].grid(True)

fig.tight_layout()
plt.show()