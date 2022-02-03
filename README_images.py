
import textwrap
import scipy as sp
import numpy as np
import seaborn as sns
from scipy import signal as sig
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

sns.set(style='darkgrid')

start = 0
end = 5 * np.pi
time_points = 1001

time = np.linspace(start, end, time_points)
time_series = np.cos(time) + np.cos(5 * time)

# Figure 1

fig, axs = plt.subplots(3, 1)
fig.tight_layout(pad=1.0)
axs[0].plot(time, time_series, LineWidth=2)
axs[0].set_title('Example Time Series', fontsize=12)
axs[0].set_ylabel('Displacement', fontsize=10)
axs[0].set_xticks((0, 1 * np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi))
axs[0].set_xticklabels(('', '', '', '', '',''), fontsize=10)
axs[0].set_yticks((-2, -1, 0, 1, 2))
axs[0].set_yticklabels(('-2', '-1', '0', '1', '2'), fontsize=10)

axs[1].plot(time, np.cos(5 * time), LineWidth=2)
axs[1].set_title('High Frequency Component', fontsize=12)
axs[1].set_ylabel('Displacement', fontsize=10)
axs[1].set_xticks((0, 1 * np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi))
axs[1].set_xticklabels(('', '', '', '', '', ''), fontsize=10)
axs[1].set_yticks((-2, -1, 0, 1, 2))
axs[1].set_yticklabels(('-2', '-1', '0', '1', '2'), fontsize=10)

axs[2].plot(time, np.cos(time), LineWidth=2)
axs[2].set_title('Low Frequency Component', fontsize=12)
axs[2].set_ylabel('Displacement', fontsize=10)
axs[2].set_xticks((0, 1 * np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi))
axs[2].set_xticklabels((r'$0$', r'$\pi$', r'2$\pi$', r'3$\pi$', r'4$\pi$', r'5$\pi$'), fontsize=10)
axs[2].set_yticks((-2, -1, 0, 1, 2))
axs[2].set_yticklabels(('-2', '-1', '0', '1', '2'), fontsize=10)
axs[2].set_xlabel('Time (s)', fontsize=10)
plt.savefig('README_images/Time_series.png')
plt.show()

# Figure 2

x_top = np.linspace(-1, 1, 100)
x2_top = x_top ** 2
y_top = np.sqrt(1 - np.linspace(-1, 1, 100) ** 2)
x_bottom = np.linspace(1, -1, 100)
x2_bottom = x_top ** 2
y_bottom = -np.sqrt(1 - np.linspace(-1, 1, 100) ** 2)

x_sin = np.linspace(0, 2 * np.pi, 100)

x_frac = 1.25 * np.linspace(-1/2, 1, 1000)
y_frac = np.sqrt(1.25 ** 2 - x_frac ** 2)

plt.title(textwrap.fill('Plot Demonstrating Frequency and Angular Frequency for 1 Second of Activity', 50))
plt.plot(x_top, y_top, c='b')
plt.plot(x_frac, y_frac, c='k')
plt.plot(np.linspace(-0.625, -0.5875, 100), np.linspace(y_frac[0], y_frac[0] + 0.1, 100), c='k')
plt.plot(np.linspace(-0.625, -0.425, 100), np.linspace(y_frac[0], y_frac[0] + 0.025, 100), c='k')
plt.plot(np.linspace(0, 1, 100), np.zeros(100), c='k')
plt.plot(np.linspace(0, -1/2, 100), np.linspace(0, np.sqrt(3)/2, 100), c='k')
plt.text(0.1, 0.25, r'$\theta$')
plt.plot(x_bottom, y_bottom, c='b')
plt.plot(np.linspace(2, 2 * np.pi + 2, 100), np.sin(x_sin), c='green')
plt.scatter(-1/2, np.sqrt(3)/2, s=50, c='k', zorder=5)
plt.scatter(1, 0, s=50, c='k', zorder=5)
plt.scatter(2 + (2/3) * np.pi, np.sqrt(3)/2, s=50, c='k', zorder=5)
plt.scatter(2, 0, s=50, c='k', zorder=5)
plt.plot(np.linspace(2, 2 + (2/3) * np.pi, 100), np.zeros(100), c='k')
plt.plot(2 + (2/3) * np.pi * np.ones(100), np.linspace(0, np.sqrt(3)/2, 100), c='k')
plt.plot(2 * np.ones(100), np.linspace(-0.4, -0.2, 100), c='k')
plt.plot((2 + (2/3) * np.pi) * np.ones(100), np.linspace(-0.4, -0.2, 100), c='k')
plt.plot(np.linspace(2, (2 + (2/3) * np.pi), 100), -0.3 * np.ones(100), c='k')
plt.plot(np.linspace(2, (2 + (2/3) * np.pi), 100), -0.6 * np.ones(100), c='k')
plt.plot(np.linspace((2 + (2/3) * np.pi) - 0.2, (2 + (2/3) * np.pi), 100), np.linspace(-0.55, -0.6, 100), c='k')
plt.plot(np.linspace((2 + (2/3) * np.pi) - 0.2, (2 + (2/3) * np.pi), 100), np.linspace(-0.65, -0.6, 100), c='k')
plt.text(2 + (1/3) * np.pi - 0.15, -0.5, r'$\theta$')
plt.plot(2 * np.ones(100), np.linspace(-1.75, 1.75, 100), 'lightgray', linestyle='--', zorder=1)
plt.plot(2 + np.pi * np.ones(100), np.linspace(-1.75, 1.75, 100), 'lightgray', linestyle='--', zorder=1)
plt.plot(2 + 2 * np.pi * np.ones(100), np.linspace(-1.75, 1.75, 100), 'lightgray', linestyle='--', zorder=1)
plt.text(0.75, 1.1, r'$\theta rad/s$')
plt.text(2.7, -0.9, r'$\frac{\theta}{2\pi} Hz$')
plt.xlim(-1.5, 8.5)
plt.ylim(-2, 2)
plt.grid(b=None)
plt.yticks([0], '')
plt.xticks([2, 2 + np.pi, 2 + 2 * np.pi], [r'$0$', r'$\pi$', r'$2\pi$'])
plt.savefig('README_images/frequency.png')
plt.show()

# Figure 3

ax = plt.subplot(111)
plt.plot(time, time_series, LineWidth=2, label=textwrap.fill('Original time series', 11))
plt.plot(time[:801], time_series[:801], '-', LineWidth=2, label=textwrap.fill('Truncated time series', 11))
plt.plot(4 * np.pi * np.ones(101), np.linspace(-2.5, 2.5, 101), '--', LineWidth=2, label=textwrap.fill('Boundary', 11))
plt.title('Example Time Series and Truncated Time Series', fontsize=14, pad=15)
plt.xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi],
           ['$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'], fontsize=10)
plt.xlabel('Time(s)', fontsize=12)
plt.yticks((-2, -1, 0, 1, 2), fontsize=10)
plt.ylabel('Displacement', fontsize=12)

box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.02, box_0.y0, box_0.width * 0.84, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
plt.savefig('README_images/Time_series_truncated.png')
plt.show()

# Figure 4

N = 1001
N_trunc = 801
end_trunc = 4 * np.pi

sample_points = N - 1
sample_spacing = end / sample_points

f_time_series = fft(time_series)
f_time = fftfreq(sample_points, sample_spacing)[:sample_points//2] * 2 * np.pi  # convert to angular frequnecy

sample_points_trunc = N_trunc - 1
sample_spacing_trunc = end_trunc / sample_points_trunc

f_time_series_trunc = fft(time_series[:801])
f_time_trunc = fftfreq(sample_points_trunc,
                       sample_spacing_trunc)[:sample_points_trunc//2] * 2 * np.pi  # convert to angular frequnecy

plt.plot(1 * np.ones(100), np.linspace(0, 1, 100), 'r-', label=r'$\omega = 1$', LineWidth=2)
plt.plot(5 * np.ones(100), np.linspace(0, 1, 100), 'g-', label=r'$\omega = 5$', LineWidth=2)
plt.plot(f_time, 2.0 / sample_points * np.abs(f_time_series[0:sample_points//2]), LineWidth=2,
         label=textwrap.fill('Fourier transform of original time series', 20))
plt.plot(f_time_trunc, 2.0 / sample_points_trunc * np.abs(f_time_series_trunc[0:sample_points_trunc//2]), '-',
         LineWidth=2, label=textwrap.fill('Fourier transform of truncated time series', 21))
plt.title(textwrap.fill('Fourier Transform of Example Time Series and Truncated Time Series', 40), fontsize=14, pad=15)
plt.plot(7 * np.ones(101), np.linspace(0.15, 0.35, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(7, 7.5, 101), np.linspace(0.15, 0.20, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(7, 6.5, 101), np.linspace(0.15, 0.20, 101), 'k-', LineWidth=3)
plt.plot(2.5 * np.ones(101), np.linspace(0.2, 0.4, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(2.5, 3, 101), np.linspace(0.2, 0.25, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(2.5, 2, 101), np.linspace(0.2, 0.25, 101), 'k-', LineWidth=3)
plt.plot(17.5 * np.ones(101), np.linspace(0.05, 0.25, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(17.5, 18, 101), np.linspace(0.05, 0.10, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(17.5, 17, 101), np.linspace(0.05, 0.10, 101), 'k-', LineWidth=3, label=textwrap.fill('Ghost frequencies', 12))
plt.xlabel('Angular frequency (rad.s$^{-1}$)', fontsize=12)
plt.xticks((0, 5, 10, 15, 20, 25), fontsize=10)
plt.ylabel('Amplitude', fontsize=12)
plt.yticks((0, 0.5, 1), fontsize=10)
plt.ylim(-0.025, 1.025)
plt.xlim(-1, 27.5)
plt.legend(loc='upper right', fontsize=10)
plt.savefig('README_images/FT_truncated.png')
plt.show()

# Figure 5

sample_points = time_points - 1
sample_spacing = end / sample_points

f_time_series = fft(time_series)
f_time = fftfreq(sample_points, sample_spacing)[:sample_points//2] * 2 * np.pi  # convert to angular frequnecy

plt.plot(f_time, 2.0 / sample_points * np.abs(f_time_series[0:sample_points//2]), LineWidth=2)
plt.plot(np.ones(100), np.linspace(0, 1, 100), 'r-', label=r'$\omega = 1$', LineWidth=2)
plt.plot(5 * np.ones(100), np.linspace(0, 1, 100), 'g-', label=r'$\omega = 5$', LineWidth=2)
plt.plot(7 * np.ones(101), np.linspace(0.15, 0.35, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(7, 7.5, 101), np.linspace(0.15, 0.20, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(7, 6.5, 101), np.linspace(0.15, 0.20, 101), 'k-', LineWidth=3)
plt.plot(2.5 * np.ones(101), np.linspace(0.2, 0.4, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(2.5, 3, 101), np.linspace(0.2, 0.25, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(2.5, 2, 101), np.linspace(0.2, 0.25, 101), 'k-', LineWidth=3)
plt.plot(17.5 * np.ones(101), np.linspace(0.05, 0.25, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(17.5, 18, 101), np.linspace(0.05, 0.10, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(17.5, 17, 101), np.linspace(0.05, 0.10, 101), 'k-', LineWidth=3, label=textwrap.fill('Ghost frequencies', 12))
plt.title('Fourier Transform of Example Time Series', fontsize=14)
plt.xlabel('Angular Frequency (rad.s$^{-1}$)', fontsize=12)
plt.xticks((0, 5, 10, 15, 20, 25), fontsize=10)
plt.ylabel('Amplitude', fontsize=12)
plt.yticks((0, 0.5, 1), fontsize=10)
plt.ylim(-0.025, 1.025)
plt.xlim(-2.5, 27.5)
plt.legend(loc='upper right', fontsize=10)
plt.savefig('README_images/FT.png')
plt.show()

# Figure 3

x_hs, y, z = [0, 5 * np.pi], f_time, (2.0 / sample_points * np.abs(f_time_series[0:sample_points//2])).reshape(-1, 1)
z_min, z_max = 0, np.abs(z).max()

ax = plt.subplot(111)
plt.title('Sample Fourier Transform Spectrum', fontsize=14)
ax.pcolormesh(x_hs, y, np.abs(z), vmin=z_min, vmax=z_max)
ax.plot(x_hs, 5 * np.ones_like(x_hs), 'g--', label=r'$\omega = 5$', Linewidth=2)
ax.plot(x_hs, 1 * np.ones_like(x_hs), 'r--', label=r'$\omega = 1$', Linewidth=2)
ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi])
ax.set_xticklabels(['$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'], fontsize=10)
ax.set_yticks([0, 5, 10])
ax.set_yticklabels(['$0$', '$5$', '$10$'], fontsize=10)
plt.ylabel(r'Angular Frequency (rad.s$^{-1}$)', fontsize=12)
plt.ylim(0, 10)
plt.xlabel('Time (s)', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
plt.savefig('README_images/FT_Heat_plot.png')
plt.show()

# Figure 4

ext_hann_window = sig.get_window(window='hann', Nx=1001, fftbins=False)

ax = plt.subplot(111)
plt.plot(time, time_series, LineWidth=2, label=textwrap.fill('Original time series', 11))
plt.plot(time, ext_hann_window * time_series, 'orange', LineWidth=2,
         label=textwrap.fill('Tapered time series', 11))
plt.scatter(time[500], (ext_hann_window * time_series)[500], s=50, c='orange', zorder=5,
            label=textwrap.fill('Instantaneous frequency estimated', 13))
plt.plot(time, 2 * ext_hann_window, 'k-', LineWidth=2, label=textwrap.fill('Hann window', 11))
plt.title(textwrap.fill('Fourier Analysis Example Time Series Tapered', 40), fontsize=14, pad=10)
plt.xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi],
           ['$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'], fontsize=10)
plt.xlabel('Time(s)', fontsize=12)
plt.yticks((-2, -1, 0, 1, 2), fontsize=10)
plt.ylabel('Displacement', fontsize=12)

box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.02, box_0.y0, box_0.width * 0.84, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
plt.savefig('README_images/FT_demonstration.png')
plt.show()

# Figure 5

sample_points = time_points - 1
sample_spacing = end / sample_points

f_time_series_tapered = fft(ext_hann_window * time_series)
f_time = fftfreq(sample_points, sample_spacing)[:sample_points//2] * 2 * np.pi  # convert to angular frequnecy

plt.plot(np.ones(100), np.linspace(0, 1, 100), 'r-', label=r'$\omega = 1$', LineWidth=2)
plt.plot(5 * np.ones(100), np.linspace(0, 1, 100), 'g-', label=r'$\omega = 5$', LineWidth=2)
plt.plot(np.linspace(17.5, 17, 101), np.linspace(0.05, 0.10, 101), 'k-', LineWidth=3, label=textwrap.fill('Ghost frequencies', 30))
plt.plot(f_time, 2.0 / sample_points * np.abs(f_time_series[0:sample_points//2]), LineWidth=2, label='Fourier transform')
plt.plot(f_time, 2.0 / sample_points * np.abs(f_time_series_tapered[0:sample_points//2]), LineWidth=2, label='Fourier transform tapered')
plt.plot(7 * np.ones(101), np.linspace(0.15, 0.35, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(7, 7.5, 101), np.linspace(0.15, 0.20, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(7, 6.5, 101), np.linspace(0.15, 0.20, 101), 'k-', LineWidth=3)
plt.plot(2.5 * np.ones(101), np.linspace(0.2, 0.4, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(2.5, 3, 101), np.linspace(0.2, 0.25, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(2.5, 2, 101), np.linspace(0.2, 0.25, 101), 'k-', LineWidth=3)
plt.plot(17.5 * np.ones(101), np.linspace(0.05, 0.25, 101), 'k-', LineWidth=3)
plt.plot(np.linspace(17.5, 18, 101), np.linspace(0.05, 0.10, 101), 'k-', LineWidth=3)
plt.title('Fourier Transform of Example Time Series', fontsize=14)
plt.xlabel('Angular Frequency (rad.s$^{-1}$)', fontsize=12)
plt.xticks((0, 5, 10, 15, 20, 25), fontsize=10)
plt.ylabel('Amplitude', fontsize=12)
plt.yticks((0, 0.5, 1), fontsize=10)
plt.ylim(-0.025, 1.025)
plt.xlim(-2.5, 27.5)
plt.legend(loc='upper right', fontsize=10)
plt.savefig('README_images/FT_tapered.png')
plt.show()

# Figure 6

fig, axs = plt.subplots(1, 2)
plt.suptitle(textwrap.fill('Fourier Transforms of Original Time Series and Tapered Time Series', 45), fontsize=16)

axs[0].set_title(textwrap.fill('Time Series', 30), fontsize=12)
axs[0].pcolormesh(x_hs, y, np.abs(z), vmin=z_min, vmax=z_max)
axs[0].plot(x_hs, 5 * np.ones_like(x_hs), 'g--', label=r'$\omega = 5$', Linewidth=2)
axs[0].plot(x_hs, 1 * np.ones_like(x_hs), 'r--', label=r'$\omega = 1$', Linewidth=2)
axs[0].set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi])
axs[0].set_xticklabels(['$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'], fontsize=10)
axs[0].set_yticks([0, 5, 10])
axs[0].set_yticklabels(['$0$', '$5$', '$10$'], fontsize=10)
axs[0].set_ylabel(r'Angular frequency (rad.s$^{-1}$)', fontsize=12)
axs[0].set_ylim(0, 10)
axs[0].set_xlabel('Time (s)', fontsize=12)
box_0 = axs[0].get_position()
axs[0].set_position([box_0.x0 - 0.025, box_0.y0, 1.05 * box_0.width, 0.9 * box_0.height])

x_hs_taper, y_taper, z_taper = \
    [0, 5 * np.pi], f_time, (2.0 / sample_points * np.abs(f_time_series_tapered[0:sample_points//2])).reshape(-1, 1)
z_min_taper, z_max_taper = 0, np.abs(z_taper).max()

axs[1].set_title(textwrap.fill('Tapered Time Series', 40), fontsize=12)
axs[1].pcolormesh(x_hs_taper, y_taper, np.abs(z_taper), vmin=z_min_taper, vmax=z_max_taper)
axs[1].plot(x_hs, 5 * np.ones_like(x_hs), 'g--', label=r'$\omega = 5$', Linewidth=2)
axs[1].plot(x_hs, 1 * np.ones_like(x_hs), 'r--', label=r'$\omega = 1$', Linewidth=2)
axs[1].set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi])
axs[1].set_xticklabels(['$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'], fontsize=10)
axs[1].set_yticks([0])
axs[1].set_yticklabels([' '], fontsize=10)
axs[1].set_ylim(0, 10)
axs[1].set_xlabel('Time (s)', fontsize=12)
axs[1].legend(loc='upper right')
box_1 = axs[1].get_position()
axs[1].set_position([box_1.x0 - 0.025, box_1.y0, 1.05 * box_1.width, 0.9 * box_1.height])
plt.savefig('README_images/FT_Heat_plot_tapered.png')
plt.show()

# Figure 7


def stft_custom(time: np.ndarray, time_series: np.ndarray, window: str = 'hann', window_width: int = 256,
                plot: bool = False, angular_frequency: bool = True) -> (np.ndarray, np.ndarray, np.ndarray):
    try:
        # normalise and left Riemann sum approximate window
        window_func = sig.get_window(window=window, Nx=(window_width + 1), fftbins=False)
        window_func = window_func[:-1] / np.sum(window_func[:-1])
    except:
        if window == 'none':
            window_func = np.ones(window_width) / np.sum(np.ones(window_width))

    # how much must be added to create an integer multiple of window length
    addition = int(int(window_width / 2) - int((len(time_series) + window_width) % int(window_width / 2)))

    # integer multiple of window length
    time_series_new = np.hstack(
        (np.zeros(int(window_width / 2)), time_series, np.zeros(int(window_width / 2 + addition))))

    # storage of windowed Fourier transform
    z = np.zeros((int(len(time_series_new) / (window_width / 2) - 1), int(window_width)))

    # calculate sampling rate for frequency
    sampling_rate = len(time) / (time[-1] - time[0])
    # sampling_rate = len(time) / np.pi

    for row in range(np.shape(z)[0]):
        # multiply window_func onto interval 'row'
        z[row, :] = window_func * time_series_new[int(row * (window_width / 2)):int((row + 2) * (window_width / 2))]
    # real Fourier transform the matrix
    z = np.transpose(sp.fft.rfft(z, n=window_width))
    # calculate frequency vector
    if angular_frequency:
        constant = 2 * np.pi
    else:
        constant = 1
    f = constant * np.linspace(0, (sampling_rate / 2), int(window_width / 2 + 1))
    # calculate time vector
    t = np.linspace(time[0], time[-1] + (time[-1] - time[0]) * (addition / len(time_series)),
                    int(len(time_series_new) / (window_width / 2) - 1))

    if plot:
        plt.pcolormesh(t, f, np.abs(z), vmin=0, vmax=np.max(np.max(np.abs(z))))
        plt.ylabel('f')
        plt.xlabel('t')
        plt.show()

    return t, f, z


x_hs, y, z = stft_custom(time=time, time_series=time_series, window_width=512)
z_min, z_max = 0, np.abs(z).max()

ax = plt.subplot(111)
plt.title('Sample Short-Time Fourier Transform Spectrum', fontsize=14)
ax.pcolormesh(x_hs, y, np.abs(z), vmin=z_min, vmax=z_max)
ax.plot(x_hs, 5 * np.ones_like(x_hs), 'g--', label=r'$\omega = 5$', Linewidth=2)
ax.plot(x_hs, 1 * np.ones_like(x_hs), 'r--', label=r'$\omega = 1$', Linewidth=2)
ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi])
ax.set_xticklabels(['$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'], fontsize=10)
ax.set_yticks([0, 5, 10])
ax.set_yticklabels(['$0$', '$5$', '$10$'], fontsize=10)
plt.ylabel(r'Angular Frequency (rad.s$^{-1}$)', fontsize=12)
plt.ylim(0, 10)
plt.xlabel('Time (s)', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
plt.savefig('README_images/STFT.png')
plt.show()

# Figure 8 (Old figure 1)

n = 512
hann_window = sig.get_window(window='hann', Nx=n + 1, fftbins=False)

extended_time_series = np.empty(1001 + 24 + 256)
extended_time_series[:] = np.nan
ext_time_series = extended_time_series.copy()
ext_time_series[256:1257] = time_series
ext_time = np.linspace(5 * np.pi * (-256 / 1000), 5 * np.pi * (1024 / 1000), 1001 + 24 + 256)

extended_time_series_1 = extended_time_series.copy()
extended_time_series_1[:513] = ext_time_series[:513] * hann_window

extended_time_series_2 = extended_time_series.copy()
extended_time_series_2[256:769] = ext_time_series[256:769] * hann_window

ax = plt.subplot(111)
plt.plot(time, time_series, LineWidth=2)
plt.plot(ext_time[:513], 2 * hann_window, LineWidth=2, c='k', label='Hann window')
plt.plot(ext_time, extended_time_series_1, LineWidth=2, c='orange',
         label=textwrap.fill('Tapered time series segment over 1st interval', 25))
plt.scatter(ext_time[256], 2,
            label=textwrap.fill('Instantaneous frequency estimated for 1st interval', 30), s=50, c='orange')
plt.plot(ext_time, extended_time_series_2, LineWidth=2, c='green',
         label=textwrap.fill('Tapered time series segment over 2nd interval', 25))
plt.plot(ext_time[256:769], 2 * hann_window, LineWidth=2, c='k')
plt.scatter(ext_time[512], ext_time_series[512],
            label=textwrap.fill('Instantaneous frequency estimated for 2st interval', 30), s=50, c='green')
plt.title('Short-Time Fourier Transform Demonstration', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.xticks((5 * np.pi * (-256 / 1000), 0, 5 * np.pi * (256 / 1000), 5 * np.pi * (512 / 1000), 5 * np.pi * (768 / 1000),
            5 * np.pi * (1024 / 1000)),
           ('-256', '0', '256', '512', '768', '1024'), fontsize=10)
for i in [5 * np.pi * (-256 / 1000), 0, 5 * np.pi * (256 / 1000), 5 * np.pi * (512 / 1000), 5 * np.pi * (768 / 1000),
          5 * np.pi * (1024 / 1000)]:
    plt.plot(i * np.ones(101), np.linspace(-2.5, 2.5, 101), 'k--')
for j in [5 * np.pi * (-256 / 1000), 0, 5 * np.pi * (256 / 1000), 5 * np.pi * (512 / 1000)]:
    plt.plot()
jump = 0.2
x1 = 5 * np.pi * (-256 / 1000)
x2 = 5 * np.pi * (256 / 1000)
x1_lower = 0
interval = 1
for k in [5 * np.pi * (-256 / 1000), 0, 5 * np.pi * (256 / 1000), 5 * np.pi * (512 / 1000)]:
    plt.plot(np.linspace(k, k + 5 * np.pi * (512 / 1000), 101), 2.6 + jump * np.ones(101), 'k-')
    plt.plot(x1 * np.ones(101), np.linspace(2.6 + jump, 2.6 + jump - 0.1, 101), 'k-')
    plt.plot(x2 * np.ones(101), np.linspace(2.6 + jump, 2.6 + jump - 0.1, 101), 'k-')

    plt.plot(np.linspace(k + 5 * np.pi * (256 / 1000), k + 5 * np.pi * (512 / 1000), 101), -2.6 - jump * np.ones(101),
             'k-')
    plt.plot(x1_lower * np.ones(101), np.linspace(-2.6 - jump, -2.6 - jump + 0.1, 101), 'k-')
    plt.plot(x2 * np.ones(101), np.linspace(-2.6 - jump, -2.6 - jump + 0.1, 101), 'k-')

    plt.text(x1 + np.pi - 1.0, 2.6 + jump + 0.3, f'Measure interval {interval}', fontsize=6)
    plt.text(x1_lower + 0.5, -2.6 - jump - 0.3, f'Plot interval {interval}', fontsize=6)

    x1 += 5 * np.pi * (256 / 1000)
    x2 += 5 * np.pi * (256 / 1000)
    x1_lower += 5 * np.pi * (256 / 1000)
    jump += 0.2
    interval += 1
plt.ylim(-4, 4)
plt.ylabel('Displacement', fontsize=12)
plt.yticks((-2, -1, 0, 1, 2), ('-2', '-1', '0', '1', '2'), fontsize=10)

box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.05, box_0.y0, box_0.width * 0.85, box_0.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)
plt.savefig('README_images/STFT_demonstration.png')
plt.show()

# Figure 4
