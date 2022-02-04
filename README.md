# Instantaneous Frequency
Demonstrating different interpretations and calculations of instantaneous frequency. The figures included in this package attempt to assist in one's understanding of the underlying processes. This educational tool come out of work done by Cole van Jaarsveldt, Prof. Gareth W. Peters, Prof. Mike Chantler, and Dr Matthew Ames. This package is free for all to use and should be cited appropriately. Some of the figures included are shown below.

## Time Series

All of the analysis techniques demonstrated below used the following time series. The example time series consists of two pure harmonics also known as sinusoids which is generic term for any Sine or Cosine function. The high frequency component and the low frequency component have frequencies of 5 rad/s and 1 rad/s respectively.  

![](./README_images/Time_series.png)

## Frequency (f) versus Angular Frequency ($\omega$)

Frequency is measured in Hertz (Hz) which has units of $s^{-1}$ which is per second. Angular frequency is measured in radians per second or $rad/s$ or $rad.s^{-1}$. The Diagram above demonstrates how these may be understood and used interchangeably.

![](./README_images/frequency.png)

## Fourier Transform

The Fourier Transform of discrete time series, $x[t]$, of length $N$ is:

$X[t] = \sum_{n=0}^{N-1}x[n]e^{-i\frac{2\pi}{N}tn}, \text{ or}$

$X[t] = \sum_{n=0}^{N-1}x[n]\Big(\text{cos}(\frac{2\pi}{N}tn) - i\text{sin}(\frac{2\pi}{N}tn)\Big)$.

In the figure below, the two frequency structures are clearly discernible, but there are also additional frequency structures present. These additional structures are referred to as "ghost frequencies" because there are no actual structures present at those frequencies, but the Fourier transform produces the figure below when it is performed on a time series with a non-integer number of wavelengths. This will be discussed further in the next section.

![](./README_images/FT.png)

The heatmap below is another way of plotting the above figure. It is not necessary here as there is only frequency content and no temporal (or time) content. Why this is here will  be more clear when we look at the short-time Fourier transform (STFT) in a later section.

![](./README_images/FT_Heat_plot.png)

## Ghost Frequencies

![](./README_images/Time_series_truncated.png)

![](./README_images/FT_truncated.png)

## Fourier Transform with Tapering

![](./README_images/FT_demonstration.png)

![](./README_images/FT_tapered.png)

![](./README_images/FT_Heat_plot_tapered.png)

### Short-Time Fourier Transform

![](./README_images/STFT.png)

![](./README_images/STFT_demonstration.png)



### Morlet Wavelet Transform



![](./README_images/Morlet_wavelet_demonstration.png)



![](./README_images/Morlet_wavelet_fixed.png)



![](./README_images/Morlet_fixed_convolution.png)



![](./README_images/Morlet_fixed_spectrum.png)

![](./README_images/Morlet_wavelet_adjust.png)



![](./README_images/Morlet_adjust_convolution.png)

![](./README_images/Morlet_adjust_spectrum.png)
