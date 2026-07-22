# Expanding HF Observables

Traditional amateur radio network analyses rely on heavily compressed metadata. This page contrasts existing high-frequency (HF) metrics against the high-dimensional observables enabled by waveform-level reconstruction.

## Existing HF Observables
Current distributed propagation networks (e.g., WSPRNet, PSKReporter, HamSCI) capture low-dimensional summary statistics optimized for data efficiency across large numbers of nodes:
* **Signal-to-Noise Ratio (SNR):** Received signal strength relative to local noise floor .
* **Doppler Frequency Shift:** Absolute carrier offset from nominal frequency.
* **Frequency Drift:** Rate of change of the frequency offset over time.
* **Path Availability & Link Persistence:** Binary indicators of whether a path exists and how long it remains active.

While these low-dimensional summaries successfully map large-scale effects (e.g., solar flares, geomagnetic storms, diurnal cycles), they discard the vast majority of physical information embedded in the original received analog waveform.

---

## Proposed Additional Waveform Observables
Rather than treating WSPR purely as a digital message payload, this framework processes the interaction between the received IQ data and the ideal reconstructed signal to extract fine-grained ionospheric metrics.

### 1. Complex Channel Response
By deriving an instantaneous channel estimate:

$$h(t) = \frac{y(t)}{x(t)}$$

We track the simultaneous evolution of both **amplitude** and **phase** across the transmission interval. Potential metrics include:
* **Phase Stability & Phase Noise:** Tracking fine-scale phase perturbations caused by localized ionospheric variations.
* **Channel Coherence:** The time window over which the channel remains highly predictable.
* **Amplitude Fluctuations:** Characterizing rapid fading dynamics.

### 2. Channel Coherence Metrics
Quantifying the speed and nature of channel variations within a single 110.6-second WSPR transmission window:
* **Coherence Time:** Evaluating the rate of channel decorrelation.
* **Phase Autocorrelation:** Uncovering cyclical or pseudo-periodic variations in the propagation path.
* **Channel Stationarity:** Assessing whether the physical statistics of the channel remain constant during the block.

### 3. Residual Waveform Analysis
Once the best-fit deterministic channel $\hat{h}(t)$ is calculated and subtracted, the remaining signal residual $r(t)$ can be statistically analyzed to find hidden structures:
* **Residual Power & Spectrum:** Assessing off-axis energy or Doppler broadening signatures.
* **Higher-Order Statistics:** Utilizing skewness and kurtosis to quantify non-Gaussian behavior induced by scattering.
* **Temporal Autocorrelation & Spectral Entropy:** Identifying structural order or total randomness within the unmodeled noise floor.
* **Fractal / Scale-Dependent Behavior:** Measuring multi-scale ionospheric turbulence dynamics.

<!-- Not sure if I'll add this

### Summary of Targets
It is hypothesized that structural shifts in these advanced metrics will act as direct remote indicators for [cite: 88, 89]:
* Traveling Ionospheric Disturbances (TIDs) [cite: 91]
* Plasma turbulence and E/F-layer scintillation [cite: 92, 93]
* Sporadic-E multi-path events [cite: 94]
* Fine-scale geomagnetic storm features [cite: 95] -->
