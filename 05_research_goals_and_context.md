# Core Research Goals & Scientific Context

This page details the guiding scientific questions of the project and outlines how this methodology bridges amateur radio data with rigorous space science frameworks.

## Central Research Questions
The investigative focus of this project is crystallized into two overlapping questions:

1. **Waveform-Level Value Discovery:** Can a direct waveform-level comparison between deterministically reconstructed WSPR signals and raw received IQ recordings reveal physical ionospheric propagation observables that are fully lost in conventional decoder text outputs?
2. **Residual Information Value:** Does the mathematical residual ($r(t) = y(t) - \hat{h}(t)x(t)$) between the ideal transmission and the received signal contain statistically significant, exploitable information about the ionosphere beyond the standard pillars of SNR, Doppler shift, and frequency drift?

---

## Scientific Context & Positioning
This methodology is not designed to replace successful existing crowdsourced citizen-science networks like HamSCI. Rather, it acts as a complementary extension by shifting our diagnostic capability one layer deeper down the signal processing chain.

```
Traditional Citizen Science Data:
[Raw Waveform] -> [Receiver Processing] -> [Extracted SNR/Doppler text logs] -> [Ionospheric Studies]

Proposed Paradigm Shift:
[Raw Waveform] -> [Waveform Reconstruction & Residual Extraction] -> [High-Dimensional Physics Metrics]
```

Instead of consuming static summaries generated *after* a receiver has finalized its decoding pass, this paradigm captures and evaluates the fragile physical information preserved in the raw time-series waveform before those reductive summary metrics are computed.

### Cross-Disciplinary Lineage
Conceptually, this project adapts, scales, and delivers established mathematical techniques from professional remote sensing disciplines into the widely deployed amateur radio infrastructure:
* **Radar Channel Estimation & Passive Radar:** Treating uncoordinated ambient transmissions as illuminators of opportunity.
* **Wireless Channel Sounding:** Using deterministic digital structures to audit unknown propagation mediums.
* **GPS/GNSS Scintillation Studies:** Adapting high-frequency amplitude and phase scintillation frameworks ($\sigma_\phi$ and $S_4$ indices) down to the HF spectrum.
* **Statistical Optics:** Emulating theories regarding coherent wave propagation through turbulent, scattering media.
