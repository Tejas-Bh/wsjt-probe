# Why WSPR & The Proposed Signal Model

This section outlines why the Weak Signal Propagation Reporter (WSPR) protocol is uniquely suited for advanced propagation channel estimation, alongside the mathematical framework governing the proposed signal model.

## Why WSPR?
WSPR is uniquely optimized for this research due to its highly constrained and deterministic waveform transmission properties [cite: 24]:
* **Public and Deterministic Protocol:** The underlying specification is fully open and predictable.
* **Post-Decode Determinism:** Once a WSPR transmission is successfully decoded by a receiver, the core operational parameters become known:
  * Transmitter Callsign
  * Maidenhead Grid Locator
  * Reported Transmitter Power (dBm)

From this thin data payload, a receiver can deterministically reconstruct the exact digital structure that was sent:
* Source message bits
* Forward Error Correction (FEC) encoding patterns
* Interleaving sequence
* Four-frequency FSK symbol sequence
* **Ideal transmitted baseband waveform**

### Hardware Caveat
While the exact physical radio frequency (RF) waveform cannot be reproduced perfectly due to transmitter-specific oscillator offsets, phase noise, filtering, amplifier nonlinearities, and other hardware imperfections, the reconstructed signal serves as a highly accurate baseline approximation of the intended transmission. This precise baseline creates a unique remote sensing opportunity that does not exist in conventional opportunistic communication systems.

---

## Proposed Signal Model

By treating the reconstructed ideal baseband waveform as a reference probing signal, we can evaluate the propagation path via a classical system model.

The received complex baseband signal $y(t)$ is modeled as:

$$y(t) = h(t) \cdot x(t) + n(t)$$

Where:
* $x(t)$ represents the deterministically reconstructed, ideal transmitted baseband waveform.
* $h(t)$ is the time-varying complex propagation channel response.
* $n(t)$ represents additive white Gaussian receiver noise combined with unmodeled atmospheric or hardware effects.

### Core Estimation Objective
The primary operational task is to estimate the complex channel function $\hat{h}(t)$, and subsequently isolate the **channel residual** $r(t)$:

$$r(t) = y(t) - \hat{h}(t) \cdot x(t)$$

This residual $r(t)$ contains unmodeled signal components, scattered energy, and higher-order structural variations not captured by basic channel tracking models, serving as a rich potential source of atmospheric data.
