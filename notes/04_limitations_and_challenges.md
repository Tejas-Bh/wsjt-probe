# Technical Limitations & Challenges

While waveform-level extraction yields significantly more data than classical spot logs, several stark physical and logistical boundaries must be factored into the research design.

## 1. Ultra-Narrow Bandwidth Constraint
The WSPR protocol operates within an extremely narrow frequency profile, occupying a total bandwidth of only approximately 6 Hz. This introduces several profound signal processing limitations:
* **Extremely Poor Delay Resolution:** According to the inverse relationship between bandwidth and time resolution ($\Delta t \approx 1/B$), a 6 Hz signal cannot resolve distinct multipath delays directly.
* **No High-Resolution Impulse Response:** It is impossible to generate standard time-delay channel profiles to map individual ionospheric layer heights.

<!-- I mean, I don't want to completely exclude multipath profiling from the picture.
### Impact on Scope
This approach does not attempt time-domain multipath profiling. Instead, signal processing must focus exclusively on time-series analysis of the narrow-band channel's **phase evolution, frequency stability, complex coherence, and amplitude fading characteristics** over the course of the 2-minute window. -->

---

## 2. Imperfect Waveform Reconstruction
The reconstructed reference signal $x(t)$ represents the *ideal, mathematically perfect* transmission specified by the WSPR protocol, rather than the physical RF energy emitted from the antenna. Differences inevitably manifest due to real-world commercial and homebrew hardware variances:
* Frequency oscillator offsets and inherent phase noise
* Power amplifier (PA) nonlinearities and intermodulation distortions
* Internal analog and digital filtering characteristics
* Baseband-to-RF timing errors and sampling jitter

### Mitigation
To ensure these hardware artifacts are not misidentified as atmospheric or ionospheric phenomena, robust calibration routines or baseline multi-receiver differential models must be introduced to subtract fixed transmitter/receiver quirks from the data.

---

## 3. Storage and Infrastructure Availability
Standard WSPR collection systems discard raw data instantly, uploading only highly compressed, alphanumeric text summaries (such as callsign, SNR, grid locator) to central databases like WSPRNet. 

Implementing this methodology requires a fundamental shift toward capturing and storing **raw complex IQ recordings**. This demands substantial local computing resources and specialized SDR software.
