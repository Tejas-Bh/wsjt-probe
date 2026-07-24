"""
Note: this file was AI-generated.
"""


"""
Watterson HF ionospheric propagation channel model.

This is a numpy re-implementation of the algorithm used in pathsim
(https://github.com/bubnikv/pathsim, Path.cpp / Path.h), which itself
implements the classic Watterson (1970) tapped-delay-line model with
Gaussian-shaped Doppler spectra, as popularized in HF modem testing
(CCIR/ITU-R "good"/"moderate"/"poor" channel presets).

Algorithm, per path/ray:
    1. Generate complex white Gaussian noise (I, Q ~ N(0, 1), independent).
    2. Low-pass filter it with a Gaussian-shaped FIR filter whose 2-sigma
       (two-sided) bandwidth equals the path's Doppler spread. This turns
       white noise into a complex Gaussian process with the correlation
       bandwidth (fading rate) that the spread parameter specifies; its
       envelope |.| is Rayleigh-distributed, which is the "R" in Watterson.
    3. Normalize the fading process to unit average power. pathsim does
       this analytically with the precomputed equivalent-noise-bandwidth
       constant KGNB of its filter (`gain_coeff * sqrt(rate/(4*spread*KGNB))`);
       here we do the equivalent thing numerically off of the actual
       filter that was designed, so the result is independent of exactly
       how the Gaussian filter is discretized.
    4. Apply the path's differential delay to the input signal.
    5. Complex-multiply the delayed input by the fading process (this is
       the actual "fading").
    6. Complex-multiply by a unit-amplitude NCO exp(j*2*pi*f_shift*t) to
       add a constant Doppler/frequency offset for that path.
    7. Scale by the path's relative gain and sum all paths.

The reference C++ implementation runs in real time and therefore
generates the Gaussian noise at a low rate (12.8 / 64 / 320 Hz, chosen
from the requested spread) and upsamples it to the working sample rate
with a chain of x5 polyphase interpolators. Since this module transforms
a complete numpy array rather than a live sample stream, that multirate
machinery is unnecessary: the Gaussian shaping filter is designed and
applied directly at the signal's sample rate, which is mathematically
equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Equivalent noise bandwidth of pathsim's Gaussian shaping filter, normalized
# to a 2-sigma bandwidth of 1 Hz (see Path.cpp: KGNB). Kept here for
# reference / for callers that want the closed-form pathsim-style gain
# formula instead of the numeric normalization used below.
KGNB = 0.62665707


def _gaussian_doppler_filter(spread_hz: float, fs: float) -> np.ndarray:
    """
    Design a (real-valued, unit-DC-gain) Gaussian-shaped FIR low-pass filter
    whose 2-sigma two-sided bandwidth equals `spread_hz`.

    This is the CCIR/Watterson Doppler power spectrum shape: a Gaussian
    frequency response of standard deviation sigma_f = spread_hz / 2, whose
    impulse response is itself Gaussian with sigma_t = 1 / (2*pi*sigma_f).
    """
    if spread_hz <= 0:
        # No spread requested -> pure gain, no fading bandwidth limiting.
        return np.array([1.0])

    sigma_f = spread_hz / 2.0
    sigma_t = 1.0 / (2.0 * np.pi * sigma_f)

    # Truncate the (infinite) Gaussian impulse response at +/- 4 sigma_t,
    # which captures effectively all of its energy.
    half_len = max(1, int(np.ceil(4.0 * sigma_t * fs)))
    n = np.arange(-half_len, half_len + 1)
    t = n / fs
    h = np.exp(-0.5 * (t / sigma_t) ** 2)
    h /= h.sum()  # unit gain at DC
    return h


def _complex_gaussian_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Independent complex Gaussian noise, I and Q ~ N(0, 1).

    pathsim generates this with a polar (Marsaglia) Box-Muller rejection
    sampler over the unit disk (see Rayleigh::sample() in Path.cpp); here
    we draw the equivalent distribution directly with numpy, which is
    exact and fully vectorized.
    """
    i = rng.standard_normal(n)
    q = rng.standard_normal(n)
    return i + 1j * q


@dataclass
class WattersonPath:
    """One ionospheric ray / propagation mode of a Watterson channel."""

    delay_s: float = 0.0             # differential propagation delay, seconds
    doppler_spread_hz: float = 1.0   # 2-sigma fading bandwidth, Hz
    doppler_shift_hz: float = 0.0    # constant frequency/Doppler offset, Hz
    gain: float = 1.0                # relative linear amplitude of this path


class WattersonChannel:
    """
    Watterson multipath fading channel.

    Example
    -------
    >>> # CCIR "moderate" 2-path HF channel: 1 ms delay spread, 1 Hz Doppler
    >>> # spread on each path, equal gains.
    >>> paths = [
    ...     WattersonPath(delay_s=0.0,    doppler_spread_hz=1.0, gain=1/np.sqrt(2)),
    ...     WattersonPath(delay_s=1e-3,   doppler_spread_hz=1.0, gain=1/np.sqrt(2)),
    ... ]
    >>> channel = WattersonChannel(paths, fs=8000.0, seed=1234)
    >>> faded_iq = channel.apply(iq)   # iq: complex numpy array in, complex numpy array out
    """

    def __init__(self, paths: List[WattersonPath], fs: float, seed: Optional[int] = None):
        if not paths:
            raise ValueError("at least one WattersonPath is required")
        self.paths = paths
        self.fs = float(fs)
        self.rng = np.random.default_rng(seed)

    def _fade(self, n_samples: int, spread_hz: float) -> np.ndarray:
        """Generate n_samples of unit-average-power complex Rayleigh fading."""
        if spread_hz <= 0:
            # No spread requested: pathsim treats this as "no fading at all"
            # (Rayleigh::init, spread < 0.1 -> constant gain, no randomness),
            # not as unfiltered per-sample noise. Return a constant unit
            # envelope so this path behaves as a clean, deterministic tap --
            # useful for validating delay/Doppler estimators before adding
            # random fading into the mix.
            return np.ones(n_samples, dtype=complex)

        h = _gaussian_doppler_filter(spread_hz, self.fs)
        pad = len(h) // 2
        noise = _complex_gaussian_noise(n_samples + 2 * pad, self.rng)
        fading = np.convolve(noise, h, mode="valid")[:n_samples]

        # Normalize to unit average power (equivalent to pathsim's
        # gain_coeff * sqrt(rate / (4 * spread * KGNB)) closed-form
        # normalization, computed here numerically off the actual filter).
        power = np.mean(np.abs(fading) ** 2)
        if power > 0:
            fading = fading / np.sqrt(power)
        return fading

    def _delay(self, iq: np.ndarray, delay_s: float) -> np.ndarray:
        """Apply a (possibly fractional) delay to a complex array via linear interpolation."""
        n = len(iq)
        delay_samples = delay_s * self.fs
        if delay_samples == 0:
            return iq
        src_idx = np.arange(n)
        query_idx = src_idx - delay_samples
        real = np.interp(query_idx, src_idx, iq.real, left=0.0, right=0.0)
        imag = np.interp(query_idx, src_idx, iq.imag, left=0.0, right=0.0)
        return real + 1j * imag

    def apply(self, iq: np.ndarray) -> np.ndarray:
        """
        Apply the Watterson channel to a complex baseband IQ array.

        Parameters
        ----------
        iq : np.ndarray
            Complex input signal, sampled at `self.fs`.

        Returns
        -------
        np.ndarray
            Complex output signal, same length as `iq`, transformed by
            multipath Rayleigh fading, Doppler shift, and delay spread.
        """
        iq = np.asarray(iq)
        if not np.iscomplexobj(iq):
            iq = iq.astype(complex)

        n = len(iq)
        t = np.arange(n) / self.fs
        out = np.zeros(n, dtype=complex)

        for path in self.paths:
            delayed = self._delay(iq, path.delay_s)
            fading = self._fade(n, path.doppler_spread_hz)
            nco = np.exp(1j * 2.0 * np.pi * path.doppler_shift_hz * t)
            out += path.gain * nco * fading * delayed

        return out


if __name__ == "__main__":
    # Quick self-test / demo: pass a CW tone through a CCIR-like
    # "moderate" 2-path HF channel and report the resulting envelope
    # statistics (should show Rayleigh-ish fading, RMS power ~ preserved).
    fs = 8000.0
    duration_s = 5.0
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    tone_hz = 1000.0
    iq_in = np.exp(1j * 2.0 * np.pi * tone_hz * t)

    paths = [
        WattersonPath(delay_s=0.0, doppler_spread_hz=1.0, doppler_shift_hz=0.0, gain=1 / np.sqrt(2)),
        WattersonPath(delay_s=1e-3, doppler_spread_hz=1.0, doppler_shift_hz=0.0, gain=1 / np.sqrt(2)),
    ]
    channel = WattersonChannel(paths, fs=fs, seed=1234)
    iq_out = channel.apply(iq_in)

    in_power = np.mean(np.abs(iq_in) ** 2)
    out_power = np.mean(np.abs(iq_out) ** 2)
    print(f"input power:  {in_power:.4f}")
    print(f"output power: {out_power:.4f}")
    print(f"output envelope: min={np.abs(iq_out).min():.4f}, "
          f"max={np.abs(iq_out).max():.4f}, "
          f"mean={np.abs(iq_out).mean():.4f}")