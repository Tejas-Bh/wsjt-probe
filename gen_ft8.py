from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "weakmon"))

import ft8
import weakutil
import scipy.signal as signal

from decode_ft8 import message

sender = ft8.FT8Send()

# Standard FT8 message format:
# CALLSIGN CALLSIGN GRID_OR_REPORT
bits77 = sender.pack(message, 1)



# 1000 Hz is the lowest of the eight FSK tones.
# FT8 tone frequencies will be:
# 1000.0, 1006.25, ..., 1043.75 Hz
sample_rate = 12000
audio = sender.tones(bits77, 1000, sample_rate)

synthetic_ft8 = signal.hilbert(audio)

if __name__ == "__main__":
    print(f"Success! Generated complex array with shape: {synthetic_ft8.shape}")
    print(f"Data type: {synthetic_ft8.dtype} (Should be complex128 or complex64)")