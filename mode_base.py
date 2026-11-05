# mode_base.py
from abc import ABC, abstractmethod
import numpy as np

class BaseModeFrontend(ABC):
    """Abstract Frontend interface for weakmon modes (FT8, WSPR, JT65)."""
    
    @property
    @abstractmethod
    def mode_name(self) -> str:
        """e.g. 'FT8', 'WSPR', 'JT65'"""
        pass

    @property
    @abstractmethod
    def symbol_duration(self) -> float:
        """Symbol length T in seconds (FT8: 0.16s, WSPR: 0.683s)."""
        pass

    @property
    @abstractmethod
    def slot_duration(self) -> float:
        """Slot window in seconds (FT8: 15s, WSPR: 120s)."""
        pass

    @abstractmethod
    def synthesize_reference_iq(self, msg: dict, sample_rate: int) -> tuple[np.ndarray, float]:
        """
        Synthesizes the ideal baseband IQ reference for a decoded message.
        Returns: (ref_iq, center_hz)
        """
        pass