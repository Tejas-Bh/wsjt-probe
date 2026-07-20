# wspr-probe: introduction
*Note: the other markdown files in this repository contain more detailed descriptions of how I plan to go about this.*

Opportunistic remote sensing is the process of collecting data without contact using __existing infrastructure__.

From my research, the use of __most__ ground systems for this purpose has been limited to:
1. Using metrics that come with the transmitted channel state information of the signal or can be calculated after the fact (e.g., SNR, RSL, link availability)
2. Proprietary waveforms that can't be analyzed __easily__ and precisely any further than broad statistics

These limiting factors have been the subject of research in __ionospheric__ propagation and characterization.

Particularly, HamSCI has been able to deploy receivers that use well-known time signals and reconstructable WSPR measurements as methods of ionospheric remote sensing from the ground.

The use of open-source amateur modes and time signals is justified by the ability of detailed waveform analysis given (e.g., phase, frequency/doppler shift).

__However__, I can't seem to find anything similar for citizen science-based technologies that allow for both *ionospheric* and *tropospheric* sensing.

---

The closest thing I've found from my research is __commercial microwave links__, which use information from microwave cell transmission and reception.

However, since these modes are proprietary, they don't allow for detailed waveform-level analysis.

__This is where I believe amateur radio modes could be used__.
