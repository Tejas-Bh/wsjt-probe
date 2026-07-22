# Multi-Receiver Networks & Passive Tomography

One of the most compelling aspects of applying waveform reconstruction to an open, crowdfunded radio network is the potential for multi-static spatial analysis.

## Exploiting Multi-Receiver Configurations
Because WSPR transmitters broadcast omnidirectionally and operate on globally coordinated schedules, a single transmission is routinely captured simultaneously by dozens of independent software-defined radio (SDR) receivers scattered across a continent.

Under this framework, each individual receiver $i$ independently estimates its own complex propagation channel function $h_i(t)$ for the exact same physical transmission event:

```
                +-------------------+
                | WSPR Transmitter  |
                +---------+---------+
                          |
            +-------------+-------------+
            |             |             |
            v             v             v
      Path 1 (h_1)  Path 2 (h_2)  Path 3 (h_3)
            |             |             |
            v             v             v
       +--------+    +--------+    +--------+
       | Rcvr 1 |    | Rcvr 2 |    | Rcvr 3 |
       +--------+    +--------+    +--------+
```

## Disentangling Signal Dynamics
By cross-correlating channel estimates and their corresponding residuals across multiple geographically separated paths, researchers can mathematically decouple mixed signal behaviors:

1. **Transmitter-Local Effects:** Phase noise, temperature drift, or power drops that appear identically across *all* reporting receivers.
2. **Receiver-Local Effects:** Local electromagnetic interference (EMI), clock jitter, or line-noise spikes unique to a single receiving node.
3. **Shared Propagation Effects:** Structural variations matching correlated geographic zones, mapping back to physical regions in the ionospheric medium.

## Move Toward Passive Radio Tomography
When spatially distributed residuals exhibit strong cross-correlation or time-delayed similarities, it implies an ionospheric structure (such as a Traveling Ionospheric Disturbance or a localized plasma bubble) is moving across the intersected propagation paths. 

Aggregating these simultaneous waveform measurements transitions standard amateur radio reporting into a high-fidelity **passive radio tomography grid**, allowing researchers to track the velocity, scale, and density variations of ionospheric phenomena over wide regions.
