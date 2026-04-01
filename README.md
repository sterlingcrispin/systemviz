# pcdiag

Small terminal dashboard for Ubuntu that keeps the main thermal and usage signals on one screen:

- GPU utilization
- GPU memory used percent
- GPU temperature
- GPU fan speed and power draw/limit when the NVIDIA driver exposes them
- CPU utilization
- CPU temperatures, with `Tdie` and `Tccd1` surfaced when available
- RAM used percent
- Top processes by current CPU, RAM, GPU, and VRAM impact

## Run

```bash
cd path/to/pc_diagnostics
./pcdiag.py
```

## Install As `systemviz`

User-local install:

```bash
cd path/to/pc_diagnostics
./install_systemviz.sh
systemviz
```

Optional global install:

```bash
cd path/to/pc_diagnostics
sudo ./install_systemviz.sh --global
systemviz
```

If you prefer the installed command to keep tracking the repo file instead of copying it:

```bash
./install_systemviz.sh --symlink
```

Useful flags:

```bash
./pcdiag.py --interval 0.5
./pcdiag.py --history 120
./pcdiag.py --top 15
./pcdiag.py --snapshot
```

Controls in the live view:

- `q` quits
- `r` clears the history graphs
- `left` and `right` arrows change the process-table sort column

## Temperature references

- CPU temperature bands are tuned for Threadripper-style `Tdie`/`Tccd` readings: `ok <60C`, `watch 60-63C`, `near 64-67C`, `max 68C+`
- GPU temperature bands are derived from the NVIDIA driver-reported temperature limit on the current card, then shown as `ok`, `watch`, `near`, and `driver limit`

## Data sources

- CPU and RAM come from `psutil`
- CPU temperatures come from Linux hwmon sensors via `psutil`
- GPU stats come from `nvidia-smi`

## Notes

- Per-process GPU percent comes from `nvidia-smi pmon`
- Per-process VRAM is only available for processes `nvidia-smi` reports through the compute-apps query, so some desktop graphics processes may show GPU percent without a VRAM number
- The main CPU temperature line intentionally removes the sensor label from the value column so the bars stay aligned; the detailed sensor names are still shown to the right
