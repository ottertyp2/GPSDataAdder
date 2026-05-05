# GPSDataAdder

Offline GUI tool for adding one plausible synthetic GPS L1 C/A satellite channel to a local IQ recording.

The tool is meant for testing the `Fraunhofer_FHR` GPS decoder against controlled offline data. It does not stream, transmit, or control RF hardware. It reads a little-endian `complex64` file, generates one synthetic satellite signal, adds it sample-by-sample in chunks, and writes a new `complex64` file.

## What It Generates

- GPS L1 C/A PRN code for PRN `1..32`
- deterministic 50 bps LNAV-like navigation bits
- valid LNAV word parity for TLM/HOW and payload words
- plausible TLM preamble and HOW subframe IDs cycling through `1..5`
- configurable Doppler, code phase, carrier phase, and amplitude
- automatic amplitude estimation from the input IQ level using a target `C/N0`
- JSON metadata describing the exact synthetic signature that was added

The payload is synthetic and deterministic. It is useful for acquisition, tracking, bit sync, preamble detection, and parity checks, but it is not real broadcast ephemeris.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## GUI Usage

```powershell
python -m app.main
```

The GUI is the default entry point. It is built for large local recordings:

- processing runs in a worker thread so the window stays responsive
- samples are streamed in chunks instead of loading the full file into RAM
- progress and cancel are available during long 10-minute recordings
- output is written to a temporary `.partial` file and moved into place only after success
- `Auto amplitude` estimates the input RMS from several windows and chooses a GPS-like level

## Automatic Amplitude

The GUI enables `Auto amplitude` by default. It probes a few windows across the input file, computes a clipped RMS so isolated spikes do not dominate, and sets the synthetic satellite level from the target `C/N0`.

The default target is `42 dB-Hz`. At the default `6.061 MSa/s` sample rate, that places the synthetic signal roughly `25.8 dB` below the measured IQ RMS in power terms, which keeps it GPS-like instead of visually obvious in raw samples.

Use manual amplitude only when you want a deliberately strong or weak stress case.

## CLI Usage

```powershell
python -m app.main --cli input.bin output_with_synthetic_prn.bin --sample-rate 6060606.0606 --prn 22 --doppler-hz 1800 --code-phase-samples 350 --auto-amplitude
```

Useful defaults match the local decoder workflow:

- input/output format: little-endian `complex64`
- default sample rate: `200e6 / 33 = 6060606.0606 Sa/s`
- default PRN: `22`
- default Doppler: `1500 Hz`
- default code phase: `350 samples`
- default auto amplitude target: `42 dB-Hz`
- default manual amplitude: `0.05`
- default chunk size: `1,000,000 samples`

If no output path is supplied, the tool writes next to the input file using a name like:

```text
input.with_prn22.bin
```

A sidecar metadata file is written by default:

```text
output_with_synthetic_prn.bin.synthetic.json
```

Disable it with:

```powershell
python -m app.main input.bin output.bin --no-metadata
```

## Example Test Workflow With Fraunhofer_FHR

1. Generate an augmented recording:

   ```powershell
   python -m app.main
   ```

2. Select the input capture, output path, sample rate, PRN, Doppler, code phase, and amplitude.
3. Start the run and wait for the output file.
4. Open the generated output in `Fraunhofer_FHR`.
5. Use the same sample-rate assumption.
6. Run acquisition for the configured PRN or scan `1..32`.
7. Track that PRN and decode navigation bits.

## Running Tests

```powershell
pytest app/tests
```

## Development Workflow

This repository expects shareable changes to be committed and pushed to both GitHub and GitLab through `origin`.

```powershell
powershell -ExecutionPolicy Bypass -File tools/check_git_sync.ps1
```

## Project Layout

- `app/main.py`: GUI entry point and optional CLI mode
- `app/gui/`: PySide6 window and worker thread
- `app/dsp/gps_ca.py`: GPS C/A PRN generation
- `app/dsp/lnav.py`: synthetic LNAV-like bit stream and parity
- `app/dsp/synthetic_satellite.py`: block signal generation and file augmentation
- `app/tests/`: focused tests

## Input Format

The current workflow assumes:

- sample type: little-endian `complex64`
- layout: `float32 I` followed by `float32 Q`
- signal family: GPS L1 C/A baseband or low-IF IQ

The generated carrier is placed at the configured `--doppler-hz` frequency relative to the IQ recording. For a low-IF recording, set `--doppler-hz` to the desired residual carrier/search-center value for the decoder experiment.
