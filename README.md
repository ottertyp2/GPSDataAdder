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
- detect-first planning for PRN, Doppler, code phase, amplitude, TOW, and compute settings
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
- samples are processed through memory-mapped blocks so 10-minute files do not need full RAM copies
- CPU processing uses multiple worker threads by default
- optional `GPU` backend uses CuPy/CUDA when installed; `Auto` falls back to CPU
- progress and cancel are available during long 10-minute recordings
- output is written to a temporary `.partial` file and moved into place only after success
- `Auto amplitude` estimates the input RMS from several windows and chooses a GPS-like level

## Detect First

Use `Detect` before `Start` when you want to see the exact plan first.

The detect mode does not write an output file. It inspects the input recording, picks a deterministic synthetic PRN/code-phase/Doppler plan from the file fingerprint, estimates a realistic amplitude, tries to determine the measurement TOW from LNAV HOW, resolves the compute backend, and writes the proposed values into the GUI fields.

For long recordings, Detect uses the sibling `Fraunhofer_FHR` project first when it is available at `..\Fraunhofer_FHR` or through `FRAUNHOFER_FHR_PATH`. The fast TOW path reuses its acquisition, tracking, and navigation-decoding stages, but avoids the full PVT solve: it scans a short window around 60 seconds, tracks only the strongest candidates for about 18 seconds, and stops as soon as a valid HOW/TOW subframe is decoded. `Auto` uses the CuPy/CUDA path when available. On the local `testv4_28_04_26_10min.bin` recording this returns a plan in about 12 seconds instead of timing out in the full PVT pipeline.

When a measurement TOW is found, `Start TOW count`, `Start subframe ID`, and `Nav seed` are set so the synthetic LNAV HOW and deterministic payload pattern are tied to the source recording. If TOW is not recoverable from the capture, Detect says so and still chooses a deterministic file-specific seed from the input fingerprint.

Modes:

- `Weak`: lower target level, useful for sensitivity testing
- `Balanced`: default realistic level
- `Strong`: easier decoder sanity check while still tied to the input level

After detect, `Auto amplitude` is disabled and the detected fixed amplitude is shown, so `Start` uses the visible value.

Set `GPSDATAADDER_FULL_PVT_FALLBACK=1` only when you deliberately want the slower full Fraunhofer PVT fallback after the fast TOW path fails.

## Automatic Amplitude

The GUI enables `Auto amplitude` by default. It probes a few windows across the input file, computes a clipped RMS so isolated spikes do not dominate, and sets the synthetic satellite level from the target `C/N0`.

The default target is `42 dB-Hz`. At the default `6.061 MSa/s` sample rate, that places the synthetic signal roughly `25.8 dB` below the measured IQ RMS in power terms, which keeps it GPS-like instead of visually obvious in raw samples.

Use manual amplitude only when you want a deliberately strong or weak stress case.

## Position Overlay

The `Position Overlay` panel adds stronger synthetic replicas of already received satellites to a new local output file. It never removes or edits samples in the source recording. The planner uses `Fraunhofer_FHR` to decode a baseline PVT solution, the received PRNs, and their LNAV ephemeris words. It then computes code-phase shifts for a selected target coordinate or a small east/north/up offset, regenerates continuous TLM/HOW timing aligned to the measurement file time, reuses the decoded ephemeris payload words, and writes a multi-PRN overlay. When the backend resolves to GPU, the overlay writer mixes the synthetic PRN replicas on CUDA/CuPy block by block.

The overlay keeps fractional C/A code phase internally and now splits large pseudorange changes into whole-millisecond LNAV arrival shifts plus the remaining sub-millisecond PRN phase. That makes targets far outside the local code-phase ambiguity practical instead of wrapping the movement into one C/A period. Each replica also adjusts carrier Doppler, Doppler drift, code rate, and code-rate drift from the target-vs-baseline geometry reported through Fraunhofer-derived satellite position, velocity, and acceleration. The integer code-phase samples shown in logs and metadata are kept as decoder-friendly summaries, but the block synthesizer uses the higher precision chip phase and timing model.

Use `Use offset from detected PVT` for first tests. A few kilometres is the intended initial range because the overlay shifts pseudoranges by code phase while keeping the decoded source ephemerides. `Write Position Overlay` uses the visible output path and writes metadata to `<output>.relocation.json` when metadata is enabled.

## CLI Usage

```powershell
python -m app.main --cli input.bin output_with_synthetic_prn.bin --sample-rate 6060606.0606 --prn 22 --doppler-hz 1800 --code-phase-samples 350 --auto-amplitude
```

Detect-only CLI:

```powershell
python -m app.main --cli input.bin --detect-only --detect-mode balanced --sample-rate 6060606.0606
```

Useful defaults match the local decoder workflow:

- input/output format: little-endian `complex64`
- default sample rate: `200e6 / 33 = 6060606.0606 Sa/s`
- default PRN: `22`
- default Doppler: `1500 Hz`
- default code phase: `350 samples`
- default auto amplitude target: `42 dB-Hz`
- default manual amplitude: `0.05`
- default compute backend: `auto`
- default chunk size: `4,000,000 samples`
- default CPU workers: automatic, up to logical cores minus one

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

## Doxygen Documentation

Source documentation is configured for Doxygen:

```powershell
doxygen Doxyfile
```

The generated HTML goes to `docs/doxygen-output/html` and is ignored by Git. The committed documentation sources are `Doxyfile`, `docs/mainpage.dox`, and the Python docstrings.

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
- `app/dsp/relocation_overlay.py`: multi-satellite position relocation overlay
- `app/dsp/synthetic_satellite.py`: block signal generation and file augmentation
- `app/tests/`: focused tests

## Input Format

The current workflow assumes:

- sample type: little-endian `complex64`
- layout: `float32 I` followed by `float32 Q`
- signal family: GPS L1 C/A baseband or low-IF IQ

The generated carrier is placed at the configured `--doppler-hz` frequency relative to the IQ recording. For a low-IF recording, set `--doppler-hz` to the desired residual carrier/search-center value for the decoder experiment.
