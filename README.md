# GPSDataAdder

Offline GPS L1 C/A IQ-file augmentation tool for testing decoder behavior with local recordings.

The tool is meant for testing the `Fraunhofer_FHR` GPS decoder against controlled offline data. It does not stream, transmit, or control RF hardware. The GUI focuses on position relocation overlays: it reads a little-endian `complex64` recording, plans a target position from Fraunhofer-derived PVT evidence, adds stronger replicas of already received satellites, and writes a new local `complex64` file. The CLI still exposes the older single-satellite augmentation path for focused DSP tests.

## What It Can Generate

- multi-satellite position overlays using target-visible received PRNs plus synthetic target-visible PRNs when needed
- target-coordinate relocation or east/north/up offset relocation from the detected baseline PVT
- continuous LNAV timing aligned to the measurement file time
- parity-valid synthetic broadcast ephemeris subframes for added target PRNs
- Doppler, Doppler-drift, code-rate, and code-rate-drift compensation for the target geometry
- GPS L1 C/A PRN code for PRN `1..32`
- deterministic 50 bps LNAV-like navigation bits
- valid LNAV word parity for TLM/HOW and payload words
- plausible TLM preamble and HOW subframe IDs cycling through `1..5`
- JSON metadata describing the overlay or synthetic signature that was added

The simple CLI payload is synthetic and deterministic. The position overlay can also synthesize deterministic, parity-valid GPS-like broadcast ephemeris subframes for extra target-visible PRNs when the requested target is not well served by the originally received satellites.

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
- the only GUI write path is `Write Position Overlay`; single synthetic-satellite controls are intentionally not shown in the GUI

## Position Overlay

The `Position Overlay` panel adds stronger synthetic received signals to a new local output file. It never removes or edits samples in the source recording. The planner uses `Fraunhofer_FHR` to decode a baseline PVT solution, received PRNs, LNAV ephemeris words, and source timing. It keeps received PRNs only when they are visible from the requested target, and fills the geometry with additional target-visible PRNs carrying deterministic GPS-like broadcast ephemeris subframes when needed. It then computes code-phase/timing for the target coordinate or east/north/up offset, regenerates continuous TLM/HOW timing aligned to the measurement file time, and writes a multi-PRN overlay. When the backend resolves to GPU, the overlay writer mixes the synthetic PRN replicas on CUDA/CuPy block by block.

The overlay keeps fractional C/A code phase internally and splits large pseudorange changes into whole-millisecond LNAV arrival shifts plus the remaining sub-millisecond PRN phase. That makes targets far outside the local code-phase ambiguity practical instead of wrapping the movement into one C/A period. Each replica also adjusts carrier Doppler, Doppler drift, code rate, and code-rate drift from target geometry. Received replicas use Fraunhofer-derived satellite position, velocity, and acceleration; added target PRNs use deterministic broadcast-orbit parameters encoded into their LNAV subframes. The integer code-phase samples shown in logs and metadata are kept as decoder-friendly summaries, but the block synthesizer uses the higher precision chip phase and timing model.

Use `Use east/north/up offset from detected PVT` when you want a relative move from the baseline solution. Disable it when you want the latitude, longitude, and altitude fields to be used directly as the target. In coordinate mode the offset fields are disabled and ignored by the planner, so stale offset values cannot silently affect the requested target. Changing target, offset, or processing settings after planning disables writing until you create a fresh plan. `Write Position Overlay` uses the visible output path and writes metadata to `<output>.relocation.json` when metadata is enabled.

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

2. Select the input capture, output path, sample rate, processing backend, and position overlay mode.
3. Use `Plan Position Overlay` to inspect the decoded baseline and requested target, then `Write Position Overlay`.
4. Open the generated output in `Fraunhofer_FHR`.
5. Use the same sample-rate assumption.
6. Run acquisition for the planned target-visible PRNs or scan `1..32`.
7. Track the overlay PRNs and decode navigation bits.

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
