# SDDj

Local SOTA generation and animation for Aseprite via Stable Diffusion + AnimateDiff.

---

## Quick Start

```powershell
setup.ps1         <- One-click: install deps, download models, build extension
start.ps1         <- One-click: launch server + Aseprite
```

## Features

- **Generation**: Complete txt2img, img2img, inpaint, and ControlNet (OpenPose, Canny, Scribble, Lineart) pipelines.
- **Animation**: Frame chaining and AnimateDiff-Lightning integration for rapid temporal consistency.
- **Audio Reactivity**: Built-in DSP engine mapping audio features (RMS, transients, spectral bands, BPM) to diffusion parameters for Deforum-style audio-reactive animations.
- **Post-Processing**: True pixel-art pipeline (background removal, strict NEAREST downscaling, CIELAB color quantization, and dithering).
- **Offline & Private**: 100% local execution. No telemetry, no cloud dependencies.

## Architecture & Performance

SDDj uses a dual architecture: a lightweight Lua UI running inside Aseprite, communicating via WebSockets to a heavyweight FastAPI/PyTorch backend.

- **Speed**: Accelerated natively via Hyper-SD, DeepCache, and `torch.compile` (Triton). 
- **Quality**: Enhanced via FreeU v2.
- **Memory**: Optimized with Attention Slicing (SDP/FlashAttention2), VAE Tiling, and FP16 inference.

See **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** for detailed Mermaid diagrams of the system design and DSP routing.

## Documentation Suite

Everything you need to master SDDj is detailed in the `docs/` folder:

| Document | Core Focus |
|----------|------------|
| **[Guide](docs/GUIDE.md)** | Getting started, core modes, generation parameters, and post-processing pipeline. |
| **[Audio Guide](docs/AUDIO-REACTIVITY.md)** | How to use the audio-reactivity system, auto-calibration, and prompt scheduling. |
| **[Audio Reference](docs/AUDIO-REFERENCE.md)** | Exhaustive list of modulation sources/targets, math expressions, and motion presets. |
| **[API Reference](docs/API-REFERENCE.md)** | Complete JSON WebSocket protocol specification and available endpoints. |
| **[Cookbook](docs/COOKBOOK.md)** | Tested generation recipes for sprites, portraits, environments, and animations. |
| **[Configuration](docs/CONFIGURATION.md)** | Complete reference for all `SDDJ_*` environment variables. |
| **[Troubleshooting](docs/TROUBLESHOOTING.md)** | Solutions for OOM errors, compilation issues, and connection timeouts. |

## Requirements

> **Windows Only**: Relies on PowerShell 7 and Visual Studio 2022 C++ workloads which restrict operation to Windows environments.

- **GPU**: NVIDIA >= 4GB VRAM (txt2img/audio at 512x512). 8GB+ recommended for AnimateDiff/ControlNet
- **CUDA**: 12.8
- **Python**: 3.11-3.13
- **uv**: Package manager
- **Visual Studio 2022**: C++ Desktop Development workload (for torch.compile / Triton)

## License

MIT
