# Nepali_Audiobook_Create

Turn **Nepali (Devanagari) text into an audiobook MP3 and an MP4 video with the
text on screen, fully offline** on your own PC. Numbers are spoken in Nepali
words (`110 → एक सय दस`), long text is synthesized in short chunks so quality
stays stable, and runs are pause/resume.

Also includes an optional English EPUB → Nepali EPUB translation stage
(needs [Ollama](https://ollama.com), not required for TTS/video).

## Example

Input text ([`examples/sample_ne.txt`](examples/sample_ne.txt)):

> एडेनको बगीचा
>
> सोफी आमुन्डसेन विद्यालयबाट आफ्नो घरतिर जाँदै थिइन्। उनी एक सय दस वर्षको
> थिएनन्, तर उनलाई जीवनबारेमा सोच्न धेरै मन लाग्थ्यो। के सबैभन्दा ठूलो
> प्रश्न के हो?

Produced speech ([`examples/sample_ne.mp3`](examples/sample_ne.mp3)):

<audio controls src="https://github.com/Bishalai/Nepali_Audiobook_Create/raw/main/examples/sample_ne.mp3"></audio>

Produced video ([`examples/sample_ne.mp4`](examples/sample_ne.mp4)) — the text
appears exactly while its audio plays (timings come from the same manifest used
to assemble the speech, so sync is exact):

<video controls width="640" src="https://github.com/Bishalai/Nepali_Audiobook_Create/raw/main/examples/sample_ne.mp4"></video>

## Quick start

```powershell
# 1. install dependencies (Python 3.10+, ffmpeg on PATH)
pip install -r requirements.txt

# 2. download the Nepali VITS model (~437 MB, straight into models/vits/,
#    no HuggingFace caches) and wire up the inference code
python scripts/download_vits.py

# 3. make speech from your own Devanagari .txt
python -m tts text examples/sample_ne.txt --out output
#    -> output/audio/sample_ne.mp3

# 4. make the video (auto-synthesizes the speech first if missing/edited)
python scripts/make_video.py examples/sample_ne.txt --out output
#    -> output/video/sample_ne.mp4
```

The model is **not** stored in this repository — step 2 downloads it once,
project-local (never on `C:`).

## Features

- **TXT → speech** (`python -m tts text file.txt`): blank-line paragraphs,
  one-sentence-per-line files, or flowing text — all detected automatically.
  Editing the `.txt` later re-synthesizes only the changed parts.
- **TXT → video** (`python scripts/make_video.py file.txt`): Devanagari text
  burned onto the video with ffmpeg/libass (HarfBuzz shaping → correct
  conjuncts/matras), options `--sentence`, `--font`, `--fontsize`,
  `--resolution 1920x1080`, `--bg`, `--fade-ms`.
- **EPUB → speech** (`python -m tts run book.epub --chapters 4`): chapter-wise
  audiobook from a translated EPUB, plus a whole-book merge.
- **Numbers spelled out** in Nepali before synthesis (`1,23,456 → एक लाख
  तेइस हजार चार सय छपन्न`).
- **Pause / resume**: `python -m tts pause --dir output`, `Ctrl+C`, or
  `--pause-after-min N` — re-run the same command to continue; finished chunks
  are never re-synthesized.
- Everything local: models live in `models/` inside the project, GPU (CUDA)
  or `--device cpu`.

Full command reference: [`USAGE.txt`](USAGE.txt).

## Translation (optional)

Needs Ollama installed with `qwen3.5:9b-q4_K_M` (~6.6 GB, keep it
project-local with `setx OLLAMA_MODELS "<project>\models"` before the first
pull):

```powershell
python -m translator list books/mybook.epub
python -m translator run books/mybook.epub --chapters 4 --out output
python -m translator run books/mybook.epub --all --out output
```

Per-chapter checkpoints make it resumable; `--glossary glossary.txt`
(`Sophie -> सोफी`) keeps names consistent.

## Project layout

```
tts/                 speech pipeline (chunking, numbers, pause/resume, manifests)
translator/          EPUB parser + optional Ollama translation stage
scripts/
  download_vits.py   one-time: download the model + clone/wire VITS code
  setup_vits.py      re-wire config/symbols/cleaners
  make_video.py      TXT -> MP4 (text on screen + TTS audio)
examples/            one example: .txt + generated .mp3 + generated .mp4
models/              model weights (git-ignored, download after cloning)
books/, output/      your inputs and generated results (git-ignored)
```
