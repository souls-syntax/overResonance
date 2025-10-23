# overResonance 🔊

**`tacetBreaker.py`**: A Python command-line tool to denoise audio and boost high frequencies for enhanced clarity, inspired by the need to make sound more accessible.

## Description

This script processes WAV audio files by first applying noise reduction using spectral gating (`noisereduce`) and then boosting higher frequencies using a Short-Time Fourier Transform (STFT) approach with Overlap-Add (OLA) processing. The goal is to reduce background noise and enhance the clarity of speech or other desired sounds, particularly for listeners who may have difficulty hearing high-frequency components.

## Features ✨

* **Noise Reduction:** Uses the `noisereduce` library based on spectral gating to suppress stationary background noise.
* **Frequency Boosting:** Implements an STFT pipeline with Hanning windowing and OLA to selectively boost a range of frequencies using a smooth gain filter.
* **Command-Line Interface:** Configurable via CLI arguments for input/output files, boost amount (dB), frequency range, and noise reduction aggressiveness.
* **Stereo Processing:** Handles stereo audio files, processing each channel independently.

## Requirements 📦

* Python 3.x
* NumPy
* SciPy
* SoundFile (or Librosa as fallback)
* Librosa
* noisereduce

## Installation ⚙️

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/overResonance.git](https://github.com/your-username/overResonance.git)
    cd overResonance
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv-overResonance
    source .venv-overResonance/bin/activate  # On Linux/macOS
    # .\ .venv-overResonance\Scripts\activate  # On Windows
    ```
3.  **Install the required libraries:**
    ```bash
    pip install numpy scipy soundfile librosa noisereduce argparse
    ```
    *(Or `pip install -r requirements.txt` if you create one)*

## Usage 🚀

Run the script from your terminal:

```bash
python tacetBreaker.py <input_file> [options]
```

### Arguments

* `input_file`: Path to the input WAV file (required).

#### Options
* `-o, --output_file`: Path to save the processed WAV file (default: output_file.wav).

* `-b, --boost_db`: Boost amount in decibels (dB) for high frequencies (default: 12.0).

* ``--low_hz``: Frequency (Hz) where the boost starts ramping up (default: 1500.0).

* ``--high_hz``: Frequency (Hz) where the boost reaches its maximum (default: 3000.0).

* ``-p, --prop_decrease``: Aggressiveness of noise reduction (0.0 to 1.0, default: 0.85). Lower values are less aggressive.

* ``-h, --help``: Show the help message.

### Examples:
* **Basic usage(defaults)**
`python tacetBreaker.py resources/my_audio.wav`

* **Specify output and boost:**
`python tacetBreaker.py resources/noisy_speech.wav -o cleaned_speech.wav -b 15`

* **Adjust frequency range and noise reduction:**
`python tacetBreaker.py resources/podcast.wav -o podcast_boosted.wav --low_hz 2000 --high_hz 5000 -p 0.9`

## How It Works (Briefly)

1. Load Audio: Reads the input WAV file using soundfile (with librosa as a fallback).

2. Denoise: Applies spectral gating noise reduction via noisereduce to the entire audio signal, using the first second as a noise profile.

3. STFT Processing (OLA):

   * Iterates through the denoised audio in overlapping chunks (chunk_size=2048, hop_size=1024).

   * Applies a Hanning window to each chunk.

   * Calculates the Fast Fourier Transform (FFT).

   * Applies the custom gain_filter (based on --boost_db, --low_hz, --high_hz) to the frequency spectrum.

   * Calculates the Inverse FFT (IFFT) to get the processed chunk.

   * Adds the processed chunk back to the output buffer (Overlap-Add).

4. Normalization: Peak-normalizes the final audio signal to prevent clipping.

5. Save Audio: Writes the processed audio to the specified output WAV file.

## Limitations:
* Mono File Handling: The current version is primarily tested on stereo files and may crash or produce incorrect results when processing mono audio files due to hardcoded stereo indexing in some parts. Fixing this is left as a future exercise (or ignored, because who uses mono in 2025?)
* Noise Reduction Type: noisereduce is most effective on stationary background noise (hums, fans, hiss). It may be less effective on non-stationary noise (clicks, bangs, other voices).


## Disclaimer

This project was developed through a significant amount of trial and error, particularly when attempting to integrate certain other noise reduction libraries. I despise RNNoise wrappers.

