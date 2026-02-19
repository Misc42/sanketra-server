#!/usr/bin/env python3
"""
Real-time STT using Whisper large-v3 + Silero VAD
- VAD detects speech segments (CPU)
- Whisper transcribes only speech (GPU)
"""

import sounddevice as sd
import numpy as np
import torch
import sys
import argparse

from stt_common import (
    init_gpu, get_gpu_stats, cleanup_gpu,
    load_whisper, load_vad, select_model, custom_model_selection,
    list_input_devices, get_device_info, resample,
    to_roman, type_text, is_typing_supported, get_vad_type,
    setup_logging, log_info, log_debug, log_error, log_warning,
    preprocess_audio_frame, get_preprocessing_config
)
import os
import preflight

# Setup logging
setup_logging()

# Global model variables (initialized in main based on preference)
whisper_model = None
vad_model = None
vad_utils = None
gpu_available = False

def init_models(preference="balanced"):
    """Initialize VAD and Whisper models with preflight checks"""
    global whisper_model, vad_model, vad_utils, gpu_available

    # Run preflight checks - this prints the startup summary
    # and exits with clear error if required deps missing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pf = preflight.run(mode="cli", preference=preference, verbose=True, script_dir=script_dir)

    # Initialize GPU monitoring
    _, gpu_available = init_gpu()

    # Load VAD (with fallback to energy-based VAD)
    vad_model, vad_utils = load_vad()
    vad_type = get_vad_type()
    log_info(f"VAD: {vad_type}")

    # Load Whisper using preflight-selected config or custom
    if preference == "custom":
        model_name, compute_type, device = custom_model_selection()
    elif pf.transcription:
        model_name = pf.transcription.model
        compute_type = pf.transcription.compute_type
        device = pf.transcription.device
    else:
        model_name, compute_type, device = select_model(preference)

    vram_before, _, _ = get_gpu_stats()
    whisper_model = load_whisper(model_name, device=device, compute_type=compute_type)
    vram_after, _, _ = get_gpu_stats()

    if gpu_available and vram_after > vram_before:
        log_info(f"VRAM used: {vram_after - vram_before:.2f} GB")

    return vad_utils

# Config
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # Silero VAD requires exactly 512 samples at 16kHz (32ms)
CHUNK_MS = CHUNK_SAMPLES / SAMPLE_RATE * 1000  # ~32ms
SPEECH_PAD_MS = 300  # Padding around speech
MIN_SPEECH_MS = 250  # Minimum speech duration
MIN_SILENCE_MS = 500  # Silence duration to trigger transcription

def transcribe(audio):
    log_debug(f"Transcribing {len(audio)/SAMPLE_RATE:.2f}s of audio")
    segments, info = whisper_model.transcribe(
        audio,
        language=None,  # Auto-detect
        vad_filter=False,  # We already did VAD
        beam_size=5
    )
    text = " ".join([seg.text for seg in segments])
    log_debug(f"Detected language: {info.language}, text length: {len(text)}")
    return to_roman(text), info.language

def main():
    parser = argparse.ArgumentParser(description="Real-time STT: Whisper + Silero VAD")
    parser.add_argument("-l", "--list", action="store_true", help="List input devices")
    parser.add_argument("-d", "--device", type=int, default=None, help="Input device ID")
    parser.add_argument("-t", "--type", action="store_true", help="Type output at cursor position")
    args = parser.parse_args()

    if args.list:
        list_input_devices()
        sys.exit(0)

    # Always interactive model selection
    preference = "custom"

    # Initialize models
    utils = init_models(preference)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Check typing support
    if args.type and not is_typing_supported():
        log_warning("Typing not supported on this platform (install xdotool on Linux or pyautogui on Windows/Mac)")

    try:
        device_id, native_rate = get_device_info(args.device)
        device_name = sd.query_devices(device_id)['name']
    except Exception as e:
        log_error(f"Failed to get audio device: {e}")
        sys.exit(1)

    log_info(f"Using device: {device_name} @ {native_rate}Hz")
    print("Listening... (Ctrl+C to stop)\n", flush=True)

    # VAD state
    vad_iterator = VADIterator(vad_model, sampling_rate=SAMPLE_RATE)
    speech_buffer = []
    is_speaking = False
    silence_chunks = 0
    silence_threshold = int(MIN_SILENCE_MS / (CHUNK_SAMPLES / SAMPLE_RATE * 1000))

    native_chunk = int(native_rate * CHUNK_SAMPLES / SAMPLE_RATE)

    try:
        with sd.InputStream(samplerate=native_rate, channels=1, dtype='float32', device=device_id) as stream:
            while True:
                audio, _ = stream.read(native_chunk)
                audio = audio.flatten()

                # Resample to exactly 512 samples at 16kHz for VAD
                audio_16k = resample(audio, native_rate, SAMPLE_RATE, target_samples=CHUNK_SAMPLES)

                # POINT H: Apply noise filtering before VAD (preserves energy for VAD accuracy)
                config = get_preprocessing_config()
                if config.enabled:
                    audio_16k = preprocess_audio_frame(audio_16k, SAMPLE_RATE, config, preserve_original_energy=True)

                # Convert to torch tensor for VAD
                audio_tensor = torch.from_numpy(audio_16k)

                # Run VAD
                speech_dict = vad_iterator(audio_tensor, return_seconds=False)

                # Visual feedback (console only, not logged)
                level = np.abs(audio_16k).mean()
                status = "SPEECH" if is_speaking else "     "
                bar = "#" * int(level * 200)
                print(f"\rMIC [{bar:<20}] {status}", end="", flush=True)

                if speech_dict:
                    if 'start' in speech_dict:
                        is_speaking = True
                        silence_chunks = 0
                        log_debug("Speech started")
                    if 'end' in speech_dict:
                        is_speaking = False
                        log_debug("Speech ended")

                if is_speaking:
                    speech_buffer.append(audio_16k)
                    silence_chunks = 0
                elif len(speech_buffer) > 0:
                    silence_chunks += 1
                    speech_buffer.append(audio_16k)  # Include some silence padding

                    # Transcribe after enough silence
                    if silence_chunks >= silence_threshold:
                        full_audio = np.concatenate(speech_buffer)

                        # Only transcribe if speech is long enough
                        if len(full_audio) > int(SAMPLE_RATE * MIN_SPEECH_MS / 1000):
                            print()  # Newline before output
                            text, lang = transcribe(full_audio)
                            if text.strip() and lang in ('hi', 'en'):
                                lang_names = {'hi': 'Hindi', 'en': 'English'}
                                lang_display = lang_names.get(lang, lang)
                                output = f"[{lang_display}] {text.strip()}"
                                print(output, flush=True)
                                log_info(f"Transcribed: {output}")

                                if args.type:
                                    type_text(text.strip() + ' ')
                                    log_debug("Text typed at cursor")
                            elif text.strip():
                                print(f"[{lang}] (skipped)", flush=True)
                                log_debug(f"Skipped language: {lang}")

                        speech_buffer = []
                        silence_chunks = 0
                        vad_iterator.reset_states()

    except sd.PortAudioError as e:
        log_error(f"Audio error: {e}")
        print(f"\nAudio error: {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup_gpu()
        log_info("Stopped by user")
        print("\n\nBand.")
