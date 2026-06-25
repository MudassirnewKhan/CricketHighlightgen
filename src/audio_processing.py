import os
import librosa
import numpy as np
import moviepy.editor as mp
import logging

def get_exact_audio_peaks(video_path, sensitivity=1.3):
    """
    Extracts, normalizes, and identifies dynamic outlier loudness peaks 
    (crowd roars) using an adaptive threshold calculation.
    """
    logging.info("🔊 Extracting and normalizing audio stream...")
    video = mp.VideoFileClip(video_path)
    temp_audio = "temp_processing.wav"

    try:
        # Write temporary audio track
        video.audio.write_audiofile(temp_audio, fps=22050, verbose=False, logger=None)
        
        # Load audio track
        y, sr = librosa.load(temp_audio, sr=22050)

        # Normalize audio for uniform baseline treatment
        y = librosa.util.normalize(y)

        # Calculate Root Mean Square (RMS) energy per second
        rms = librosa.feature.rms(y=y, frame_length=sr, hop_length=sr)[0]

        # Adaptive Threshold Engine: Mean + (Sensitivity * StdDev)
        mean_vol = np.mean(rms)
        std_vol = np.std(rms)
        dynamic_threshold = mean_vol + (sensitivity * std_vol)

        exact_peaks = np.where(rms >= dynamic_threshold)[0]
        logging.info(f"📊 Stats -> Max: {np.max(rms):.3f} | Mean: {mean_vol:.3f} | Threshold: {dynamic_threshold:.3f}")

        return exact_peaks.tolist()
        
    finally:
        video.close()
        if os.path.exists(temp_audio):
            os.remove(temp_audio)