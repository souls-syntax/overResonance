from scipy.io import *
from scipy import fft
import soundfile as sf
import numpy as np
import librosa
import noisereduce as nr 
import argparse

def main():
    #================================= ARGUMENT PARSING ======================
    parser = argparse.ArgumentParser(description='Denoise and boost frequencies in an audio file.')
    
    parser.add_argument('input_file', help='Path to the input WAV file.')
    parser.add_argument('-o', '--output_file', default='output_file.wav',
                        help='Path to save the processed WAV file (default: output_file.wav)')
    parser.add_argument('-b', '--boost_db', type=float, default=12.0,
                        help='Boost amount in decibels (dB) for high frequencies (default: 12.0)')
    parser.add_argument('--low_hz', type=float, default=1500.0,
                        help='Frequency (Hz) where the boost starts ramping up (default: 1500.0)')
    parser.add_argument('--high_hz', type=float, default=3000.0,
                        help='Frequency (Hz) where the boost reaches its maximum (default: 3000.0)')
    parser.add_argument('-p', '--prop_decrease', type=float, default=0.85,
                        help='Aggressiveness of noise reduction (0.0 to 1.0, default: 0.85)')
    
    args = parser.parse_args()
    #================================= END ARG PARSING ======================
    try:
        try:
            data, sample_rate = sf.read(args.input_file, dtype='float32')
        except Exception as e:
            print(f"Soundfile failed, trying librosa... Error: {e}")
            data, sample_rate = librosa.load(args.input_file, mono=False, sr=None)
            if data.ndim > 1:
                data = data.T
        
        

        print(f"Successfully loaded file.")
        print(f"Sample Rate (Hz): {sample_rate}")
        print(f"Data type: {data.dtype}")
        print(f"Data shape: {data.shape}")

        if len(data.shape) == 1:
            print("This is mono audio file.")
            num_samples = data.shape[0]
            is_stereo = False
        else:
            is_stereo = True
            print(f"This is a STEREO audio file with {data.shape[1]} channels.")
            num_samples = data.shape[0]
        # data = data[:,0]
        duration_sec = num_samples/sample_rate

        print(f"Total samples: {num_samples}")
        print(f"Duration: {duration_sec:.2f} seconds")
        if is_stereo:
            print(f"Data max one channel: {np.max(data[:,0])}")
            print(f"Data max one channel: {np.max(data[:,1])}")
        else:
            print(f"Data max mono: {np.max(data)}")
        chunk_size = 2048
        hop_size = chunk_size//2
        num_samples = data.shape[0] 
        no_of_perfect_chunks = num_samples//chunk_size
        print(f"\nProcessing {no_of_perfect_chunks} chunks of size {chunk_size}...")
        
        window = np.hanning(chunk_size)
        processed_data = np.zeros_like(data)

        real_requencies = fft.fftfreq(chunk_size, 1/sample_rate)
        #=================================Filter Smart======================

        low_hz = args.low_hz
        high_hz = args.high_hz
        boost_db = args.boost_db

        boost_factor = 10.0**(boost_db / 20.0) # Math formula for db

        gain_filter = np.ones(chunk_size)

        low_indices = np.where(np.abs(real_requencies) >= low_hz)[0]

        high_indices = np.where(np.abs(real_requencies) >= high_hz)[0]

        gain_filter[high_indices] = boost_factor

        # Applying the "Gradual slope" part

        slope_indices = np.where((np.abs(real_requencies)>=low_hz) & (np.abs(real_requencies) < high_hz))[0]

        if len(slope_indices) > 0:
            slope_freq = np.abs(real_requencies[slope_indices])
            ramp = np.interp(slope_freq, [low_hz, high_hz], [1.0, boost_factor])
            gain_filter[slope_indices] = ramp
        print(f"Built a gain filter ramping {low_hz}Hz to {high_hz}Hz")

        #=================================Filter======================

        # indices_to_boost = np.where(np.abs(real_requencies) > target_hz)
        #=================================Noisereducing native=============

        print("Denoising audio... (this may take a moment)")
        prop_decrease = args.prop_decrease
        noise_clip_mono = data[:min(sample_rate,len(data)), 0]
        denoised_left = nr.reduce_noise(y=data[:,0], sr=sample_rate, y_noise=noise_clip_mono, prop_decrease=prop_decrease, stationary=True)
        if is_stereo:
            noise_clip_mono_right = data[0:sample_rate, 1]
            denoised_right = nr.reduce_noise(y=data[:,1], sr=sample_rate, y_noise=noise_clip_mono_right, prop_decrease=prop_decrease, stationary=True)
        
        denoised_data = np.zeros_like(data)
        
        if is_stereo:
            denoised_data[:, 0] = denoised_left
            denoised_data[:, 1] = denoised_right
        else:
            denoised_data = denoised_left
        print("Denoising complete.")
        #=================================NR==============================
        #left_chunk = num_samples%chunk_size
        start = 0
        end = 0
        overlap_scale = np.zeros(num_samples)
        current_pos = 0
        while current_pos + chunk_size <= num_samples:
            start = current_pos
            end = start + chunk_size
            overlap_scale[start:end] += window

            current_chunk = denoised_data[start:end, :]

            # for ch in range(2):

            chunk_left = current_chunk[:,0] * window

            frequencies_left = fft.fft(chunk_left)

            frequencies_left = frequencies_left * gain_filter

            new_chunk_left = fft.ifft(frequencies_left)

            processed_data[start:end, 0] += new_chunk_left.real # left channel
            
            # Process Right channel
            chunk_right = current_chunk[:,1] * window

            frequencies_right = fft.fft(chunk_right)

            frequencies_right = frequencies_right * gain_filter

            new_chunk_right = fft.ifft(frequencies_right) 

            processed_data[start:end, 1] += new_chunk_right.real # right channel    
            current_pos += hop_size

            if current_pos % (hop_size * 100)  == 0:
                print(f"Processed chunk {current_pos}")

        print("Finished processing all the chunks.")
        
        remaining_chunk_unpadded = denoised_data[end:, :]
        num_remaining = remaining_chunk_unpadded.shape[0]
        if num_remaining > 0 :

            print(f"Processing remaining {num_remaining} samples...")

            remaining_chunk = np.zeros((chunk_size,2))

            remaining_chunk[:num_remaining, :] = remaining_chunk_unpadded

        # Handling left channel
            overlap_scale[end:] += window[:num_remaining]

            chunk_left = remaining_chunk[:,0] * window
            frequencies_left = fft.fft(chunk_left)
            frequencies_left = frequencies_left * gain_filter

            new_chunk_left = fft.ifft(frequencies_left)

        # Handling right channel leftover
            chunk_right = remaining_chunk[:,1] * window

            frequencies_right = fft.fft(chunk_right)
            frequencies_right = frequencies_right * gain_filter

            new_chunk_right = fft.ifft(frequencies_right)

            processed_data[end:,0] += new_chunk_left.real[:num_remaining]
            processed_data[end:,1] += new_chunk_right.real[:num_remaining]

            print(f"Successfully parsed the remaining chunk")
        else:
            print("No remaining chunk to process.")

        #processed_data /= (overlap_scale[:, None] + 1e-9)
        
        #processed_data = np.clip(processed_data, -1.0, 1.0)
        processed_data /= (np.max(np.abs(processed_data)) + 1e-9)

        sf.write(args.output_file, processed_data, sample_rate)
        print(f"Successfully saved processed audio to '{args.output_file}'")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

