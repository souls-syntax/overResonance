from scipy.io import *
from scipy import fft
import soundfile as sf
import numpy as np
try:
    data,sample_rate = sf.read('resources/sample1.wav')

    print(f"Successfully loaded file.")
    print(f"Sample Rate (Hz): {sample_rate}")
    print(f"Data type: {data.dtype}")
    print(f"Data shape: {data.shape}")

    if len(data.shape) == 1:
        print("This is mono audio file.")
        num_samples = data.shape[0]
    else:
        print(f"This is a STEREO audio file with {data.shape[1]} channels.")
        num_samples = data.shape[0]
    # data = data[:,0]
    duration_sec = num_samples/sample_rate

    print(f"Total samples: {num_samples}")
    print(f"Duration: {duration_sec:.2f} seconds")
    print(f"Data max one channel: {np.max(data[:,0])}")
    print(f"Data max one channel: {np.max(data[:,1])}")
    chunk_size = 2048
    hop_size = chunk_size//2
    num_samples = data.shape[0] 
    no_of_perfect_chunks = num_samples//chunk_size
    print(f"\nProcessing {no_of_perfect_chunks} chunks of size {chunk_size}...")
    
    window = np.hanning(chunk_size)
    processed_data = np.zeros_like(data)

    real_requencies = fft.fftfreq(chunk_size, 1/sample_rate)
    #=================================Filter Smart====================== 
    low_hz = 1500.0
    high_hz = 3000.0
    boost_db = 12.0

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

    #left_chunk = num_samples%chunk_size
    start = 0
    end = 0
    overlap_scale = np.zeros(num_samples)
    current_pos = 0
    while current_pos + chunk_size <= num_samples:
        start = current_pos
        end = start + chunk_size
        overlap_scale[start:end] += window

        current_chunk = data[start:end, :]

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
    
    remaining_chunk_unpadded = data[end:, :]
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

    processed_data /= np.max(np.abs(processed_data) + 1e-9)

    sf.write('output_file.wav',processed_data, sample_rate)
    print(f"Successfully saved to 'output_file.wav'")

except FileNotFoundError:
    print("Error")
except Exception as e:
    print(f"An error occurred: {e}")


