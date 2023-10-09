from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Activation
import numpy as np
import wave
import os
#-----------------------------------------------------------------------------------------
# Function to normalize audio data
def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

# Read multiple audio files from a directory
import glob

def read_audio_from_directory(directory_path):
    global sample_size, framerate
    audio_paths = glob.glob(f"{directory_path}/*.wav")
    x_train_list = []
    y_train_list = []
    for audio_path in audio_paths:
        with wave.open(audio_path, 'r') as wav_file:
            n_channels, sampwidth, framerate, n_frames = wav_file.getparams()[:4]
            audio_data = wav_file.readframes(n_frames)
        samples = np.frombuffer(audio_data, dtype=np.int16)

    # Normalize the audio data
        samples = normalize_audio(samples)

    # Prepare your training data
        sample_size = min(4096, len(samples) - 1)
        x_train = samples[:sample_size].reshape(1, sample_size, 1)
        y_train = samples[1:sample_size + 1].reshape(1, sample_size, 1)
    
        x_train_list.append(x_train)
        y_train_list.append(y_train)
    return x_train_list, y_train_list

# Change the audio_paths to read from a directory
directory_path = "1yeezus"
x_train_list, y_train_list = read_audio_from_directory(directory_path)


# Convert lists to numpy arrays for training
x_train_multi = np.vstack(x_train_list)
y_train_multi = np.vstack(y_train_list)

# Initialize a simple Sequential model
model = Sequential()

# Add a Conv1D layer to simulate a WaveNet layer
model.add(Conv1D(filters=40, kernel_size=2, dilation_rate=2, padding='causal', input_shape=(sample_size, 1)))
# Add an activation layer
model.add(Activation('relu'))
model.add(Conv1D(filters=40, kernel_size=2, dilation_rate=4, padding='causal'))
model.add(Activation('relu'))
model.add(Conv1D(filters=40, kernel_size=2, dilation_rate=8, padding='causal'))
model.add(Activation('relu'))

# Compile the model
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.008)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model with multiple audio files
log_dir = "logs/fit/"

from keras.callbacks import TensorBoard
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train_multi, y_train_multi, epochs=20, batch_size=2,validation_split=0.2,callbacks=[tensorboard_callback])

# Your code for generating and saving audio goes here

# Function to generate audio
def generate_audio(model, input_audio, num_samples=16000):
    generated_audio = np.zeros(num_samples)
    for i in range(num_samples):
        prediction = model.predict(input_audio)
        generated_audio[i] = prediction[0][0][0]
        input_audio = np.roll(input_audio, -1)
        input_audio[-1] = generated_audio[i]
    return generated_audio

# After training the model
# Generate audio using the model
input_audio = np.random.randn(1, 4096, 1)
'''generated_audio = generate_audio(model, input_audio, num_samples=80000)


# Save the generated audio
with wave.open('generated_audio.wav', 'w') as wav_file:
    wav_file.setparams((1, 2, framerate, 0, 'NONE', 'not compressed'))
    wav_file.writeframes((np.array(generated_audio) * 32767).astype(np.int16).tobytes())'''
