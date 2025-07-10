import os
#import librosa as ls

dataset_path = "snoring_sound_dataset"
files = os.listdir(dataset_path)
print("Files in dataset:", files)

# for file in files:
#     if file.endswith('.wav'):
#         audio, sr = librosa.load(os.path.join(dataset_path, file), sr=None)
#         print(f"Loaded {file} with sample rate {sr}")