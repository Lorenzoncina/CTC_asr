import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from load_params import load_params

"""
Function that process the audio in the dataset into an STFT
"""
def encode_single_sample(wav_file, label):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
    file = tf.io.read_file(wavs_path +"/"+ wav_file + ".wav")
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    ###########################################
    ##  Process the label
    ##########################################
    # 7. Convert label to Lower case
    label = tf.strings.lower(label)
    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 9. Map the characters in label to numbers
    label = char_to_num(label)
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label

"""
data preparation funtion. It takes all the arguments needed from the parameters.yaml file
"""
def data_preparation(params):
    global wavs_path, frame_length, frame_step, fft_length, char_to_num
    wavs_path = str(Path(params.data_preparation.wavs_path))
    metadata_path = Path(params.data_preparation.metadata_path)
    frame_length = params.data_preparation.frame_length
    frame_step = params.data_preparation.frame_step
    fft_length = params.data_preparation.fft_length
    batch_size = params.data_preparation.batch_size
    train_save_path = params.data_preparation.train_save_path
    val_save_path = params.data_preparation.val_save_path

    #read data
    metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
    metadata_df = metadata_df[["file_name", "normalized_transcription"]]
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

    #train test split
    split = int(len(metadata_df) * 0.90)
    df_train = metadata_df[:split]
    df_val = metadata_df[split:]

    # The set of characters accepted in the transcription.
    characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
    # Mapping characters to integers
    char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
    # Mapping integers back to original characters
    num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )

    #define train and validation set 
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_train["file_name"]), list(df_train["normalized_transcription"]))
    )
    train_dataset = (
        train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Define the validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_val["file_name"]), list(df_val["normalized_transcription"]))
    )
    validation_dataset = (
        validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    #save train and validation set on disk to be used by the train script   
    train_dataset.save(train_save_path)
    validation_dataset.save(val_save_path)
    
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    args = arg_parser.parse_args()
    #load hyperparameters from the params.yaml file
    params = load_params(params_path=args.config)

    data_preparation(params)