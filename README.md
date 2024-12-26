# CTC_asr
CNN-RNN architecture using CTC for ASR. Insipired by DeepSpeech2.
The convolutional network consisting of a few Residual CNN layers that process the input spectrogram images and output feature maps of those images.
The recurrent network consisting of a few Bidirectional LSTM layers that process the feature maps as a series of distinct timesteps or ‘frames’ that correspond to our desired sequence of output characters.  In other words, it takes the feature maps which are a continuous representation of the audio, and converts them into a discrete representation. Then a linear layer with softmax that uses the LSTM outputs to produce character probabilities for each timestep of the output. So our model takes the Spectrogram images and outputs character probabilities for each timestep or ‘frame’ in that Spectrogram. Finally the CTC algorithm produce the output sequence.
This project uses DVC to implement a ML pipeline that run two stages: data preparation and training.

Resources:
[Audio Deep Learning Made Simple: Automatic Speech Recognition (ASR), How it Works](https://towardsdatascience.com/audio-deep-learning-made-simple-automatic-speech-recognition-asr-how-it-works-716cfce4c706)
[DeepSpeech2](https://arxiv.org/pdf/1512.02595)
[Keras Tutorial](https://keras.io/examples/audio/ctc_asr/)