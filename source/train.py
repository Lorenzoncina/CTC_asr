import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_params import load_params
from jiwer import wer

# A callback class to output a few transcriptions during training
class CallbackEval(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

# CTC loss function definition
def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def build_model(input_dim, output_dim, cnn_filter=32, kernel_size_cnn_1=[11, 41], kernel_size_cnn_2=[11, 21] ,  rnn_layers=5, rnn_units=128, learning_rate=1e-4):
    """Model similar to DeepSpeech2."""
    # Model's input
    input_spectrogram = layers.Input((None, input_dim), name="input")
    # Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    # Convolution layer 1
    x = layers.Conv2D(
        filters=cnn_filter,
        kernel_size=kernel_size_cnn_1,
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    # Convolution layer 2
    x = layers.Conv2D(
        filters=cnn_filter,
        kernel_size=kernel_size_cnn_2,
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model

def train(params, train_dataset, validation_datase):
    fft_length = params.data_preparation.fft_length
    rnn_units = params.train.rnn_units
    rnn_layers = params.train.rnn_layers
    cnn_filter = params.train.cnn_filter
    kernel_size_cnn_1 = params.train.kernel_size_cnn_1
    kernel_size_cnn_2 = params.train.kernel_size_cnn_2
    epochs = params.train.epochs
    learning_rate = float(params.train.learning_rate)
    model_path = params.train.model_path
    model_checkpoints = params.train.model_checkpoints

    # The set of characters accepted in the transcription.
    global char_to_num
    characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
    # Mapping characters to integers
    char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
    # Mapping integers back to original characters
    global num_to_char
    num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )
  
    # Get the model
    global model
    model = build_model(
        input_dim=fft_length // 2 + 1,
        output_dim=char_to_num.vocabulary_size(),
        cnn_filter=cnn_filter,
        kernel_size_cnn_1=kernel_size_cnn_1,
        kernel_size_cnn_2=kernel_size_cnn_2,
        rnn_layers = rnn_layers,
        rnn_units=rnn_units,
    )
    
    # Callback function to check transcription on the val set.
    validation_callback = CallbackEval(validation_dataset)
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        model_checkpoints,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
    )
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[validation_callback, early_stopping,model_checkpoint_callback],
    )

    #save the model
    model.save(model_path)

    

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    args = arg_parser.parse_args()
    #load hyperparameters from the params.yaml file
    params = load_params(params_path=args.config)
    #load datasets
    train_save_path = params.data_preparation.train_save_path
    val_save_path = params.data_preparation.val_save_path
    train_dataset = tf.data.Dataset.load(train_save_path)
    validation_dataset =  tf.data.Dataset.load(val_save_path)
    train(params, train_dataset, validation_dataset)