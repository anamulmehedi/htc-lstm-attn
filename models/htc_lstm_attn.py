import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, LSTM, Bidirectional, Concatenate, Dense, Dropout, Multiply, Lambda
from tensorflow.keras.models import Model

def build_htc_lstm_attn_model(input_shape, hp):
    time_steps, num_features = input_shape

    inputs = Input(shape=(time_steps, num_features))

    # Hierarchical Temporal Convolutional (HTC) Layers
    conv1 = Conv1D(filters=hp['filters1'], kernel_size=1, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv3 = Conv1D(filters=hp['filters3'], kernel_size=3, padding='same')(inputs)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    conv5 = Conv1D(filters=hp['filters5'], kernel_size=5, padding='same')(inputs)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    # Concatenate the outputs of the parallel Conv1D layers
    x = Concatenate()([conv1, conv3, conv5])

    # Another Conv1D to combine features
    x = Conv1D(filters=hp['filters_combine'], kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(hp['dropout1'])(x)

    # Bidirectional LSTM for short-term dependencies
    lstm_out = Bidirectional(LSTM(hp['lstm_units'], return_sequences=True))(x)
    lstm_out = Dropout(hp['dropout2'])(lstm_out)

    # Attention Mechanism
    attention_scores = Dense(1, activation='tanh')(lstm_out)
    attention_weights = tf.keras.layers.Softmax(axis=1)(attention_scores)
    context_vector = Multiply()([lstm_out, attention_weights])
    context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)

    # Dense layers for output
    x = Dense(hp['dense_units'], activation='relu')(context_vector)
    x = Dropout(hp['dropout3'])(x)

    # Output layers for temperature and humidity
    temp_output = Dense(1, name='temperature')(x)
    hum_output = Dense(1, name='humidity')(x)

    model = Model(inputs, [temp_output, hum_output])
    model.compile(optimizer='adam', loss='mse', metrics=[['mae'], ['mae']])
    return model
