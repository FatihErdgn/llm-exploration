# utils/model_definition.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from config import MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM, LSTM_UNITS

def define_model():
    # Encoder
    encoder_inputs = Input(shape=(MAX_LEN,))
    encoder_embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM)(encoder_inputs)
    encoder_lstm = LSTM(LSTM_UNITS, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(MAX_LEN,))
    decoder_embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM)(decoder_inputs)
    decoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    # Sonuç katmanı
    decoder_dense = Dense(VOCAB_SIZE, activation='softmax')
    output = decoder_dense(decoder_outputs)
    
    # Modeli derle ve dön
    model = Model([encoder_inputs, decoder_inputs], output)
    optimizer=Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
