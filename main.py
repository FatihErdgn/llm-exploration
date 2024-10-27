# main.py
import numpy as np
import tensorflow as tf
from utils.data_processing import load_data, preprocess_data, save_tokenizer
from utils.model_definition import define_model
from config import BATCH_SIZE, EPOCHS

# 1. Veriyi yükle ve işle
data = load_data()
prompts, decoder_input_data, decoder_target_data, tokenizer = preprocess_data(data)
VOCAB_SIZE = len(tokenizer.word_index) + 1  # Güncel VOCAB_SIZE
save_tokenizer(tokenizer)  # Tokenizer'ı kaydet

print("GPUs:", tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)
print("GPU support?:", tf.test.is_built_with_cuda())

# 3. Modeli tanımla ve eğit
model = define_model()
history = model.fit(
    [prompts, decoder_input_data],
    decoder_target_data[..., np.newaxis],  # Hedef veriye ek bir eksen eklenir
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)

# 4. Modeli kaydet
model.save("models/gpt_model.h5")
print("Model saved successfully.")
