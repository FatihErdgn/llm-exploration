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

gpus = tf.config.list_physical_devices('GPU')

print(gpus)
print("GPU support?:", tf.test.is_built_with_cuda())
print("Is GPU available?:",tf.test.is_gpu_available())

if gpus:
    try:
        # Yalnızca GPU'ları görünür yapın
        tf.config.set_visible_devices(gpus[0], 'GPU')
        # Bellek büyümesini etkinleştirerek, GPU bellek kullanımını kademeli hale getirin
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print("Hata:", e)
else:
    print("GPU bulunamadı.")

# 3. Modeli tanımla
model = define_model()

# Modeli eğit
history = model.fit(
    [prompts, decoder_input_data],
    decoder_target_data,  # Ek eksen olmadan
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)


# 4. Modeli kaydet
model.save("models/gpt_model.h5")
print("Model saved successfully.")
