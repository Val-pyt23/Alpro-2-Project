import os
import cv2
import numpy as np
from keras.models import load_model

# Memuat model
model_validasi = load_model('model_validasi_256x256.h5')  # Model validasi gambar
model_tbc = load_model('hasil_model_cnn.h5')  # Model prediksi TBC

# Fungsi preprocessing untuk model validasi
def preprocess_image_validation(image_path, image_size=256):
    """
    Preprocessing gambar untuk model validasi.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Gambar tidak ditemukan atau format tidak didukung.")
    image = cv2.resize(image, (image_size, image_size))  # Resize ke 256x256
    image = image.astype('float32') / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=-1)  # Tambahkan channel (grayscale -> 1 channel)
    image = np.expand_dims(image, axis=0)   # Tambahkan batch dimension
    return image

# Fungsi preprocessing untuk model prediksi TBC
def preprocess_image_tbc(image_path, image_size=256):
    """
    Preprocessing gambar untuk model TBC.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Gambar tidak ditemukan atau format tidak didukung.")
    image = cv2.resize(image, (image_size, image_size))  # Resize ke 256x256
    image = image.astype('float32') / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=-1)  # Tambahkan channel (grayscale -> 1 channel)
    image = np.expand_dims(image, axis=0)   # Tambahkan batch dimension
    return image

# Fungsi prediksi validasi
def predict_image_validation(image_path):
    """
    Memproses gambar dan memprediksi apakah itu gambar rontgen paru-paru atau bukan.
    """
    processed_image = preprocess_image_validation(image_path)
    prediction = model_validasi.predict(processed_image)
    confidence = prediction[0][0]

    # Interpretasi hasil prediksi
    if confidence >= 0.5:
        return True, f"Gambar rontgen paru-paru (confidence: {confidence:.2f})"
    else:
        return False, f"Bukan gambar rontgen paru-paru (confidence: {confidence:.2f})"

# Fungsi prediksi TBC
def predict_tbc(image_path):
    """
    Memproses gambar rontgen paru-paru dan memprediksi apakah terkena TBC atau normal.
    """
    processed_image = preprocess_image_tbc(image_path)
    prediction = model_tbc.predict(processed_image)
    confidence = prediction[0][0]

    # Interpretasi hasil prediksi
    if confidence >= 0.5:
        return f"TBC (confidence: {confidence:.2f})"
    else:
        return f"Normal (confidence: {confidence:.2f})"

# Program utama
if __name__ == "__main__":
    # Meminta input path gambar dari user
    image_path = input("Masukkan path gambar yang ingin diuji: ")
    
    if os.path.exists(image_path):
        try:
            # Validasi apakah gambar adalah rontgen paru-paru
            is_rontgen, validation_result = predict_image_validation(image_path)
            print(f"Hasil validasi: {validation_result}")
            
            if is_rontgen:
                # Jika gambar valid, prediksi apakah TBC atau Normal
                tbc_result = predict_tbc(image_path)
                print(f"Hasil prediksi TBC: {tbc_result}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Path gambar tidak ditemukan. Pastikan Anda memasukkan path yang benar.")
