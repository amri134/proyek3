from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from google.oauth2 import service_account
from google.cloud import firestore

app = Flask(__name__)

# Konfigurasi folder unggahan
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Konfigurasi model TFLite
MODEL_PATH = "model.tflite"
LABELS = ["organik", "berbahaya", "non-organik"]

try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# Detail input dan output dari model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Konfigurasi Firestore
FIREBASE_PROJECT = "jinam-446907"
FIREBASE_CREDENTIALS_PATH = "jinam-446907-firebase-adminsdk-h5ozk-83579d2459.json"

firebase_credentials = service_account.Credentials.from_service_account_file(FIREBASE_CREDENTIALS_PATH)
firestore_client = firestore.Client(project=FIREBASE_PROJECT, credentials=firebase_credentials)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Periksa apakah file ada di request
        file = request.files.get("file")
        if not file or file.filename == "":
            return "Tidak ada file yang diunggah atau file kosong!", 400

        # Simpan file sementara
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)

        try:
            # Pastikan file yang disimpan dibaca kembali untuk diproses
            with open(image_path, "rb") as img_file:
                image_content = img_file.read()

            # Decode dan preprocessing gambar
            image = tf.io.decode_image(image_content, channels=3)
            image = tf.image.resize(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))

            # Normalisasi gambar sesuai tipe data model
            if input_details[0]['dtype'] == np.uint8:
                image = tf.cast(image, tf.float32) / 255.0  # Normalize ke [0, 1]
                image = tf.cast(image * 255.0, tf.uint8)    # Ubah kembali ke uint8
            else:
                image = tf.cast(image, tf.float32) / 255.0  # Normalize ke [0, 1]

            # Menambahkan dimensi batch
            image = tf.expand_dims(image, axis=0)

            # Lakukan inferensi
            interpreter.set_tensor(input_details[0]['index'], image.numpy())
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Konversi output model jika diperlukan
            if output_details[0]['dtype'] == np.uint8:
                scale, zero_point = output_details[0]['quantization']
                output_data = scale * (output_data.astype(np.float32) - zero_point)

            probabilities = tf.nn.softmax(output_data[0]).numpy()
            predicted_index = np.argmax(probabilities)
            predicted_label = LABELS[predicted_index]
            confidence = probabilities[predicted_index] * 100

            # Simpan hasil prediksi ke Firestore
            doc_ref = firestore_client.collection("predictions").document()
            doc_ref.set({
                "label": predicted_label,
                "confidence": round(confidence, 2)
            })

            predictions = [{"label": predicted_label, "confidence": round(confidence, 2)}]

            # Hapus file sementara
            os.remove(image_path)

            return render_template("results.html", predictions=predictions)
        except Exception as e:
            if os.path.exists(image_path):
                os.remove(image_path)  # Hapus file jika terjadi kesalahan
            return f"Terjadi kesalahan saat memproses gambar: {str(e)}", 500

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
