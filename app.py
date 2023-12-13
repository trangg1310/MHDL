from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from skimage.transform import resize
from datetime import datetime

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def vector_quantization(img, codebook):
    m, n, c = img.shape
    vec_q = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            diff = codebook - img[i, j, :]
            distance = np.sqrt(np.sum(diff ** 2, axis=1))
            index = np.argmin(distance)
            vec_q[i, j] = index

    return vec_q

def compute_PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def create_codebook(img, k):
    m, n, c = img.shape
    pixels = img.reshape((m * n, c))
    pixels = scale(pixels)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    codebook = kmeans.cluster_centers_
    return codebook

def save_image(img, title, result_folder, result_image_name):
    result_image_path = os.path.join(result_folder, f"{result_image_name}_{timestamp}.png")
    plt.imshow(img)
    plt.title(title)
    plt.savefig(result_image_path)
    plt.clf()
    return f"{result_image_name}_{timestamp}.png"


timestamp = None
@app.route("/", methods=["GET", "POST"])
def index():
    global timestamp
    if request.method == "POST" and "photo" in request.files:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        photo = request.files["photo"]
        if photo.filename == "" or not allowed_file(photo.filename):
            return redirect(request.url)

        filename = secure_filename(photo.filename)
        photo_path = os.path.join(app.root_path, app.config["UPLOAD_FOLDER"], filename)
        photo.save(photo_path)

        img = imageio.imread(photo_path)
        img = resize(img, (256, 256))
        k = 20
        codebook = create_codebook(img, k)
        img_vq = vector_quantization(img, codebook)
        img_decoded = np.zeros_like(img)
        for i in range(256):
            for j in range(256):
                img_decoded[i, j, :] = codebook[int(img_vq[i, j])]

        # Tạo thư mục 'static/images' nếu nó chưa tồn tại
        result_folder = os.path.join(app.root_path, "static", "images")
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # Lưu hình ảnh kết quả và nhận tên file kết quả
        original_image = save_image(img, "Original Image", result_folder, "original")
        quantized_image = save_image(img_vq, "Vector Quantized Image", result_folder, "quantized")
        decoded_image = save_image(img_decoded, "Decoded Image", result_folder, "decoded")

        psnr = compute_PSNR(img, img_decoded)

        compressed_size_before = os.path.getsize(photo_path) / 1024  # Kích thước trước khi nén (KB)
        compressed_size_after = os.path.getsize(os.path.join(result_folder, f"quantized_{timestamp}.png")) / 1024
        compression_ratio = compressed_size_before / compressed_size_after
        
        return render_template("index.html", photo=photo_path, psnr=psnr,
                               original_image=original_image,
                               quantized_image=quantized_image,
                               decoded_image=decoded_image,
                               compression_ratio=compression_ratio)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)