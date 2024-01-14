from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import imageio
from sklearn.preprocessing import scale
from skimage.transform import resize
from datetime import datetime
from PIL import Image
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def save_image(img, title, result_folder, result_image_name, timestamp):
    # Chuyển đổi kiểu dữ liệu từ float sang uint8
    img_uint8 = (img * 255).astype(np.uint8)

    result_image_path = os.path.join(result_folder, f"{result_image_name}_{timestamp}.png")

    # Chuyển đổi từ mảng NumPy sang hình ảnh của Pillow
    img_pil = Image.fromarray(img_uint8)

    # Lưu hình ảnh
    img_pil.save(result_image_path)

    return f"{result_image_name}_{timestamp}.png"

def compute_PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

def compress_image(image, n_clusters):
    # Chuyển đổi ảnh thành ma trận số
    rows, cols, channels = image.shape
    data_matrix = image.reshape((rows * cols, channels))

    # Sử dụng K-Means để tạo codebook
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data_matrix)

    # Mã hóa ảnh bằng cách sử dụng chỉ số từ K-Means
    indices, _ = vq(data_matrix, kmeans.cluster_centers_)

    return kmeans.cluster_centers_, indices.reshape((rows, cols))

def decompress_image(codebook, indices):
    # Giải mã ảnh bằng cách sử dụng codebook và chỉ số
    decoded_image = np.take(codebook, indices, axis=0)

    return decoded_image.reshape((indices.shape[0], indices.shape[1], codebook.shape[1]))
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
        img = resize(img, (1024, 1024))
        k = 32
        
        codebook, compressed_indices = compress_image(img, k)
        decompressed_image = decompress_image(codebook, compressed_indices)
        
        psnr = compute_PSNR(img, decompressed_image)
        
        # Tạo thư mục 'static/images' nếu nó chưa tồn tại
        result_folder = os.path.join(app.root_path, "static", "images")
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # Lưu hình ảnh kết quả và nhận tên file kết quả
        original_image = save_image(img, "Original Image", result_folder, "original", timestamp)

        decoded_image = save_image(decompressed_image, "Decoded Image", result_folder, "decoded", timestamp)


        return render_template("index.html", photo=photo_path, psnr=psnr,
                               original_image=original_image,
                               decoded_image=decoded_image)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
