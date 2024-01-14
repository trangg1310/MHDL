# app.py
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.cluster import KMeans
from skimage.transform import resize
from datetime import datetime

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
# Thực hiện thuật toán K-Means để tạo ra bộ codebook
def create_codebook(image, k):
    vectorized_image = image.reshape((-1, 3))
    vectorized_image = np.float32(vectorized_image)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(vectorized_image)

    codebook = kmeans.cluster_centers_
    return codebook, kmeans

# Mã hóa ảnh bằng thuật toán Vector Quantization (VQ) sử dụng K-Means
def encode_image(image, codebook, kmeans):
    vectorized_image = image.reshape((-1, 3))
    vectorized_image = np.float32(vectorized_image)  # Chuyển đổi kiểu dữ liệu nếu cần

    # Tìm index của centroid gần nhất cho mỗi vector pixel
    indices = kmeans.predict(vectorized_image)
    
    # Gán mỗi vector pixel bằng centroid tương ứng
    encoded_image = codebook[indices]

    return encoded_image.astype(np.uint8).reshape(image.shape)

# Giải nén ảnh bằng thuật toán Vector Quantization (VQ) sử dụng K-Means
def decode_image(image, codebook, kmeans):
    vectorized_image = image.reshape((-1, 3))
    vectorized_image = np.float32(vectorized_image)  # Chuyển đổi kiểu dữ liệu nếu cần

    # Tìm index của centroid gần nhất cho mỗi vector pixel
    indices = kmeans.predict(vectorized_image)
    
    # Gán mỗi vector pixel bằng centroid tương ứng
    decoded_image = codebook[indices]

    return decoded_image.astype(np.uint8).reshape(image.shape)


# Đo lường chất lượng của ảnh sau khi giải nén bằng độ đo PSNR
def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def save_image(img, title, result_folder, result_image_name, timestamp):
    result_image_path = os.path.join(result_folder, f"{result_image_name}_{timestamp}.png")
    plt.imshow(img)
    plt.title(title)
    plt.savefig(result_image_path)
    plt.clf()
    return f"{result_image_name}_{timestamp}.png"

timestamp = None
codebook = None
kmeans = None
k = 64  # Số lượng clusters (số lượng centroids)

@app.route("/", methods=["GET", "POST"])
def index():
    global timestamp, codebook, kmeans, k
    if request.method == "POST" and "photo" in request.files:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        photo = request.files["photo"]
        if photo.filename == "" or not allowed_file(photo.filename):
            return redirect(request.url)

        filename = secure_filename(photo.filename)
        photo_path = os.path.join(app.root_path, app.config["UPLOAD_FOLDER"], filename)
        photo.save(photo_path)

        img = imageio.imread(photo_path)

        if codebook is None or kmeans is None:
            codebook, kmeans = create_codebook(img, k)

        img_vq = encode_image(img, codebook, kmeans)        
        img_decoded = decode_image(img_vq, codebook, kmeans)
        
        # Tạo thư mục 'static/images' nếu nó chưa tồn tại
        result_folder = os.path.join(app.root_path, "static", "images")
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # Lưu hình ảnh kết quả và nhận tên file kết quả
        original_image = save_image(img, "Original Image", result_folder, "original", timestamp)
        quantized_image = save_image(img_vq, "Vector Quantized Image", result_folder, "quantized", timestamp)
        decoded_image = save_image(img_decoded, "Decoded Image", result_folder, "decoded", timestamp)

        psnr = calculate_psnr(img, img_decoded)

        return render_template("index.html", photo=photo_path, psnr=psnr,
                               original_image=original_image,
                               quantized_image=quantized_image,
                               decoded_image=decoded_image)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
