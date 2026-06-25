from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 加载模型
try:
    model = tf.keras.models.load_model("plant_model.h5")
    print("模型加载成功")
except Exception as e:
    print(f"警告: 模型加载失败: {e}")
    model = None

# 类别
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# 养护知识库
plant_info = {
    "daisy": "喜阳光充足环境，土壤保持微湿。",
    "dandelion": "适应能力强，保持充足光照。",
    "rose": "每天至少6小时光照，每周浇水2~3次。",
    "sunflower": "需要充足阳光，保持土壤湿润。",
    "tulip": "喜凉爽环境，避免高温暴晒。"
}


def predict_image(img_array):
    """共享预测逻辑：对预处理后的 (1, 224, 224, 3) 数组进行推理。
    返回 (类别名, 置信度%, 养护建议)。"""
    if model is None:
        raise RuntimeError("模型未加载，无法进行识别")
    pred = model.predict(img_array)[0]
    idx = np.argmax(pred)
    result = class_names[idx]
    confidence = round(float(pred[idx]) * 100, 2)
    care = plant_info.get(result, "暂无养护信息")
    return result, confidence, care


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """提供上传图片的访问"""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    care = None
    error = None
    uploaded_image = None

    if request.method == "POST":
        try:
            file = request.files.get("image")
            if file and file.filename:
                filepath = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    file.filename
                )
                file.save(filepath)
                uploaded_image = file.filename

                # 图片预处理
                img = image.load_img(filepath, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                result, confidence, care = predict_image(img_array)
            else:
                error = "未选择文件"
        except Exception as e:
            error = f"识别失败: {str(e)}"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        care=care,
        error=error,
        uploaded_image=uploaded_image
    )


@app.route("/predict_camera", methods=["POST"])
def predict_camera():
    """摄像头拍摄识别接口。
    接收 JSON: {"image": "data:image/jpeg;base64,..."}
    返回 JSON: {"success": true, "result": "...", "confidence": 95.5, "care": "..."}
    """
    try:
        data = request.get_json()
        if not data or not data.get("image"):
            return jsonify({"success": False, "error": "未收到图片数据"}), 400

        image_data_url = data["image"]

        # 去除 data URL 前缀 "data:image/jpeg;base64,"
        if "," not in image_data_url:
            return jsonify({"success": False, "error": "无效的图片格式"}), 400

        header, base64_str = image_data_url.split(",", 1)
        img_bytes = base64.b64decode(base64_str)

        # 打开图片并缩放到模型输入尺寸
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        result, confidence, care = predict_image(img_array)
        return jsonify({
            "success": True,
            "result": result,
            "confidence": confidence,
            "care": care
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
