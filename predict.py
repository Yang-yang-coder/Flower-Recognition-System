import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ======================
# 1. 加载模型
# ======================
model = tf.keras.models.load_model("plant_model.h5")

# ======================
# 2. 类别（必须和训练一致）
# ======================
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# ======================
# 3. 植物养护知识库
# ======================
plant_info = {
    "daisy": """
植物名称：雏菊

养护建议：
1. 喜阳光充足环境
2. 土壤保持微湿
3. 生长期每月施肥一次
4. 避免长期积水
""",

    "dandelion": """
植物名称：蒲公英

养护建议：
1. 适应能力强
2. 保持充足光照
3. 土壤排水良好即可
4. 生长期间适量浇水
""",

    "rose": """
植物名称：玫瑰

养护建议：
1. 每天至少6小时光照
2. 每周浇水2~3次
3. 定期修剪枯枝
4. 花期适量补肥
""",

    "sunflower": """
植物名称：向日葵

养护建议：
1. 需要充足阳光
2. 土壤保持湿润
3. 生长期补充磷钾肥
4. 避免长期阴暗环境
""",

    "tulip": """
植物名称：郁金香

养护建议：
1. 喜凉爽环境
2. 避免高温暴晒
3. 土壤保持微湿
4. 开花后减少浇水
"""
}

# ======================
# 4. 图片路径
# ======================
img_path = "test.png"

# ======================
# 5. 读取图片
# ======================
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # 模型内部已包含 Rescaling 层

# ======================
# 6. 模型预测
# ======================
pred = model.predict(img_array)[0]

# Top3结果
top3_idx = np.argsort(pred)[::-1][:3]

print("\n===== 预测结果 =====")

for i, idx in enumerate(top3_idx):
    print(f"Top{i+1}: {class_names[idx]} ({pred[idx]*100:.2f}%)")

# ======================
# 7. 输出养护建议
# ======================
top1_class = class_names[top3_idx[0]]

print("\n===== 植物养护建议 =====")

if top1_class in plant_info:
    print(plant_info[top1_class])
else:
    print("暂无养护信息")