from flask import Flask, request, render_template
from joblib import load
import numpy as np
import base64
from io import BytesIO
from PIL import Image

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

app = Flask(__name__)

model = load('web\model.joblib')
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

class_names = ['Agaric', 'Bolete', 'Lactarius', 'Russula', 'Stinkhorn']

mushroom_info = {
    'Agaric': 'Nấm mỡ (Agaric): Những loài nấm ăn được thường có phiến nâu hoặc đen. Khi bẻ nón nấm, nếu màu vàng tươi xuất hiện thì nấm có độc. Các loài nấm độc thường có mùi đặc biệt có chịu.',
    'Bolete': 'Nấm rơm (Bolete): Nếu có bất kì vùng màu đỏ nào trên cây nấm, đó là nấm có độc. Khi cắt đôi cây nấm theo chiều dọc và màu xanh xuất hiện, đó là nấm có độc',
    'Lactarius': 'Nấm sữa (Lactarius): Gồm chỉ vài loài được dùng làm thực phẩm. Saffron milk caps: có màu cam sáng, khi bị cắt, bên trong chuyển từ cam sáng sang xanh lá. Blue lactarius: có màu xanh đậm, khi bị cắt bên trong chuyển từ xanh đậm sang xanh lá.',
    'Russula': 'Nấm ngọc cẩm (Russula): Đặt một phần nấm nhỏ lên lưỡi. Nếu có cảm giác rát lưỡi, khó chịu thì đó là nấm độc. Phương pháp có nhiều nguy cơ, không khuyến khích ăn.',
    'Stinkhorn': 'Nấm thối (Stinkhorn): Một số loài chỉ ăn được khi cây còn non. Nhìn chung thì không nên ăn..'
}

def extract_features(img: Image.Image):
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_model.predict(img_array)
    return features.flatten()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_base64 = None
    info = None

    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file.stream)
        
        # Xử lý và dự đoán
        features = extract_features(img)
        pred_num = model.predict([features])[0]
        prediction = class_names[pred_num]
        
        # Lấy thông tin về loại nấm
        info = mushroom_info.get(prediction, 'Không có thông tin.')
        
        # Xử lý ảnh để hiển thị
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return render_template('index.html', prediction=prediction, img_base64=img_base64, info=info)

if __name__ == '__main__':
    app.run(debug=True)