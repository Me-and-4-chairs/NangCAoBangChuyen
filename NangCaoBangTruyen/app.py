from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)

# ==== Load các mô hình đã train ====
model_paths = {
    'Phân loại rác': 'models/best-PhanLoaiRac.pt',
    'Tình trạng cà chua': 'models/best-TinhTrangCaChua.pt',
    'Phân loại trái cây': 'models/best-PhanLoaiHoaQua.pt'
}
models = {name: YOLO(path) for name, path in model_paths.items()}

# ==== Label map cho từng mô hình ====
label_map = {
    'Phân loại rác': {'plastic': 'Nhựa', 'metal': 'Kim loại', 'Milk_box': 'Hộp sữa'},
    'Tình trạng cà chua': {'fully_ripened': 'Chín', 'half_ripened': 'Nửa chín', 'green': 'Xanh'},
    'Phân loại trái cây': {'apple': 'Táo', 'banana': 'Chuối', 'carrot': 'Cà rốt'}
}


def base64_to_cv2_img(base64_str):
    """Chuyển ảnh base64 sang ảnh OpenCV"""
    header, encoded = base64_str.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    img_np = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(img_np, cv2.IMREAD_COLOR)


@app.route('/')
def index():
    return render_template('index.html', model_names=list(model_paths.keys()))


@app.route('/detect_image', methods=['POST'])
def detect_image():
    data = request.get_json()
    model_name = data.get('model')
    image_data = data.get('image')

    if not model_name or not image_data:
        return jsonify({'error': 'Thiếu dữ liệu'}), 400

    model = models.get(model_name)
    if model is None:
        return jsonify({'error': 'Model không tồn tại'}), 400

    img = base64_to_cv2_img(image_data)
    results = model(img)[0]

    predictions = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        cls_id = int(cls)
        label_eng = results.names[cls_id]
        label_vn = label_map[model_name].get(label_eng, label_eng)
        predictions.append(label_vn)

    return jsonify({'result': predictions})


# ==== CAMERA STREAMING CHO BĂNG CHUYỀN ====
camera_model = models['Phân loại rác']  # Có thể tùy chọn sau
camera_index = 1  # Cổng camera USB

def gen_frames():
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = camera_model(frame)[0]
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls)
            label_eng = results.names[cls_id]
            label_vn = label_map['Phân loại rác'].get(label_eng, label_eng)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_vn, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    model_name = request.args.get('model')
    if model_name not in models:
        return 'Model không hợp lệ', 400

    def gen_frames():
        cap = cv2.VideoCapture(1)  # Cổng USB camera
        model = models[model_name]
        label_dict = label_map[model_name]

        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model(frame)[0]
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(cls)
                label_eng = results.names[cls_id]
                label_vn = label_dict.get(label_eng, label_eng)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_vn, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
