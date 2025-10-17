from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from werkzeug.utils import secure_filename
from mmaction.apis import inference_recognizer, init_recognizer
from operator import itemgetter
from flask_cors import CORS
import cv2
from PIL import Image
import base64
from io import BytesIO
from datetime import datetime
import json

app = Flask(__name__, template_folder='website_test', static_folder='website_test/assets')
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['KEYFRAME_FOLDER'] = 'keyframes'
app.config['HISTORY_FILE'] = 'keyframe_history.json'  # 历史记录存储文件
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['KEYFRAME_FOLDER'], exist_ok=True)

# 初始化模型
config_path = 'configs/recognition/tsn/tsn_20250513_480_tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-Copy1 (1).py'
checkpoint_path = 'work_dirs/best_acc_top1_epoch_23 (1).pth'
labels = open('data/action_label.txt').readlines()
labels = [x.strip() for x in labels]
model = init_recognizer(config_path, checkpoint_path, device="cuda:0")

# 球队映射字典
team_mapping = {
    'CR_kick': ('Real Madrid', '7'),
    'MS_kick': ('Barcelona', '10'),
    'KK_kick': ('AC Milan', '22'),
    'BZ_kick': ('Real Madrid', '9'),
    'mod_kick': ('Real Madrid', '10')
}

# 加载历史记录
def load_history():
    try:
        if os.path.exists(app.config['HISTORY_FILE']):
            with open(app.config['HISTORY_FILE'], 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"加载历史记录失败: {e}")
    return []

# 保存历史记录
def save_history(history):
    try:
        with open(app.config['HISTORY_FILE'], 'w') as f:
            json.dump(history, f)
    except Exception as e:
        print(f"保存历史记录失败: {e}")

keyframe_history = load_history()
MAX_HISTORY = 5  # 最大保存记录数

def generate_jersey_prompt(results):
    for result in results:
        key = result[0]
        if key in team_mapping:
            team, number = team_mapping[key]
            return f"'dress the human in picture with {team}'s cloth.'"
    return "'dress the human in picture with plain clothes'"

def extract_middle_frame(video_path, output_size=(1080,1920)):
    """提取视频中间帧并调整为指定大小的PIL.Image"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("无法打开视频文件")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_pos = total_frames // 2 + 20
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_pos)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("无法读取中间帧")
        return None
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    pil_image = pil_image.resize(output_size)
    
    return pil_image
@app.route('/')  # 确保有这一行
def home():
    return render_template('/index.html')
# 新增的三个HTML页面路由
@app.route('/index.html')
def index():
    return render_template('/index.html')

@app.route('/elements.html')
def generate():
    return render_template('/elements.html')

@app.route('/generic.html')
def picture():
    return render_template('/generic.html')

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(video_file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(save_path)
    
    try:
        # 提取关键帧
        key_frame = extract_middle_frame(save_path)
        
        if key_frame is None:
            raise Exception("无法提取关键帧")
        
        # 保存关键帧到本地用于验证
        keyframe_filename = f"keyframe_{filename.split('.')[0]}.jpg"
        keyframe_path = os.path.join(app.config['KEYFRAME_FOLDER'], keyframe_filename)
        key_frame.save(keyframe_path)
        print(f"关键帧已保存到: {keyframe_path}")
        
        # 将关键帧转换为base64用于前端显示
        buffered = BytesIO()
        key_frame.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 分析视频
        result = inference_recognizer(model, save_path)
        pred_scores = result.pred_score.tolist()
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
        top5_label = score_sorted[:5]
        
        results = [{'label': labels[k[0]], 'score': float(k[1])} for k in top5_label]
        prompt = generate_jersey_prompt([(labels[k[0]], k[1]) for k in top5_label])
        
        # 添加到历史记录
        history_entry = {
            'keyframe': encoded_image,
            'prompt': prompt,
            'timestamp': datetime.now().isoformat(),
            'video_filename': filename,
            'keyframe_path': keyframe_path
        }
        
        keyframe_history.insert(0, history_entry)
        if len(keyframe_history) > MAX_HISTORY:
            keyframe_history.pop()
        save_history(keyframe_history)
        
        return jsonify({
            'results': results,
            'prompt': prompt,
            'key_frame': encoded_image,
            'keyframe_saved_path': keyframe_path,
            'history': keyframe_history  # 返回完整历史记录
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

@app.route('/keyframes/<filename>')
def get_keyframe(filename):
    return send_from_directory(app.config['KEYFRAME_FOLDER'], filename)

@app.route('/get_keyframe_history', methods=['GET'])
def get_keyframe_history():
    return jsonify({'history': keyframe_history})

@app.route('/get_keyframe/<int:index>', methods=['GET'])
def get_keyframe_by_index(index):
    if 0 <= index < len(keyframe_history):
        return jsonify(keyframe_history[index])
    return jsonify({'error': 'Invalid index'}), 404

@app.route('/delete_keyframe/<int:index>', methods=['DELETE'])
def delete_keyframe(index):
    try:
        if 0 <= index < len(keyframe_history):
            # 删除本地保存的关键帧文件
            if os.path.exists(keyframe_history[index]['keyframe_path']):
                os.remove(keyframe_history[index]['keyframe_path'])
            
            # 从历史记录中删除
            keyframe_history.pop(index)
            save_history(keyframe_history)
            
            return jsonify({'success': True})
        return jsonify({'error': 'Invalid index'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_style_transfer', methods=['POST'])
def handle_style_transfer():
    data = request.json
    
    try:
        # 解码图片
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data))
        
        # 使用你的Gemini代码生成新图片
        # ... 你的生成逻辑 ...
        
        # 将生成的图片转换为base64
        buffered = BytesIO()
        generated_image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'generated_image': encoded_image,
            'prompt': data['prompt']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 指定端口和入口
    app.run(host='0.0.0.0', port=5000, debug=True)