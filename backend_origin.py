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
from google import genai
from google.genai import types
import time

app = Flask(__name__, template_folder='website_test', static_folder='website_test/assets')
CORS(app)
app.config['UPLOAD_FOLDER'] = 'website_test/uploads'
app.config['KEYFRAME_FOLDER'] = 'website_test/keyframes'
app.config['STYLE_TRANSFER_FOLDER'] = 'website_test/style_transfer'  # 新增风格迁移文件夹
app.config['HISTORY_FILE'] = 'website_test/keyframe_history.json'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['KEYFRAME_FOLDER'], exist_ok=True)
os.makedirs(app.config['STYLE_TRANSFER_FOLDER'], exist_ok=True)  # 确保文件夹存在

# 初始化模型
config_path = 'configs/recognition/tsn/tsn_20250513_480_tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-Copy1.py'
checkpoint_path = 'work_dirs/new_backend_model.pth'
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

# 球队球衣映射
# 修改球队球衣映射，使用绝对路径
TEAM_JERSEY_MAP = {
    'Barcelona': os.path.join(app.root_path, 'website_test', 'images', 'barcelona.png'),
    'Real Madrid': os.path.join(app.root_path, 'website_test', 'images', 'realmadrid.png'),
    'AC Milan': os.path.join(app.root_path, 'website_test', 'images', 'ac.png')
}

# 初始化Gemini
client = genai.Client(api_key="AIzaSyACbfe8yCB7DMdNjiaASK-nGxbxMFZFHe0")

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

@app.route('/get_recent_style_transfers', methods=['GET'])
def get_recent_style_transfers():
    try:
        # Get list of files in style transfer folder, sorted by modification time
        files = os.listdir(app.config['STYLE_TRANSFER_FOLDER'])
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['STYLE_TRANSFER_FOLDER'], x)), reverse=True)
        
        # Limit to most recent 3 files
        recent_files = files[:3]
        
        # Prepare response with file paths and timestamps
        result = []
        for file in recent_files:
            file_path = os.path.join(app.config['STYLE_TRANSFER_FOLDER'], file)
            timestamp = os.path.getmtime(file_path)
            result.append({
                'path': f'/style_transfer/{file}',
                'timestamp': datetime.fromtimestamp(timestamp).isoformat()
            })
            
        return jsonify({'recent_works': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Also add this route to serve style transfer images
@app.route('/style_transfer/<filename>')
def serve_style_transfer(filename):
    return send_from_directory(app.config['STYLE_TRANSFER_FOLDER'], filename)

def generate_jersey_prompt(results):
    for result in results:
        key = result[0]
        if key in team_mapping:
            team, number = team_mapping[key]
            return f"dress the human in picture with {team}'s jersey, number {number}"
    return "dress the human in picture with plain clothes"

def extract_team_name(prompt_text):
    prompt_text = prompt_text.lower()
    for team in TEAM_JERSEY_MAP.keys():
        if team.lower() in prompt_text:
            return team
    return None
def get_jersey_image(team_name):
    """根据球队名称获取对应的球衣图片"""
    jersey_path = TEAM_JERSEY_MAP.get(team_name)
    if not jersey_path or not os.path.exists(jersey_path):
        raise ValueError(f"No jersey image found for team: {team_name}")
    
    with open(jersey_path, 'rb') as f:
        return f.read()   
def extract_middle_frame(video_path, output_size=(1080,1920)):
    """提取视频中间帧并调整为指定大小的PIL.Image"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("无法打开视频文件")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_pos = total_frames // 2 
    
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

@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory(os.path.join(app.root_path, 'website_test', 'images'), filename)

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

# 新增：处理风格迁移请求
@app.route('/generate_style_transfer', methods=['POST'])
def handle_style_transfer():
    data = request.json
    print("收到风格迁移请求")  # 添加这行
    if not data or 'image' not in data or 'prompt' not in data:
        return jsonify({'error': 'Missing image or prompt data'}), 400
    
    try:
        # 解码图片
       #print("开始解码")
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data))
        
        # 提取球队名称
        team = extract_team_name(data['prompt'])
        if not team:
            #暂时默认皇马，如果偶尔有bug
            team=extract_team_name("dress the human in picture with Real Madrid's cloth.")
            #return jsonify({'error': 'No team identified in prompt'}), 400
            
        # 获取球衣图片
        jersey_path = TEAM_JERSEY_MAP.get(team)
        if not jersey_path or not os.path.exists(jersey_path):
            print("没找到球衣")
            return jsonify({'error': f'Jersey image not found for team: {team}'}), 404
        image_bytes=get_jersey_image(team)
        response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
        types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/png',
        ),
      'Caption this image,and remember it,I have mission then want you do,based on the cloth in the image.'
        ]
        )

        print(response.text)
        # 构建详细提示
       # detailed_prompt = (
           # f"Apply 'Soccer Jersey Transfer' style to the image UNDER STRICT CONDITIONS:\n"
          #3  f"1. ABSOLUTELY NO ALTERATIONS TO HUMAN BODY:\n"
          #  f"   - PRESERVE ORIGINAL: Head angle, limb positions, facial features, body proportions EXACTLY as input\n"
          #  f"   - NO MORPHING: Do not stretch/bend/rotate joints or change posture in any way\n"
          #  f"2. ALLOWED STYLIZATION:\n"
         #   f"   - {data['prompt']}\n"
         #   f"   - Apply jersey texture and colors from reference image\n"
         #   f"3. IMPORTANT:\n"
        #    f"   - Keep background unchanged\n"
        #    f"   - Maintain lighting and shadows consistent with original\n"
        #    f"   - Do not add or remove any elements from the image"
        #)
        
        # 使用Gemini生成风格迁移
        #model = 'gemini-2.0-flash'
        text_input = (
           data['prompt'],
         "Apply 'Colored Pencil Art' style transfer to the image UNDER STRICT CONDITIONS:",
          "1.  ABSOLUTELY NO ALTERATIONS TO HUMAN BODY:",
          "   - PRESERVE ORIGINAL: Head angle, limb positions, facial features, body proportions EXACTLY as input",
           "   - NO MORPHING: Do not stretch/bend/rotate joints or change posture in any way",
           "   - NO REPLACEMENT: Never substitute the human with athletes/celebrities or anyone",
           "   = NO border/frame: Add border/frame if space is not allowed"
           "",
           "2.  ALLOWED STYLIZATION:",
          "   - Cloth: Clothes the human wear",
           "### limited ###",
           "- Use non-rigid style transfer: Only affect color/texture domains",
           "- Disable pose estimation modules to prevent interference",
           "- Run integrity check: Compare limb coordinates pre/post processing",
           "",
          "### important tips ###",
           " DON'T: Turn arms into another poses",
           " DON'T: Adjust head tilt to match famous athletes",
           " DON'T: Modify hand shapes to hold sports equipment",
           " DON'T: Add texture content",   
           " YOU MUST: Change the clothes to what I have give to you, the color must be the same"
           " Remember: You just do style transfer"
           "",
           "Style reference: with visible strokes, soft color blending, and a hand - drawn aesthetic "
           )
        response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        #model="imagen-3.0-generate-002",
        contents=[text_input, image],
        config=types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE'],
 
                                           )
    
                                                 )
        # 准备图像数据
       # image_parts = [
         #   {
         #       "mime_type": "image/jpeg",
          #      "data": data['image']  # 使用base64字符串
         #   },
         #   {
         #       "mime_type": "image/png",
        #        "data": base64.b64encode(open(jersey_path, "rb").read()).decode('utf-8')
          #  }
       # ]
        
        # 发送请求
       # response = model.generate_content(
         #   contents=[detailed_prompt, *image_parts],
         #   generation_config={
         #       "temperature": 0.3,
         #       "max_output_tokens": 2048,
         #   }
       # )
        
        # 处理响应
        if response.candidates and response.candidates[0].content.parts:
            generated_image_data = None
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    generated_image_data = part.inline_data.data
                    break
            
            if generated_image_data:
                # 保存生成的图片
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"style_transfer_{timestamp}.png"
                output_path = os.path.join(app.config['STYLE_TRANSFER_FOLDER'], output_filename)
                
                with open(output_path, "wb") as f:
                    f.write(generated_image_data)
                
                # 转换为base64用于返回
                encoded_result = base64.b64encode(generated_image_data).decode('utf-8')
                
                return jsonify({
                    'generated_image': encoded_result,
                    'saved_path': output_path
                })
        
        return jsonify({'error': 'No image generated by Gemini'}), 500
        
    except Exception as e:
        print(f"风格迁移错误: {str(e)}")
        return jsonify({'error': f'风格迁移失败: {str(e)}'}), 500

if __name__ == '__main__':
    # 指定端口和入口
    app.run(host='0.0.0.0', port=5000, debug=True)