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
import subprocess
app = Flask(__name__, template_folder='website_test', static_folder='website_test/assets')
CORS(app)
app.config['UPLOAD_FOLDER'] = 'website_test/uploads'
app.config['KEYFRAME_FOLDER'] = 'website_test/keyframes'
app.config['STYLE_TRANSFER_FOLDER'] = 'website_test/style_transfer'  # 新增风格迁移文件夹
app.config['BACK_VIDEOS'] = 'website_test/back_video' 
app.config['HISTORY_FILE'] = 'website_test/keyframe_history.json'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['KEYFRAME_FOLDER'], exist_ok=True)
os.makedirs(app.config['STYLE_TRANSFER_FOLDER'], exist_ok=True)  # 确保文件夹存在
os.makedirs(app.config['BACK_VIDEOS'], exist_ok=True)  # 确保文件夹存在
# 初始化模型
config_path = 'configs/recognition/tsn/tsn_20250513_480_tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
checkpoint_path = 'work_dirs/tsn_20250513_480_tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-Copy1/best_acc_top1_epoch_28.pth'
labels = open('data/action_label.txt').readlines()
labels = [x.strip() for x in labels]
model = init_recognizer(config_path, checkpoint_path, device="cuda:1")

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
    
    try:
        video_file.save(save_path)
        print(f"视频已保存到: {save_path}")
        
        cap = cv2.VideoCapture(save_path)
        if not cap.isOpened():
            raise Exception("无法打开视频文件")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"视频信息: {total_frames}帧, {fps} FPS, {duration}秒")
        
        if duration <= 0:
            raise Exception("无效的视频时长")
        
        # 计算分割点（分成5段）
        segment_duration = duration / 5
        segment_starts = [i * segment_duration for i in range(5)]
        segment_ends = [(i + 1) * segment_duration for i in range(5)]
        
        # 提取关键帧（整个视频的中间帧 + 5个分段的中间帧）
        key_frames = []
        
        # 整个视频的中间帧
        middle_frame_pos = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_pos)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            key_frames.append({
                'type': 'full',
                'position': 'middle',
                'frame_num': middle_frame_pos,
                'time': middle_frame_pos / fps,
                'image': pil_image
            })
            print("已提取整体关键帧")
        else:
            print("无法提取整体关键帧")
        
        # 各分段的中间帧
        for i in range(5):
            frame_pos = int(segment_starts[i] + segment_duration/2) * fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                key_frames.append({
                    'type': 'segment',
                    'segment_num': i + 1,
                    'frame_num': frame_pos,
                    'time': frame_pos / fps,
                    'image': pil_image
                })
                print(f"已提取分段 {i+1} 关键帧")
            else:
                print(f"无法提取分段 {i+1} 关键帧")
        
        cap.release()
        
        # 分析视频（整体分析）
        print("开始整体分析...")
        full_result = inference_recognizer(model, save_path)
        pred_scores = full_result.pred_score.tolist()
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
        top5_label = score_sorted[:5]
        
        results = [{'label': labels[k[0]], 'score': float(k[1])} for k in top5_label]
        prompt = generate_jersey_prompt([(labels[k[0]], k[1]) for k in top5_label])
        print(f"整体分析完成: {results}")
        
        # 分析各片段 - 为每个片段创建临时视频并独立分析
        segment_results = []
        
        # 创建临时目录存放片段视频
        segment_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'segments')
        os.makedirs(segment_dir, exist_ok=True)
        
        # 使用ffmpeg分割视频并分析每个片段
        for i in range(5):
            segment_start = segment_starts[i]
            segment_end = segment_ends[i]
            segment_filename = f"segment_{i+1}_{filename}"
            segment_path = os.path.join(segment_dir, segment_filename)
            
            # 使用ffmpeg提取视频片段
            ffmpeg_command = [
                'ffmpeg',
                '-i', save_path,
                '-ss', str(segment_start),
                '-to', str(segment_end),
                '-c', 'copy',
                segment_path
            ]
            
            try:
                # 运行ffmpeg命令
                subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"已创建分段视频: {segment_path}")
                
                # 分析该片段
                segment_result = inference_recognizer(model, segment_path)
                seg_pred_scores = segment_result.pred_score.tolist()
                seg_score_tuples = tuple(zip(range(len(seg_pred_scores)), seg_pred_scores))
                seg_score_sorted = sorted(seg_score_tuples, key=itemgetter(1), reverse=True)
                seg_top5_label = seg_score_sorted[:1]  # 只取最高分
                
                # 查找该片段对应的关键帧信息
                segment_frame_info = next(
                    (f for f in key_frames if f.get('segment_num') == i+1),
                    {'time': segment_start + segment_duration/2}
                )
                
                segment_results.append({
                    'segment_num': i+1,
                    'time': segment_frame_info.get('time', segment_start + segment_duration/2),
                    'results': [{'label': labels[k[0]], 'score': float(k[1])} for k in seg_top5_label]
                })
                print(f"分段 {i+1} 分析完成")
            except Exception as e:
                print(f"分段 {i+1} 分析失败: {str(e)}")
                segment_results.append({
                    'segment_num': i+1,
                    'time': segment_start + segment_duration/2,
                    'results': [{'label': 'error', 'score': 0}]
                })
            finally:
                # 删除临时分段视频
                if os.path.exists(segment_path):
                    try:
                        os.remove(segment_path)
                        print(f"已删除临时分段视频: {segment_path}")
                    except Exception as e:
                        print(f"删除临时分段视频失败: {str(e)}")
        
        # 保存关键帧并转换为base64
        keyframe_base64 = []
        for frame_info in key_frames:
            try:
                buffered = BytesIO()
                frame_info['image'].save(buffered, format="JPEG")
                encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # 保存文件
                keyframe_filename = f"keyframe_{frame_info.get('type', 'unknown')}_{filename.split('.')[0]}.jpg"
                keyframe_path = os.path.join(app.config['KEYFRAME_FOLDER'], keyframe_filename)
                frame_info['image'].save(keyframe_path)
                
                keyframe_base64.append({
                    'type': frame_info.get('type'),
                    'segment_num': frame_info.get('segment_num'),
                    'image': encoded_image,
                    'path': keyframe_path
                })
                print(f"关键帧保存成功: {keyframe_path}")
            except Exception as e:
                print(f"关键帧保存失败: {str(e)}")
        
        # 添加到历史记录（只添加整体关键帧）
        full_frame = next((f for f in keyframe_base64 if f['type'] == 'full'), None)
        if full_frame:
            history_entry = {
                'keyframe': full_frame['image'],
                'prompt': prompt,
                'timestamp': datetime.now().isoformat(),
                'video_filename': filename,
                'keyframe_path': full_frame['path']
            }
            
            keyframe_history.insert(0, history_entry)
            if len(keyframe_history) > MAX_HISTORY:
                keyframe_history.pop()
            save_history(keyframe_history)
        
        return jsonify({
            'results': results,
            'prompt': prompt,
            'key_frame': full_frame['image'] if full_frame else '',
            'keyframe_saved_path': full_frame['path'] if full_frame else '',
            'history': keyframe_history,
            'segments': segment_results,
            'segment_keyframes': [f for f in keyframe_base64 if f['type'] == 'segment']
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        
    finally:
        # 确保清理临时文件
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
                print(f"已删除原始视频: {save_path}")
            except Exception as e:
                print(f"删除原始视频失败: {str(e)}")

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
        model='gemini-2.0-flash',
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

    
    
#20250907修改，添加后端选择视频代码
# 添加获取预置视频列表的路由
@app.route('/get_preset_videos', methods=['GET'])
def get_preset_videos():
    preset_video_dir = app.config['BACK_VIDEOS']  # 预置视频存放目录
    if not os.path.exists(preset_video_dir):
        os.makedirs(preset_video_dir)
        return jsonify({'videos': []})
    
    # 获取所有视频文件
    videos = [f for f in os.listdir(preset_video_dir) 
              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))]
    
    return jsonify({'videos': videos})

# 添加分析预置视频的路由
@app.route('/analyze_preset_video', methods=['POST'])
def analyze_preset_video():
    data = request.json
    if not data or 'video_name' not in data:
        return jsonify({'error': 'No video name provided'}), 400
    
    video_name = data['video_name']
    preset_video_dir = app.config['BACK_VIDEOS']
    video_path = os.path.join(preset_video_dir, video_name)
    
    if not os.path.exists(video_path):
        return jsonify({'error': 'Preset video not found'}), 404
    
    try:
        # 使用与analyze_video相同的处理逻辑
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("无法打开视频文件")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"视频信息: {total_frames}帧, {fps} FPS, {duration}秒")
        
        if duration <= 0:
            raise Exception("无效的视频时长")
        
        # 计算分割点（分成5段）
        segment_duration = duration / 5
        segment_starts = [i * segment_duration for i in range(5)]
        segment_ends = [(i + 1) * segment_duration for i in range(5)]
        
        # 提取关键帧（整个视频的中间帧 + 5个分段的中间帧）
        key_frames = []
        
        # 整个视频的中间帧
        middle_frame_pos = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_pos)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            key_frames.append({
                'type': 'full',
                'position': 'middle',
                'frame_num': middle_frame_pos,
                'time': middle_frame_pos / fps,
                'image': pil_image
            })
            print("已提取整体关键帧")
        else:
            print("无法提取整体关键帧")
        
        # 各分段的中间帧
        for i in range(5):
            frame_pos = int(segment_starts[i] + segment_duration/2) * fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                key_frames.append({
                    'type': 'segment',
                    'segment_num': i + 1,
                    'frame_num': frame_pos,
                    'time': frame_pos / fps,
                    'image': pil_image
                })
                print(f"已提取分段 {i+1} 关键帧")
            else:
                print(f"无法提取分段 {i+1} 关键帧")
        
        cap.release()
        
        # 分析视频（整体分析）
        print("开始整体分析...")
        full_result = inference_recognizer(model, video_path)
        pred_scores = full_result.pred_score.tolist()
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
        top5_label = score_sorted[:5]
        
        results = [{'label': labels[k[0]], 'score': float(k[1])} for k in top5_label]
        prompt = generate_jersey_prompt([(labels[k[0]], k[1]) for k in top5_label])
        print(f"整体分析完成: {results}")
        
        # 分析各片段 - 为每个片段创建临时视频并独立分析
        segment_results = []
        
        # 创建临时目录存放片段视频
        segment_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'segments')
        os.makedirs(segment_dir, exist_ok=True)
        
        # 使用ffmpeg分割视频并分析每个片段
        for i in range(5):
            segment_start = segment_starts[i]
            segment_end = segment_ends[i]
            segment_filename = f"segment_{i+1}_{video_name}"
            segment_path = os.path.join(segment_dir, segment_filename)
            
            # 使用ffmpeg提取视频片段
            ffmpeg_command = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(segment_start),
                '-to', str(segment_end),
                '-c', 'copy',
                segment_path
            ]
            
            try:
                # 运行ffmpeg命令
                subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"已创建分段视频: {segment_path}")
                
                # 分析该片段
                segment_result = inference_recognizer(model, segment_path)
                seg_pred_scores = segment_result.pred_score.tolist()
                seg_score_tuples = tuple(zip(range(len(seg_pred_scores)), seg_pred_scores))
                seg_score_sorted = sorted(seg_score_tuples, key=itemgetter(1), reverse=True)
                seg_top5_label = seg_score_sorted[:1]  # 只取最高分
                
                # 查找该片段对应的关键帧信息
                segment_frame_info = next(
                    (f for f in key_frames if f.get('segment_num') == i+1),
                    {'time': segment_start + segment_duration/2}
                )
                
                segment_results.append({
                    'segment_num': i+1,
                    'time': segment_frame_info.get('time', segment_start + segment_duration/2),
                    'results': [{'label': labels[k[0]], 'score': float(k[1])} for k in seg_top5_label]
                })
                print(f"分段 {i+1} 分析完成")
            except Exception as e:
                print(f"分段 {i+1} 分析失败: {str(e)}")
                segment_results.append({
                    'segment_num': i+1,
                    'time': segment_start + segment_duration/2,
                    'results': [{'label': 'error', 'score': 0}]
                })
            finally:
                # 删除临时分段视频
                if os.path.exists(segment_path):
                    try:
                        os.remove(segment_path)
                        print(f"已删除临时分段视频: {segment_path}")
                    except Exception as e:
                        print(f"删除临时分段视频失败: {str(e)}")
        
        # 保存关键帧并转换为base64
        keyframe_base64 = []
        for frame_info in key_frames:
            try:
                buffered = BytesIO()
                frame_info['image'].save(buffered, format="JPEG")
                encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # 保存文件
                keyframe_filename = f"keyframe_{frame_info.get('type', 'unknown')}_{video_name.split('.')[0]}.jpg"
                keyframe_path = os.path.join(app.config['KEYFRAME_FOLDER'], keyframe_filename)
                frame_info['image'].save(keyframe_path)
                
                keyframe_base64.append({
                    'type': frame_info.get('type'),
                    'segment_num': frame_info.get('segment_num'),
                    'image': encoded_image,
                    'path': keyframe_path
                })
                print(f"关键帧保存成功: {keyframe_path}")
            except Exception as e:
                print(f"关键帧保存失败: {str(e)}")
        
        # 添加到历史记录（只添加整体关键帧）
        full_frame = next((f for f in keyframe_base64 if f['type'] == 'full'), None)
        if full_frame:
            history_entry = {
                'keyframe': full_frame['image'],
                'prompt': prompt,
                'timestamp': datetime.now().isoformat(),
                'video_filename': video_name,
                'keyframe_path': full_frame['path']
            }
            
            keyframe_history.insert(0, history_entry)
            if len(keyframe_history) > MAX_HISTORY:
                keyframe_history.pop()
            save_history(keyframe_history)
        
        return jsonify({
            'results': results,
            'prompt': prompt,
            'key_frame': full_frame['image'] if full_frame else '',
            'keyframe_saved_path': full_frame['path'] if full_frame else '',
            'history': keyframe_history,
            'segments': segment_results,
            'segment_keyframes': [f for f in keyframe_base64 if f['type'] == 'segment']
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 添加服务预置视频的路由
@app.route('/back_video/<filename>')
def serve_preset_video(filename):
    preset_video_dir = app.config['BACK_VIDEOS']
    return send_from_directory(preset_video_dir, filename)
    
#20250907修改，添加后端选择视频代码结束   
    
    
    
    
    
if __name__ == '__main__':
    # 指定端口和入口
    app.run(host='0.0.0.0', port=5000, debug=True)
