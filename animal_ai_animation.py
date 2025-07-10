import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from flask import Flask, request, render_template, send_file, jsonify, url_for
import subprocess
import tempfile
import uuid
import math
import random
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_animal_body(image):
    """동물 몸체 감지 (색상 기반 세그멘테이션)"""
    # HSV 색상 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 동물 털 색상 범위 (갈색, 검정, 흰색 등)
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([20, 255, 255])
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    # 마스크 생성
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # 마스크 결합
    mask = cv2.bitwise_or(mask_brown, mask_black)
    mask = cv2.bitwise_or(mask, mask_white)
    
    # 노이즈 제거
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def apply_breathing_effect(image, frame_num, total_frames):
    """동물 호흡 효과 - 더 자연스러운 확대/축소"""
    height, width = image.shape[:2]
    
    # 호흡 주기 (약 3초)
    breathing_cycle = math.sin(frame_num * 2 * math.pi / (30 * 3))
    
    # 부드러운 호흡 효과
    scale_factor = 1.0 + breathing_cycle * 0.025  # 2.5% 변화
    
    # 중앙 기준으로 확대/축소
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    resized = cv2.resize(image, (new_width, new_height))
    
    # 중앙에서 원본 크기만큼 크롭
    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2
    cropped = resized[start_y:start_y+height, start_x:start_x+width]
    
    return cropped

def apply_tail_wagging(image, frame_num, total_frames):
    """꼬리 흔들기 효과"""
    height, width = image.shape[:2]
    
    # 꼬리 영역 (오른쪽 하단)
    tail_region = image[height//2:, width//2:]
    
    if tail_region.size > 0:
        # 꼬리 흔들기 주기
        wag_cycle = math.sin(frame_num * 0.3) * 0.1  # 10도 회전
        
        # 회전 중심점
        center_x, center_y = tail_region.shape[1]//2, tail_region.shape[0]//2
        
        # 회전 행렬
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), wag_cycle * 180 / math.pi, 1.0)
        
        # 회전 적용
        rotated_tail = cv2.warpAffine(tail_region, rotation_matrix, (tail_region.shape[1], tail_region.shape[0]))
        
        # 원본에 적용
        image[height//2:, width//2:] = rotated_tail
    
    return image

def apply_ear_movement(image, frame_num, total_frames):
    """귀 움직임 효과"""
    height, width = image.shape[:2]
    
    # 귀 영역 (상단)
    left_ear_region = image[:height//3, :width//3]
    right_ear_region = image[:height//3, 2*width//3:]
    
    # 귀 움직임 주기
    ear_cycle = math.sin(frame_num * 0.2) * 0.05  # 5도 회전
    
    if left_ear_region.size > 0:
        # 왼쪽 귀 회전
        center_x, center_y = left_ear_region.shape[1]//2, left_ear_region.shape[0]//2
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), ear_cycle * 180 / math.pi, 1.0)
        rotated_left_ear = cv2.warpAffine(left_ear_region, rotation_matrix, (left_ear_region.shape[1], left_ear_region.shape[0]))
        image[:height//3, :width//3] = rotated_left_ear
    
    if right_ear_region.size > 0:
        # 오른쪽 귀 회전 (반대 방향)
        center_x, center_y = right_ear_region.shape[1]//2, right_ear_region.shape[0]//2
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -ear_cycle * 180 / math.pi, 1.0)
        rotated_right_ear = cv2.warpAffine(right_ear_region, rotation_matrix, (right_ear_region.shape[1], right_ear_region.shape[0]))
        image[:height//3, 2*width//3:] = rotated_right_ear
    
    return image

def apply_eye_blinking(image, frame_num, total_frames):
    """눈 깜빡임 효과"""
    height, width = image.shape[:2]
    
    # 눈 영역 (상단 중앙)
    left_eye_region = image[height//4:height//2, width//4:width//2]
    right_eye_region = image[height//4:height//2, width//2:3*width//4]
    
    # 깜빡임 주기 (4초마다)
    blink_period = 120  # 30fps 기준 4초
    blink_frame = frame_num % blink_period
    
    if 5 < blink_frame < 15:  # 깜빡임 구간
        # 눈을 어둡게
        darken_factor = 0.2
        left_eye_region = (left_eye_region * darken_factor).astype(np.uint8)
        right_eye_region = (right_eye_region * darken_factor).astype(np.uint8)
        
        image[height//4:height//2, width//4:width//2] = left_eye_region
        image[height//4:height//2, width//2:3*width//4] = right_eye_region
    
    return image

def apply_head_movement(image, frame_num, total_frames):
    """고개 움직임 효과"""
    height, width = image.shape[:2]
    
    # 고개 움직임 주기
    head_cycle = math.sin(frame_num * 0.1) * 0.02  # 2% 좌우 이동
    
    # 미세한 좌우 이동
    shift_x = int(head_cycle * width)
    
    # 이미지 이동
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, 0]])
    shifted_image = cv2.warpAffine(image, translation_matrix, (width, height))
    
    return shifted_image

def apply_fur_movement(image, frame_num, total_frames):
    """털 움직임 효과"""
    height, width = image.shape[:2]
    
    # 털 움직임을 위한 워핑 효과
    rows, cols = image.shape[:2]
    
    # 바람 효과
    wind_strength = 0.3
    wind_frequency = 1.5
    
    # 워핑 맵 생성
    map_x = np.zeros((rows, cols), np.float32)
    map_y = np.zeros((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            # 바람에 의한 털 움직임
            offset_x = wind_strength * math.sin(i * 0.05 + frame_num * 0.1)
            offset_y = wind_strength * math.cos(j * 0.05 + frame_num * 0.1)
            
            map_x[i, j] = j + offset_x
            map_y[i, j] = i + offset_y
    
    # 워핑 적용
    warped_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    
    return warped_image

def apply_natural_movement(image, frame_num, total_frames):
    """자연스러운 움직임 조합"""
    # 호흡 효과
    image = apply_breathing_effect(image, frame_num, total_frames)
    
    # 고개 움직임
    image = apply_head_movement(image, frame_num, total_frames)
    
    # 눈 깜빡임
    image = apply_eye_blinking(image, frame_num, total_frames)
    
    # 귀 움직임
    image = apply_ear_movement(image, frame_num, total_frames)
    
    # 꼬리 흔들기
    image = apply_tail_wagging(image, frame_num, total_frames)
    
    # 털 움직임
    image = apply_fur_movement(image, frame_num, total_frames)
    
    return image

def create_video_from_image(image_path, output_path, duration=5, fps=30, effect='natural'):
    """이미지를 동영상으로 변환 (메모리 최적화 버전)"""
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지를 로드할 수 없습니다.")
    
    # 이미지 크기 조정 (매우 작게)
    max_size = 200  # 400에서 200으로 더 줄임
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    height, width = image.shape[:2]
    
    # FPS와 duration 제한 (매우 빠른 처리)
    fps = min(fps, 10)  # 최대 10fps
    duration = min(duration, 2)  # 최대 2초
    total_frames = duration * fps
    
    # 비디오 작성자 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        for frame_num in range(total_frames):
            # 메모리 절약을 위해 copy() 대신 직접 수정
            current_frame = image.copy()
            
            # 가장 간단한 효과만 적용
            if effect == 'natural':
                # 미세한 호흡 효과만
                breathing_cycle = math.sin(frame_num * 2 * math.pi / (fps * 2)) * 0.01
                scale_factor = 1.0 + breathing_cycle
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                resized = cv2.resize(current_frame, (new_width, new_height))
                start_x = (new_width - width) // 2
                start_y = (new_height - height) // 2
                current_frame = resized[start_y:start_y+height, start_x:start_x+width]
            elif effect == 'breathing':
                breathing_cycle = math.sin(frame_num * 2 * math.pi / (fps * 2)) * 0.01
                scale_factor = 1.0 + breathing_cycle
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                resized = cv2.resize(current_frame, (new_width, new_height))
                start_x = (new_width - width) // 2
                start_y = (new_height - height) // 2
                current_frame = resized[start_y:start_y+height, start_x:start_x+width]
            elif effect == 'zoom_in':
                # 줌인 효과 (점점 확대)
                zoom_factor = 1.0 + (frame_num / total_frames) * 0.5  # 50% 확대
                new_width = int(width * zoom_factor)
                new_height = int(height * zoom_factor)
                resized = cv2.resize(current_frame, (new_width, new_height))
                start_x = (new_width - width) // 2
                start_y = (new_height - height) // 2
                current_frame = resized[start_y:start_y+height, start_x:start_x+width]
            elif effect == 'zoom_out':
                # 줌아웃 효과 (점점 축소)
                zoom_factor = 1.5 - (frame_num / total_frames) * 0.5  # 1.5배에서 1.0배로
                new_width = int(width * zoom_factor)
                new_height = int(height * zoom_factor)
                resized = cv2.resize(current_frame, (new_width, new_height))
                start_x = (new_width - width) // 2
                start_y = (new_height - height) // 2
                current_frame = resized[start_y:start_y+height, start_x:start_x+width]
            elif effect == 'zoom_pulse':
                # 줌 펄스 효과 (확대-축소 반복)
                pulse_cycle = math.sin(frame_num * 2 * math.pi / (fps * 1.5)) * 0.2  # 1.5초 주기
                zoom_factor = 1.0 + pulse_cycle
                new_width = int(width * zoom_factor)
                new_height = int(height * zoom_factor)
                resized = cv2.resize(current_frame, (new_width, new_height))
                start_x = (new_width - width) // 2
                start_y = (new_height - height) // 2
                current_frame = resized[start_y:start_y+height, start_x:start_x+width]
            
            # 프레임을 비디오에 추가
            out.write(current_frame)
            
            # 메모리 정리
            del current_frame
    
    finally:
        out.release()
    
    return output_path

@app.route('/')
def index():
    return render_template('animal_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
    
    # 설정 가져오기 (매우 제한된 값으로)
    duration = min(int(request.form.get('duration', 2)), 2)  # 최대 2초
    fps = min(int(request.form.get('fps', 10)), 10)  # 최대 10fps
    effect = request.form.get('effect', 'natural')
    
    # 파일 저장
    filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # 비디오 생성
        output_filename = str(uuid.uuid4()) + '.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        result_path = create_video_from_image(filepath, output_path, duration, fps, effect)
        
        # 결과 파일명 추출
        result_filename = os.path.basename(result_path)
        
        return jsonify({
            'success': True,
            'filename': result_filename,
            'download_url': url_for('download_file', filename=result_filename)
        })
    
    except Exception as e:
        return jsonify({'error': f'비디오 생성 중 오류가 발생했습니다: {str(e)}'}), 500
    
    finally:
        # 업로드된 원본 파일 삭제
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port) 