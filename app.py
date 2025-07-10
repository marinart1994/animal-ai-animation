import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from flask import Flask, request, render_template, send_file, jsonify
import subprocess
import tempfile
import uuid
import math
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_faces(image):
    """간단한 얼굴 감지 (OpenCV Haar Cascade 사용)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def apply_blinking_effect(image, frame_num, total_frames, face_region):
    """눈 깜빡임 효과"""
    x, y, w, h = face_region
    eye_region = image[y:y+h//2, x:x+w]
    
    # 눈 영역을 어둡게 만들어 깜빡임 효과
    blink_frequency = 3  # 3초마다 깜빡임
    blink_duration = 0.2  # 깜빡임 지속시간 (초)
    
    current_time = frame_num / 30.0  # 30fps 가정
    blink_cycle = math.sin(current_time * 2 * math.pi / blink_frequency)
    
    if blink_cycle < 0.1:  # 깜빡임 구간
        # 눈 영역을 어둡게
        darken_factor = 0.3
        eye_region = (eye_region * darken_factor).astype(np.uint8)
        image[y:y+h//2, x:x+w] = eye_region
    
    return image

def apply_breathing_effect(image, frame_num, total_frames):
    """호흡 효과 - 미세한 확대/축소"""
    height, width = image.shape[:2]
    
    # 호흡 주기 (약 4초)
    breathing_cycle = math.sin(frame_num * 2 * math.pi / (30 * 4))
    scale_factor = 1.0 + breathing_cycle * 0.02  # 2% 변화
    
    # 중앙 기준으로 확대/축소
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    resized = cv2.resize(image, (new_width, new_height))
    
    # 중앙에서 원본 크기만큼 크롭
    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2
    cropped = resized[start_y:start_y+height, start_x:start_x+width]
    
    return cropped

def apply_hair_movement(image, frame_num, total_frames, face_region):
    """머리카락 움직임 효과"""
    x, y, w, h = face_region
    
    # 머리 영역 (얼굴 위쪽)
    hair_region = image[max(0, y-h//2):y, x:x+w]
    
    if hair_region.size > 0:
        # 미세한 워핑 효과
        rows, cols = hair_region.shape[:2]
        
        # 바람 효과를 위한 워핑
        wind_strength = 0.5
        wind_frequency = 2.0
        
        for i in range(rows):
            for j in range(cols):
                offset_x = int(wind_strength * math.sin(i * 0.1 + frame_num * 0.1))
                offset_y = int(wind_strength * math.cos(j * 0.1 + frame_num * 0.1))
                
                new_i = max(0, min(rows-1, i + offset_y))
                new_j = max(0, min(cols-1, j + offset_x))
                
                hair_region[i, j] = hair_region[new_i, new_j]
    
    return image

def apply_smile_effect(image, frame_num, total_frames, face_region):
    """미소 효과 - 입 주변 미세한 변화"""
    x, y, w, h = face_region
    
    # 입 영역 (얼굴 아래쪽)
    mouth_region = image[y+h//2:y+h, x:x+w]
    
    if mouth_region.size > 0:
        # 미소 주기 (약 6초)
        smile_cycle = math.sin(frame_num * 2 * math.pi / (30 * 6))
        
        if smile_cycle > 0.5:  # 미소 구간
            # 입 주변을 밝게
            brightness_factor = 1.1
            mouth_region = np.clip(mouth_region * brightness_factor, 0, 255).astype(np.uint8)
    
    return image

def apply_cinematic_camera(image, frame_num, total_frames):
    """시네마틱 카메라 모션"""
    height, width = image.shape[:2]
    
    # 천천히 확대하는 줌 효과
    zoom_factor = 1.0 + (frame_num / total_frames) * 0.1  # 10% 확대
    
    # 미세한 팬 효과
    pan_x = math.sin(frame_num * 0.02) * width * 0.02  # 2% 좌우 이동
    pan_y = math.cos(frame_num * 0.015) * height * 0.01  # 1% 상하 이동
    
    # 확대된 이미지 생성
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    resized = cv2.resize(image, (new_width, new_height))
    
    # 팬 효과 적용
    start_x = int((new_width - width) // 2 + pan_x)
    start_y = int((new_height - height) // 2 + pan_y)
    
    # 경계 확인
    start_x = max(0, min(new_width - width, start_x))
    start_y = max(0, min(new_height - height, start_y))
    
    cropped = resized[start_y:start_y+height, start_x:start_x+width]
    
    return cropped

def apply_animal_effects(image, frame_num, total_frames):
    """동물 전용 효과"""
    height, width = image.shape[:2]
    
    # 동물 호흡 효과 (더 큰 변화)
    breathing_cycle = math.sin(frame_num * 2 * math.pi / (30 * 3))
    scale_factor = 1.0 + breathing_cycle * 0.03  # 3% 변화
    
    # 중앙 기준으로 확대/축소
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    resized = cv2.resize(image, (new_width, new_height))
    
    # 중앙에서 원본 크기만큼 크롭
    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2
    cropped = resized[start_y:start_y+height, start_x:start_x+width]
    
    return cropped

def get_face_landmarks(image):
    """OpenCV를 이용한 간단한 얼굴 감지"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    landmarks_list = []
    for (x, y, w, h) in faces:
        # 간단한 랜드마크 생성 (사각형 기반)
        landmarks = np.array([
            [x + w//2, y + h//3],      # 왼쪽 눈
            [x + w//2, y + h//3],      # 오른쪽 눈
            [x + w//2, y + 2*h//3],    # 코
            [x + w//2, y + 5*h//6],    # 입
        ])
        landmarks_list.append(landmarks)
    
    return landmarks_list

def animate_face_landmarks(image, frame_num, total_frames, landmarks):
    """OpenCV 기반 간단한 얼굴 애니메이션"""
    img = image.copy()
    
    if len(landmarks) >= 4:
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        mouth = landmarks[3]
        
        # 1. 눈 깜빡임 (3초 주기)
        blink_period = 90  # 30fps 기준 3초
        blink_frame = frame_num % blink_period
        if 5 < blink_frame < 15:
            # 눈 영역을 어둡게
            cv2.circle(img, (int(left_eye[0]), int(left_eye[1])), 10, (30,30,30), -1)
            cv2.circle(img, (int(right_eye[0]), int(right_eye[1])), 10, (30,30,30), -1)
        
        # 2. 미소 (6초 주기)
        smile_period = 180
        smile_frame = frame_num % smile_period
        smile_ratio = max(0, np.sin(np.pi * smile_frame / smile_period))
        if smile_ratio > 0.5:
            # 입 주변에 미소 효과
            cv2.circle(img, (int(mouth[0]), int(mouth[1])), 15, (255,180,180), 2)
        
        # 3. 고개 미세 회전 (좌우 5도)
        center = np.mean(landmarks, axis=0)
        angle = np.sin(2 * np.pi * frame_num / total_frames) * 5
        M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    
    return img

def create_video_from_image(image_path, output_path, duration=5, fps=30, effect='ai_animate'):
    """
    이미지를 영상으로 변환하는 함수 (AI 애니메이션 포함)
    
    Args:
        image_path: 입력 이미지 경로
        output_path: 출력 비디오 경로
        duration: 비디오 길이 (초)
        fps: 초당 프레임 수
        effect: 적용할 효과
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지를 로드할 수 없습니다.")
    
    height, width = img.shape[:2]
    total_frames = duration * fps
    
    # 비디오 작성자 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if effect == 'ai_animate':
        # AI 기반 자연스러운 애니메이션
        landmarks_list = get_face_landmarks(img)
        
        for i in range(total_frames):
            frame = img.copy()
            
            if landmarks_list:
                for landmarks in landmarks_list:
                    frame = animate_face_landmarks(frame, i, total_frames, landmarks)
            else:
                frame = apply_animal_effects(frame, i, total_frames)
            
            # 시네마틱 카메라 모션
            frame = apply_cinematic_camera(frame, i, total_frames)
            
            out.write(frame)
    
    elif effect == 'zoom':
        # 줌 효과: 천천히 확대
        for i in range(total_frames):
            scale = 1.0 + (i / total_frames) * 0.3  # 30% 확대
            scaled_width = int(width * scale)
            scaled_height = int(height * scale)
            
            # 이미지 리사이즈
            scaled_img = cv2.resize(img, (scaled_width, scaled_height))
            
            # 중앙에서 크롭
            start_x = (scaled_width - width) // 2
            start_y = (scaled_height - height) // 2
            cropped_img = scaled_img[start_y:start_y+height, start_x:start_x+width]
            
            out.write(cropped_img)
    
    elif effect == 'pan':
        # 팬 효과: 좌에서 우로 이동
        for i in range(total_frames):
            offset_x = int((i / total_frames) * (width * 0.3))  # 30% 이동
            offset_y = int((i / total_frames) * (height * 0.2))  # 20% 이동
            
            # 이미지 리사이즈 (약간 확대)
            scale = 1.2
            scaled_width = int(width * scale)
            scaled_height = int(height * scale)
            scaled_img = cv2.resize(img, (scaled_width, scaled_height))
            
            # 이동된 위치에서 크롭
            start_x = offset_x
            start_y = offset_y
            if start_x + width > scaled_width:
                start_x = scaled_width - width
            if start_y + height > scaled_height:
                start_y = scaled_height - height
            
            cropped_img = scaled_img[start_y:start_y+height, start_x:start_x+width]
            out.write(cropped_img)
    
    elif effect == 'fade':
        # 페이드 효과: 페이드 인/아웃
        for i in range(total_frames):
            # 페이드 인 (처음 20%)
            if i < total_frames * 0.2:
                alpha = i / (total_frames * 0.2)
            # 페이드 아웃 (마지막 20%)
            elif i > total_frames * 0.8:
                alpha = (total_frames - i) / (total_frames * 0.2)
            else:
                alpha = 1.0
            
            # 알파 블렌딩
            overlay = img.copy()
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, overlay)
            out.write(overlay)
    
    elif effect == 'rotation':
        # 회전 효과
        for i in range(total_frames):
            angle = (i / total_frames) * 360  # 360도 회전
            
            # 회전 행렬 계산
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 회전된 이미지 생성
            rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
            out.write(rotated)
    
    elif effect == 'blur':
        # 블러 효과
        for i in range(total_frames):
            # 블러 강도 변화
            blur_strength = int(1 + (i / total_frames) * 10)
            if blur_strength % 2 == 0:
                blur_strength += 1  # 홀수로 만들기
            
            blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
            out.write(blurred)
    
    elif effect == 'color_shift':
        # 색상 변화 효과
        for i in range(total_frames):
            # HSV 색상 공간으로 변환
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 색조 변화
            hue_shift = int((i / total_frames) * 30)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            
            # BGR로 다시 변환
            color_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            out.write(color_shifted)
    
    else:  # static
        # 정적 이미지 (단순 반복)
        for _ in range(total_frames):
            out.write(img)
    
    out.release()
    
    # FFmpeg로 최적화 (선택사항)
    optimized_path = output_path.replace('.mp4', '_optimized.mp4')
    try:
        result = subprocess.run([
            'ffmpeg', '-i', output_path,
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-y', optimized_path
        ], check=True, capture_output=True, text=True)
        
        # 최적화된 파일이 성공적으로 생성되었는지 확인
        if os.path.exists(optimized_path) and os.path.getsize(optimized_path) > 0:
            os.remove(output_path)  # 원본 파일 삭제
            return optimized_path
        else:
            return output_path  # 최적화 실패시 원본 반환
            
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 최적화 실패: {e}")
        return output_path  # 최적화 실패시 원본 반환
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        return output_path  # 오류 발생시 원본 반환

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
    
    # 설정 가져오기
    duration = int(request.form.get('duration', 5))
    fps = int(request.form.get('fps', 30))
    effect = request.form.get('effect', 'ai_animate')
    
    # 파일 저장
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # 비디오 생성
        output_filename = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        final_path = create_video_from_image(filepath, output_path, duration, fps, effect)
        
        # 임시 파일 정리
        os.remove(filepath)
        
        # 최종 파일이 실제로 생성되었는지 확인
        if not os.path.exists(final_path):
            return jsonify({'error': '비디오 파일이 생성되지 않았습니다.'}), 500
        
        # 최종 파일명 추출
        final_filename = os.path.basename(final_path)
        
        return jsonify({
            'success': True,
            'video_url': f'/download/{final_filename}',
            'message': '비디오가 성공적으로 생성되었습니다!'
        })
        
    except Exception as e:
        # 에러 발생시 임시 파일 정리
        if os.path.exists(filepath):
            os.remove(filepath)
        print(f"비디오 생성 오류: {str(e)}")
        return jsonify({'error': f'비디오 생성 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=f"video_{filename}"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True) 