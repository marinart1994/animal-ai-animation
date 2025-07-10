# 🐾 AI 동물 애니메이션 변환기

정적인 동물 사진을 AI로 자연스럽게 움직이는 영상으로 변환하는 웹 애플리케이션입니다.

## ✨ 주요 기능

### 🎬 AI 애니메이션 효과
- **호흡 효과**: 자연스러운 호흡 움직임
- **귀 움직임**: 귀가 살짝 움직이는 효과  
- **눈 깜빡임**: 4초마다 자연스럽게 깜빡임
- **꼬리 흔들기**: 꼬리가 좌우로 흔들리는 효과
- **털 움직임**: 바람에 날리는 털 효과
- **고개 움직임**: 미세한 좌우 움직임

### 🎨 사용자 친화적 인터페이스
- 📸 드래그 앤 드롭 파일 업로드
- 🎬 다양한 애니메이션 효과 선택
- ⏱️ 영상 길이 조절 (3초~10초)
- 🎥 프레임 레이트 선택 (24/30/60 FPS)
- 📱 모바일 반응형 디자인

## 🚀 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/animal-ai-animation.git
cd animal-ai-animation
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 애플리케이션 실행
```bash
python animal_ai_animation.py
```

### 4. 웹 브라우저에서 접속
```
http://localhost:5000
```

## 📋 지원 파일 형식

- PNG
- JPG/JPEG  
- GIF
- BMP
- TIFF

## 🎯 사용 방법

1. **동물 사진 업로드**: 드래그 앤 드롭 또는 클릭하여 선택
2. **애니메이션 설정**: 원하는 효과와 길이 선택
3. **AI 처리**: 30초~2분 정도 소요
4. **영상 다운로드**: 완성된 애니메이션 영상 다운로드

## 💡 사용 팁

- **강아지**: 정면을 바라보는 사진이 가장 좋은 결과
- **고양이**: 귀와 꼬리가 잘 보이는 사진 사용
- **여우**: 털이 잘 보이는 사진이 자연스러운 움직임
- **토끼**: 긴 귀가 잘 보이는 사진 선택

## 🛠️ 기술 스택

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: OpenCV, PIL, scikit-image
- **Animation**: NumPy, Matplotlib
- **Deployment**: Gunicorn, ngrok

## 📁 프로젝트 구조

```
animal-ai-animation/
├── animal_ai_animation.py    # 메인 Flask 애플리케이션
├── requirements.txt          # Python 의존성
├── templates/
│   └── animal_index.html     # 웹 인터페이스
├── uploads/                  # 업로드된 이미지 (자동 생성)
├── outputs/                  # 생성된 영상 (자동 생성)
└── README.md                 # 프로젝트 설명
```

## 🔧 환경 설정

### 포트 충돌 해결 (macOS)
macOS에서 포트 5000이 사용 중인 경우:
1. System Preferences → General → AirDrop & Handoff
2. "AirPlay Receiver" 비활성화
3. 또는 다른 포트 사용: `python animal_ai_animation.py --port 8000`

### 외부 접근 설정
ngrok을 사용하여 외부에서 접근 가능:
```bash
ngrok http 5000
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요.

---

⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요! 