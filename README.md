# 🎮 Wordfeusk - Wordfeud Board Analyzer 🔍

🚀 A powerful FastAPI-based service for analyzing Wordfeud board screenshots using computer vision and OCR techniques. This project helps players analyze their Wordfeud game state through screenshot processing and provides a web interface for easy interaction.

## ✨ Features

- 📸 Screenshot analysis with OpenCV and Tesseract OCR
- 🎯 Board state recognition and visualization
- 🎲 Rack letter detection
- 🌐 Modern web interface for uploads
- ⚡ Async processing for long-running tasks
- 🔄 Real-time status updates

## 📋 Prerequisites

### 💻 System Requirements

- 🐍 Python 3.10 or higher
- 🔠 Tesseract OCR

#### 🛠️ Installing Tesseract

##### 🍎 macOS
```bash
brew install tesseract
```

For other platforms, see [Tesseract documentation](https://github.com/tesseract-ocr/tesseract).

### 📦 Python Dependencies

```
numpy==1.24.3
opencv-python==4.8.1.78
packaging==24.2
pillow==11.0.0
pytesseract>=0.3.10
matplotlib==3.8.2
fastapi==0.109.1
python-multipart==0.0.6
uvicorn==0.27.0
jinja2==3.1.3
```

## 🚀 Installation

1. 📥 Clone the repository:
```bash
git clone https://github.com/martinkallstrom/wordfeud-player.git
cd wordfeud-player
```

2. 🌍 Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. ⚡ Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### 🚀 Starting the Server

Run the FastAPI server:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 🌐 Accessing the Service

- 🖥️ Web Interface: http://localhost:8000
- 📚 API Documentation: http://localhost:8000/docs

### 🔌 API Endpoints

#### 📤 POST /analyze
Upload and analyze a Wordfeud board screenshot:
```bash
curl -X POST "http://localhost:8000/analyze" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@your_screenshot.png"
```

#### 📥 GET /status/{task_id}
Check the status of an analysis task:
```bash
curl -X GET "http://localhost:8000/status/task_1" -H "accept: application/json"
```

## 📂 Project Structure

- 🔍 `ocr/`: OCR-related code and template matching
- 🎨 `visualization/`: Board and match visualization tools
- 🧪 `test/`: Test suite
- 📝 `letter_templates/`: Reference images for letter recognition
- 🗂️ `static/`: Generated visualizations and static assets
- 🎯 `templates/`: HTML templates for web interface
- 🛠️ `utils/`: Shared utility functions

## 👩‍💻 Development

### 🔄 Git Workflow

Two helper scripts are provided for git operations:

1. 📤 Push changes to your repository:
```bash
./push_git.sh "Your commit message"
```

2. 📥 Pull updates from upstream:
```bash
./pull_upstream.sh
```

## 🔧 Technical Details

### 🧩 Core Components

1. **🔍 OCR Engine**
   - 📸 OpenCV for image processing
   - 🔠 Tesseract OCR for text recognition
   - 🎯 Custom template matching for letter detection

2. **🎮 Game Logic**
   - 🎲 Board state representation
   - ✅ Move validation
   - 📊 Game state analysis

3. **🎨 Visualization**
   - 🖼️ Board state rendering
   - 🔄 Match visualization
   - 🎯 Template matching results

4. **⚡ API Service**
   - 🔄 Async processing
   - 🔙 Background tasks
   - 📊 Status tracking
   - 🛡️ Error handling

## 🤝 Contributing

1. 🔱 Fork the repository
2. 🌿 Create your feature branch
3. 💾 Commit your changes
4. 📤 Push to your branch
5. 🎯 Create a Pull Request

## 📜 License

This project is based on the work by [mrcz](https://github.com/mrcz/Wordfeud-Player) and maintains the same license terms.

## 🙏 Acknowledgments

- 🎮 Original project by [mrcz](https://github.com/mrcz/Wordfeud-Player)
- 🔍 OpenCV and Tesseract OCR communities
- ⚡ FastAPI framework developers
