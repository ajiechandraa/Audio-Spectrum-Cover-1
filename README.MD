# Buat environment bernama 'venv'
python -m venv venv

# Aktifkan environment
.\venv\Scripts\activate

# Instal Package
pip install -r requirements.txt

# Output Normal
python main.py song.wav

# Output Custom Name
python main.py song.wav -o cool_video.mp4

# Output 4K 60fps
python main.py song.wav --quality 4k --fps 60

