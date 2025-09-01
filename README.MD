# Buat environment bernama 'venv'
python -m venv venv

# Aktifkan environment
.\venv\Scripts\activate

# Instal Package
pip install -r requirements.txt

# Output Normal
python main.py 1.wav

# Output Custom Name
python main.py song.wav -o cool_video.mp4
