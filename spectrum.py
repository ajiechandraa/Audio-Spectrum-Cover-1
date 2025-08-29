# --- START OF FILE spectrum.py ---

# spectrum.py - Core Engine dengan Gradien Bawah dan Spektrum Garis Solid

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
from matplotlib.artist import Artist
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
import librosa
import os
import math
from tqdm import tqdm
from config import SETTINGS
import subprocess
from typing import Sequence, Any
import random # <--- TAMBAHKAN BARIS INI

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class GlassCardSpectrumVisualizer:
    def __init__(self, audio_file, output_file, card_background='card-background.png', 
                 main_background='background.png', metadata=None, custom_config=None):
        print("   Initializing Engine...")
        self.audio_file = audio_file
        self.output_file = output_file
        self.card_bg_path = card_background
        self.main_bg_path = main_background
        self.metadata = metadata or {'title': 'Unknown Title', 'artist': 'Unknown Artist'}
        self.config = SETTINGS.copy()
        if custom_config: self.config.update(custom_config)
        
        self._init_canvas_config()
        self._init_audio()
        self._init_visual_components()
        self._init_animation_state()
        print("   ✅ Engine initialized successfully!")

    def cfg_get(self, key: str, default: Any) -> Any:
        value = self.config.get(key, default)
        return default if value is None else value

    def _init_canvas_config(self):
        quality_map = {'standard':(1280,720), 'hd':(1920,1080), '4k':(3840,2160)}
        quality = self.cfg_get('QUALITY', 'hd')
        self.config['CANVAS_WIDTH'], self.config['CANVAS_HEIGHT'] = quality_map.get(quality, (1920, 1080))

    def _init_audio(self):
        print("   Loading and processing audio data...")
        self.audio, self.sr = librosa.load(self.audio_file, sr=44100, mono=True)
        self.duration = librosa.get_duration(y=self.audio, sr=self.sr)
        fft_size, hop_length = self.cfg_get('FFT_SIZE', 4096), self.cfg_get('HOP_LENGTH', 512)
        self.stft = librosa.stft(y=self.audio, n_fft=fft_size, hop_length=hop_length)
        self.magnitude_db = librosa.amplitude_to_db(np.abs(self.stft), ref=np.max)
        self.times = librosa.frames_to_time(np.arange(self.magnitude_db.shape[1]), sr=self.sr, hop_length=hop_length)
        self.frequencies = librosa.fft_frequencies(sr=self.sr, n_fft=fft_size)
        self._setup_frequency_bands()
        self._compute_bass_envelope()
        print(f"   Audio loaded: {self.duration:.2f} seconds")

    def _setup_frequency_bands(self):
        num_points = self.cfg_get('LINE_SPECTRUM_POINTS', 128)
        min_freq, max_freq = self.cfg_get('MIN_FREQUENCY', 30), self.cfg_get('MAX_FREQUENCY', 16000)
        if self.cfg_get('FREQUENCY_SCALE', 'log') == 'log':
            self.band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), num_points + 1)
        else:
            self.band_edges = np.linspace(min_freq, max_freq, num_points + 1)

    def _compute_bass_envelope(self):
        cutoff = self.cfg_get('BASS_FREQUENCY_CUTOFF', 150)
        bass_idx = np.where(self.frequencies <= cutoff)[0]
        if len(bass_idx) > 0:
            bass_mag = np.mean(self.magnitude_db[bass_idx, :], axis=0)
            bass_mag = (bass_mag - np.min(bass_mag)) / (np.max(bass_mag) - np.min(bass_mag) + 1e-9)
            self.bass_envelope = gaussian_filter1d(bass_mag, sigma=2)
        else:
            self.bass_envelope = np.zeros(len(self.times))

    def _init_visual_components(self):
        print("   Setting up visual components...")
        W, H = self.cfg_get('CANVAS_WIDTH', 1920), self.cfg_get('CANVAS_HEIGHT', 1080)
        self.fig = plt.figure(figsize=(W/100, H/100), dpi=100, facecolor='black')
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, W), ylim=(0, H), aspect='equal')
        self.ax.axis('off')
        
        self._setup_backgrounds()
        self._create_glass_card()
        self._create_line_spectrum()
        self._add_text_elements()

    def _setup_backgrounds(self):
        W, H = self.cfg_get('CANVAS_WIDTH', 1920), self.cfg_get('CANVAS_HEIGHT', 1080)
        zoom = self.cfg_get('BACKGROUND_ZOOM_FACTOR', 1.1)
        
        try:
            img = mpimg.imread(self.main_bg_path)
            extent = (-W*(zoom-1)/2, W*(1+(zoom-1)/2), -H*(zoom-1)/2, H*(1+(zoom-1)/2))
            
            # Latar belakang utama (paling atas)
            self.main_bg_artist = self.ax.imshow(img, extent=extent, aspect='auto', zorder=0)
            
            # Latar belakang untuk motion blur (di bawah utama)
            if self.cfg_get('ENABLE_MOTION_BLUR', True):
                self.blur_bg_artist = self.ax.imshow(img, extent=extent, aspect='auto', zorder=-1, alpha=0)

        except (FileNotFoundError, IOError):
            print(f"   ⚠️ Warning: Latar belakang utama '{self.main_bg_path}' tidak ditemukan.")
            self.ax.add_patch(Rectangle((0, 0), W, H, facecolor='#101018', zorder=0))

        overlay_alpha = self.cfg_get('BACKGROUND_OVERLAY_ALPHA', 0.5)
        if overlay_alpha > 0:
            self.ax.add_patch(Rectangle((0, 0), W, H, facecolor='black', alpha=overlay_alpha, zorder=1))

    def _create_glass_card(self):
        W, H = self.cfg_get('CANVAS_WIDTH', 1920), self.cfg_get('CANVAS_HEIGHT', 1080)
        cw, ch = W * self.cfg_get('CARD_WIDTH_RATIO', 0.3), H * self.cfg_get('CARD_HEIGHT_RATIO', 0.7)
        cx, cy = (W - cw) / 2, (H - ch) * self.cfg_get('CARD_POSITION_Y', 0.5)
        radius = self.cfg_get('CARD_CORNER_RADIUS', 20)
        self.card_bounds = {'x': cx, 'y': cy, 'width': cw, 'height': ch}
        
        self.card_clip_patch = FancyBboxPatch((cx, cy), cw, ch, boxstyle=f"round,pad=0,rounding_size={radius}", ec='none', fc='none', zorder=2)
        self.ax.add_patch(self.card_clip_patch)
        
        try:
            card_bg_img = mpimg.imread(self.card_bg_path)
            img_h, img_w, _ = card_bg_img.shape
            card_aspect, img_aspect = cw / ch, img_w / img_h
            new_w, new_h = (cw, cw/img_aspect) if card_aspect > img_aspect else (ch*img_aspect, ch)
            extent = (cx-(new_w-cw)/2, cx+cw+(new_w-cw)/2, cy-(new_h-ch)/2, cy+ch+(new_h-ch)/2)
            self.card_bg_artist = self.ax.imshow(card_bg_img, extent=extent, zorder=3)
            self.card_bg_artist.set_clip_path(self.card_clip_patch)
        except (FileNotFoundError, IOError):
            print(f"   ⚠️ Warning: Latar belakang kartu '{self.card_bg_path}' tidak ditemukan.")
            
        if self.cfg_get('ENABLE_BOTTOM_GRADIENT', True):
            grad_height = ch * self.cfg_get('GRADIENT_HEIGHT_RATIO', 0.4)
            gradient = np.array([[0,0], [1,1]])
            cmap = LinearSegmentedColormap.from_list('bottom_grad', 
                [self.cfg_get('GRADIENT_COLOR_BOTTOM', '#000000c0'), self.cfg_get('GRADIENT_COLOR_TOP', '#00000000')])
            grad_artist = self.ax.imshow(gradient, cmap=cmap, interpolation='bicubic', aspect='auto',
                                         extent=(cx, cx + cw, cy, cy + grad_height), zorder=4)
            grad_artist.set_clip_path(self.card_clip_patch)
        
        border = FancyBboxPatch((cx, cy), cw, ch, boxstyle=f"round,pad=0,rounding_size={radius}", lw=self.cfg_get('CARD_BORDER_WIDTH', 1.5), ec=self.cfg_get('CARD_BORDER_COLOR', '#ffffff20'), fc='none', zorder=5)
        self.ax.add_patch(border)
        
        if self.cfg_get('ENABLE_BORDER_GLOW', True):
            self.border_glow = FancyBboxPatch((cx, cy), cw, ch, boxstyle=f"round,pad=0,rounding_size={radius}", lw=self.cfg_get('BORDER_GLOW_WIDTH', 4), ec=self.cfg_get('BORDER_GLOW_COLOR', '#ffffff'), fc='none', zorder=5, alpha=0)
            self.ax.add_patch(self.border_glow)

    def _create_line_spectrum(self):
        cb = self.card_bounds
        n_points = self.cfg_get('LINE_SPECTRUM_POINTS', 128)
        width = self.cfg_get('LINE_SPECTRUM_WIDTH', 2.0)
        margin_x = cb['width'] * self.cfg_get('LINE_SPECTRUM_MARGIN_X', 0.08)
        self.spectrum_x = np.linspace(cb['x'] + margin_x, cb['x'] + cb['width'] - margin_x, n_points)
        y_pos = cb['y'] + cb['height'] * self.cfg_get('LINE_SPECTRUM_POSITION_Y', 0.22)
        
        self.line_spectrum, = self.ax.plot(self.spectrum_x, np.zeros(n_points)+y_pos, color=self.cfg_get('LINE_SPECTRUM_COLOR', '#ffffff'), lw=width, zorder=4)
        self.line_spectrum.set_clip_path(self.card_clip_patch)

    def _add_text_elements(self):
        cb = self.card_bounds
        margin_x = cb['width'] * self.cfg_get('TEXT_MARGIN_X', 0.08)
        text_x_pos = cb['x'] + margin_x
        
        if self.cfg_get('SHOW_ARTIST', True):
            artist_y_pos = cb['y'] + cb['height'] * self.cfg_get('ARTIST_POSITION_Y', 0.08)
            self.artist_text = self.ax.text(text_x_pos, artist_y_pos, self.metadata['artist'], ha='left', va='center', color=self.cfg_get('ARTIST_COLOR', '#ffffffa0'), fontsize=self.cfg_get('ARTIST_FONT_SIZE', 18), zorder=4)
            self.artist_text.set_clip_path(self.card_clip_patch)

        if self.cfg_get('SHOW_TITLE', True):
            title_y_pos = cb['y'] + cb['height'] * self.cfg_get('TITLE_POSITION_Y', 0.15)
            self.title_text = self.ax.text(text_x_pos, title_y_pos, self.metadata['title'], ha='left', va='center', color=self.cfg_get('TITLE_COLOR', '#ffffff'), fontsize=self.cfg_get('TITLE_FONT_SIZE', 36), weight=self.cfg_get('TITLE_FONT_WEIGHT', 'bold'), zorder=4)
            self.title_text.set_clip_path(self.card_clip_patch)

    def _init_animation_state(self):
        self.spectrum_heights = np.zeros(self.cfg_get('LINE_SPECTRUM_POINTS', 128))
        self.shake_magnitude = 0.0
        self.border_glow_alpha = 0.0
        self.last_shake_offset = (0, 0) # <--- BARU: Untuk menyimpan posisi getaran sebelumnya

    def _update_frame(self, frame_num) -> Sequence[Artist]:
        t = frame_num / self.cfg_get('FPS', 60)
        time_idx = np.searchsorted(self.times, t)
        if time_idx >= len(self.times): time_idx = -1
        
        target_heights = self._get_heights_at_time(time_idx)
        smoothing = self.cfg_get('TEMPORAL_SMOOTHING', 0.75)
        self.spectrum_heights = self.spectrum_heights * smoothing + target_heights * (1 - smoothing)
        self._update_line_spectrum()

        bass_val = self.bass_envelope[time_idx] if 0 <= time_idx < len(self.bass_envelope) else 0
        self._update_bass_effects(bass_val, t)

        return self._get_updated_artists()

    def _get_heights_at_time(self, time_idx):
        n_points = self.cfg_get('LINE_SPECTRUM_POINTS', 128)
        heights = np.zeros(n_points)
        for i in range(n_points):
            start_f, end_f = self.band_edges[i], self.band_edges[i+1]
            idx_s, idx_e = np.searchsorted(self.frequencies, [start_f, end_f])
            val = np.mean(self.magnitude_db[idx_s:idx_e, time_idx]) if idx_s < idx_e else -80.0
            heights[i] = (np.clip(val, -60, 0) + 60) / 60
        return heights

    def _update_line_spectrum(self):
        cb = self.card_bounds
        max_h = cb['height'] * self.cfg_get('LINE_SPECTRUM_HEIGHT_RATIO', 0.1)
        base_y = cb['y'] + cb['height'] * self.cfg_get('LINE_SPECTRUM_POSITION_Y', 0.22)
        spatial_smoothing = self.cfg_get('LINE_SPECTRUM_SPATIAL_SMOOTHING', 3.0)
        smoothed_heights = gaussian_filter1d(self.spectrum_heights, sigma=spatial_smoothing)
        y_data = base_y + smoothed_heights * max_h
        self.line_spectrum.set_ydata(y_data)
    
    def _update_bass_effects(self, bass_val, t):
        # Memicu efek saat bass terdeteksi
        if bass_val > self.cfg_get('BASS_THRESHOLD', 0.6):
            if self.cfg_get('ENABLE_BASS_SHAKE', True):
                self.shake_magnitude = self.cfg_get('SHAKE_INTENSITY', 15)
            if self.cfg_get('ENABLE_BORDER_GLOW', True):
                self.border_glow_alpha = 1.0
        
        # Meredam magnitudo efek seiring waktu
        self.shake_magnitude *= self.cfg_get('SHAKE_DECAY', 0.75)
        self.border_glow_alpha *= self.cfg_get('BORDER_GLOW_DECAY', 0.90)
        
        # Logika untuk getaran latar belakang
        if self.cfg_get('ENABLE_BASS_SHAKE', True) and hasattr(self, 'main_bg_artist'):
            offset_x, offset_y = 0, 0
            shake_style = self.cfg_get('SHAKE_STYLE', 'jolt')

            if self.shake_magnitude > 0.1: # Hanya goyang jika ada energi
                if shake_style == 'jolt':
                    # Gaya JOLT: Hentakan acak, terasa lebih seperti getaran
                    offset_x = random.uniform(-self.shake_magnitude, self.shake_magnitude)
                    offset_y = random.uniform(-self.shake_magnitude, self.shake_magnitude)
                else: # 'smooth' atau default
                    # Gaya SMOOTH: Gerakan sinusoidal yang lama
                    freq_x = self.cfg_get('SHAKE_FREQUENCY_X', 25.0) * 2 * math.pi
                    freq_y = self.cfg_get('SHAKE_FREQUENCY_Y', 30.0) * 2 * math.pi
                    offset_x = self.shake_magnitude * math.sin(t * freq_x)
                    offset_y = self.shake_magnitude * math.cos(t * freq_y)

            # Terapkan motion blur jika diaktifkan
            enable_blur = self.cfg_get('ENABLE_MOTION_BLUR', True)
            if enable_blur and hasattr(self, 'blur_bg_artist'):
                # Gunakan offset DARI FRAME SEBELUMNYA untuk 'jejak' blur
                prev_x, prev_y = self.last_shake_offset
                blur_transform = transforms.Affine2D().translate(prev_x, prev_y)
                self.blur_bg_artist.set_transform(self.ax.transData + blur_transform)
                
                # Atur transparansi blur berdasarkan intensitas getaran
                blur_alpha = self.cfg_get('MOTION_BLUR_ALPHA', 0.4)
                blur_intensity = min(1.0, self.shake_magnitude / self.cfg_get('SHAKE_INTENSITY', 15))
                self.blur_bg_artist.set_alpha(blur_intensity * blur_alpha)

            # Terapkan transformasi ke latar belakang utama
            transform = transforms.Affine2D().translate(offset_x, offset_y)
            self.main_bg_artist.set_transform(self.ax.transData + transform)
            
            # Simpan offset saat ini untuk digunakan di frame berikutnya (untuk blur)
            self.last_shake_offset = (offset_x, offset_y)

        # Logika untuk glow pada border
        if self.cfg_get('ENABLE_BORDER_GLOW', True):
            self.border_glow.set_alpha(self.border_glow_alpha)

    def _get_updated_artists(self) -> Sequence[Artist]:
        artists: list[Artist] = [self.line_spectrum]
        if hasattr(self, 'main_bg_artist'):
            artists.append(self.main_bg_artist)
        if hasattr(self, 'blur_bg_artist') and self.cfg_get('ENABLE_MOTION_BLUR', True):
            artists.append(self.blur_bg_artist) # <--- TAMBAHKAN ARTIST BLUR
        if hasattr(self, 'border_glow') and self.cfg_get('ENABLE_BORDER_GLOW', True):
            artists.append(self.border_glow)
        return artists

    def create_video(self):
        temp_video_file = self.output_file + ".temp.mp4"
        total_frames = int(self.duration * self.cfg_get('FPS', 60))
        
        if os.path.exists(temp_video_file):
            os.remove(temp_video_file)
            
        ani = animation.FuncAnimation(self.fig, self._update_frame, frames=total_frames, interval=1000/self.cfg_get('FPS', 60), blit=True)
        
        try:
            print("   1/2: Rendering Video Frames (using CPU)...")
            writer = animation.FFMpegWriter(
                fps=self.cfg_get('FPS', 60),
                codec=self.cfg_get('VIDEO_CODEC', 'libx264'),
                bitrate=self.cfg_get('BITRATE', 8000),
                extra_args=['-pix_fmt', 'yuv420p']
            )
            with tqdm(total=total_frames, desc="   CPU Render", unit="frame") as pbar:
                ani.save(temp_video_file, writer=writer, progress_callback=lambda i, n: pbar.update(1))
            
            print("   ✅ Video frames rendered successfully.")
            print("   2/2: Combining video and audio...")
            
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', temp_video_file,
                '-i', self.audio_file,
                '-c:v', 'copy',
                '-c:a', self.cfg_get('AUDIO_CODEC', 'aac'),
                '-b:a', self.cfg_get('AUDIO_BITRATE', '320k'),
                '-shortest', '-y', self.output_file
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True

        except (Exception, KeyboardInterrupt) as e:
            print(f"\n❌ An error occurred during video creation: {e}")
            if isinstance(e, subprocess.CalledProcessError):
                print("   FFmpeg Error Output:", e.stderr)
            return False
            
        finally:
            if os.path.exists(temp_video_file):
                os.remove(temp_video_file)
# --- END OF FILE spectrum.py ---