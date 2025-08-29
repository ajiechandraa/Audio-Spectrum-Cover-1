# --- START OF FILE config.py ---

# config.py - Konfigurasi dengan Gradien Bawah dan Spektrum Garis Solid

SETTINGS = {
    # ==================== KANVAS & KUALITAS VIDEO ====================
    'CANVAS_WIDTH': 1920,
    'CANVAS_HEIGHT': 1080,
    'FPS': 60,
    'QUALITY': 'hd',
    'BITRATE': 12000,
    
    # ==================== PENGATURAN RENDERER (BARU) ====================
    'RENDERER': 'auto',             # Pilihan: 'auto', 'gpu', 'cpu'
    'VIDEO_CODEC_CPU': 'libx264',   # Codec untuk CPU
    'VIDEO_CODEC_GPU': 'h264_nvenc',# Codec untuk NVIDIA GPU (pastikan driver terinstall)

    # ==================== PENGATURAN KARTU ====================
    'CARD_WIDTH_RATIO': 0.3,
    'CARD_HEIGHT_RATIO': 0.7,
    'CARD_CORNER_RADIUS': 20,
    'CARD_POSITION_Y': 0.5,
    'CARD_BORDER_WIDTH': 1.5,
    'CARD_BORDER_COLOR': '#ffffff20',
    'CARD_SHADOW_BLUR': 50,
    'CARD_SHADOW_COLOR': '#00000090',
    'CARD_SHADOW_OFFSET_Y': 15,
    
    # ==================== GRADIENT OVERLAY HITAM ====================
    'ENABLE_BOTTOM_GRADIENT': True,
    'GRADIENT_HEIGHT_RATIO': 0.4,
    'GRADIENT_COLOR_BOTTOM': '#00000000',
    'GRADIENT_COLOR_TOP': '#000000c0',
    
    # ==================== EFEK BORDER GLOW ====================
    'ENABLE_BORDER_GLOW': True,
    'BORDER_GLOW_WIDTH': 4,
    'BORDER_GLOW_COLOR': '#ffffff',
    'BORDER_GLOW_DECAY': 0.90,
    
    # ==================== SPEKTRUM GARIS (Gaya Solid) ====================
    'LINE_SPECTRUM_POINTS': 128,
    'LINE_SPECTRUM_HEIGHT_RATIO': 0.1,
    'LINE_SPECTRUM_POSITION_Y': 0.22,
    'LINE_SPECTRUM_MARGIN_X': 0.08,
    'LINE_SPECTRUM_COLOR': '#ffffff',
    'LINE_SPECTRUM_WIDTH': 3.0,
    'LINE_SPECTRUM_SPATIAL_SMOOTHING': 3.0,
    
    # ==================== LATAR BELAKANG & OVERLAY ====================
    'BACKGROUND_TYPE': 'image',
    'BACKGROUND_OVERLAY_ALPHA': 0.5,
    'BACKGROUND_ZOOM_FACTOR': 1.1,
    'VIGNETTE_INTENSITY': 0.3,
    
    # ==================== ANALISIS FREKUENSI ====================
    'MIN_FREQUENCY': 30,
    'MAX_FREQUENCY': 16000,
    'FREQUENCY_SCALE': 'log',
    'FFT_SIZE': 4096,
    'HOP_LENGTH': 512,
    
    # ==================== EFEK & ANIMASI ====================
    'TEMPORAL_SMOOTHING': 0.3,
    'ENABLE_BASS_SHAKE': True,
    'SHAKE_INTENSITY': 5,
    'SHAKE_DECAY': 0.75,
    'SHAKE_STYLE': 'jolt',
    'SHAKE_FREQUENCY_X': 25.0,
    'SHAKE_FREQUENCY_Y': 30.0,
    'ENABLE_MOTION_BLUR': True,
    'MOTION_BLUR_ALPHA': 0.4,
    'ENABLE_CARD_PULSE': False,
    
    # Deteksi Bass
    'BASS_FREQUENCY_CUTOFF': 150,
    'BASS_THRESHOLD': 0.6,
    
    # ==================== TEKS & METADATA ====================
    'TEXT_MARGIN_X': 0.08,
    'SHOW_ARTIST': True,
    'ARTIST_FONT_SIZE': 18,
    'ARTIST_COLOR': '#ffffffa0',
    'ARTIST_POSITION_Y': 0.08,
    'SHOW_TITLE': True,
    'TITLE_FONT_SIZE': 36,
    'TITLE_COLOR': '#ffffff',
    'TITLE_POSITION_Y': 0.15,
    'TITLE_FONT_WEIGHT': 'bold',
    'SHOW_FEATURE_TEXT': False,
    
    # ==================== PENGATURAN LAIN-LAIN ====================
    'ENABLE_PARTICLES': False,
    'AUDIO_CODEC': 'aac',
    'AUDIO_BITRATE': '320k'
}
# --- END OF FILE config.py ---