# --- START OF FILE main.py ---

# main.py - Entry point untuk aplikasi visualizer
# Versi ini langsung berjalan tanpa konfirmasi.

import argparse
import sys
import os
import json
from pathlib import Path
from spectrum import GlassCardSpectrumVisualizer

def print_banner():
    """Menampilkan banner aplikasi"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           🎵 AUDIO SPECTRUM VISUALIZER 🎵                   ║
    ║                     Orion Grove                              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def validate_input_file(file_path):
    """Validasi file input audio"""
    if not os.path.exists(file_path):
        return False, f"File tidak ditemukan: {file_path}"
    
    valid_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma']
    if Path(file_path).suffix.lower() not in valid_extensions:
        return False, f"Format file tidak didukung: {Path(file_path).suffix}"
    return True, "OK"

def get_metadata_from_user():
    """Interaktif mendapatkan metadata dari user"""
    print("\n📝 Metadata Lagu (Opsional - tekan Enter untuk skip):")
    title = input("   Judul lagu: ").strip() or "Unknown Title"
    artist = input("   Nama artist: ").strip() or "Unknown Artist"
    return {'title': title, 'artist': artist}

def create_output_filename(input_file, suffix='_visualized'):
    """Generate nama file output otomatis"""
    return f"{Path(input_file).stem}{suffix}.mp4"

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description='Membuat visualisasi spektrum audio.')
    parser.add_argument('input', help='Path ke file audio (mp3, wav, flac, dll.)')
    parser.add_argument('-o', '--output', help='Path file video output (opsional)')
    parser.add_argument('-t', '--title', help='Judul lagu')
    parser.add_argument('-a', '--artist', help='Nama artis')
    parser.add_argument('--card-bg', default='card-background.png', help='Path ke gambar latar belakang kartu')
    parser.add_argument('--quality', choices=['standard', 'hd', '4k'], help='Preset kualitas video')
    parser.add_argument('--fps', type=int, help='FPS video output')
    parser.add_argument('--config', help='Path ke file konfigurasi JSON kustom')
    
    args = parser.parse_args()
    
    print("\n🔍 Validating input file...")
    is_valid, message = validate_input_file(args.input)
    if not is_valid:
        print(f"   ❌ Error: {message}")
        sys.exit(1)
    print(f"   ✅ Input file valid: {args.input}")
    
    metadata = {}
    if args.title and args.artist:
        metadata['title'], metadata['artist'] = args.title, args.artist
    elif not args.title and not args.artist:
        metadata = get_metadata_from_user()
    else:
        metadata['title'] = args.title or "Unknown Title"
        metadata['artist'] = args.artist or "Unknown Artist"
    
    output_file = args.output if args.output else create_output_filename(args.input)
    
    custom_config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
            print(f"\n📋 Loaded custom config: {args.config}")
        except Exception as e:
            print(f"   ⚠️ Warning: Gagal memuat file config: {e}")
    
    if args.fps: custom_config['FPS'] = args.fps
    if args.quality: custom_config['QUALITY'] = args.quality

    print("\n📊 Configuration Summary:")
    print(f"   🎵 Input: {Path(args.input).name}")
    print(f"   💾 Output: {output_file}")
    # Gunakan .get() untuk akses aman
    print(f"   🎬 Quality: {custom_config.get('QUALITY', 'hd')} @ {custom_config.get('FPS', 60)}fps")
    print(f"   🎯 Title: {metadata['title']}")
    print(f"   🎤 Artist: {metadata['artist']}")
    print("\n" + "="*60)
    
    print("\n🚀 Initializing Visualizer...")
    
    try:
        visualizer = GlassCardSpectrumVisualizer(
            audio_file=args.input,
            output_file=output_file,
            card_background=args.card_bg,
            metadata=metadata,
            custom_config=custom_config
        )
        
        print("\n🎬 Starting video generation...")
        print("   Proses ini bisa memakan waktu beberapa menit...")
        
        success = visualizer.create_video()
        
        if success:
            print("\n" + "="*60)
            print("🎉 SUKSES! Video berhasil dibuat!")
            print(f"📁 Output disimpan di: {os.path.abspath(output_file)}")
            if os.path.exists(output_file):
                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                print(f"📊 Ukuran file: {size_mb:.2f} MB")
            print("\n✨ Selamat menikmati visualisasi Anda! ✨")
        else:
            print("\n❌ Gagal membuat video. Periksa pesan error di atas.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Proses dihentikan oleh pengguna.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Terjadi error tak terduga: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
# --- END OF FILE main.py ---