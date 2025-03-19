import os
import numpy as np
import librosa
import pygame
import threading
import tempfile
import soundfile as sf
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Fix the import path by adding the parent directory to sys.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.preprocess import apply_vad, pad_or_trim_audio

class VADTesterGUI:
    """
    Giao diện đồ họa để kiểm tra chức năng VAD và hiển thị kết quả.
    """
    def __init__(self, sample_rate=16000, duration=3.0):
        """
        Khởi tạo giao diện VAD Tester
        
        Args:
            sample_rate (int): Tần số lấy mẫu cho âm thanh
            duration (float): Độ dài mỗi đoạn âm thanh tính bằng giây
        """
        self.sample_rate = sample_rate
        self.duration = duration
        
        # Khởi tạo pygame mixer cho việc phát âm thanh
        pygame.mixer.init()
        
        # Biến để lưu trữ dữ liệu âm thanh
        self.audio_data = {
            'original': {'path': None, 'audio': None, 'sr': None},
            'processed': {'path': None, 'audio': None, 'sr': None}
        }
        
    def load_audio(self):
        """Tải file âm thanh từ người dùng"""
        file_path = filedialog.askopenfilename(
            title="Chọn File Âm Thanh",
            filetypes=[("Audio Files", "*.wav")]
        )
        
        if file_path:
            try:
                # Tải âm thanh
                y, sr = librosa.load(file_path, sr=self.sample_rate)
                self.audio_data['original']['path'] = file_path
                self.audio_data['original']['audio'] = y
                self.audio_data['original']['sr'] = sr
                
                # Cập nhật UI
                self.file_label.config(text=f"Đã tải: {os.path.basename(file_path)}")
                self.process_button.config(state=tk.NORMAL)
                self.play_original_button.config(state=tk.NORMAL)
                
                # Vẽ dạng sóng gốc
                self.plot_waveform(y, sr, "Âm thanh gốc")
                
                # Xóa âm thanh đã xử lý
                self.audio_data['processed']['path'] = None
                self.audio_data['processed']['audio'] = None
                self.play_processed_button.config(state=tk.DISABLED)
                
            except Exception as e:
                self.error_label.config(text=f"Lỗi khi tải âm thanh: {str(e)}")
    
    def process_audio(self):
        """Xử lý âm thanh với VAD"""
        if self.audio_data['original']['audio'] is not None:
            try:
                # Áp dụng VAD
                y = self.audio_data['original']['audio']
                sr = self.audio_data['original']['sr']
                
                y_vad, is_baby_crying, rms, zcr, f0 = apply_vad(y, sr=sr)
                
                # Cập nhật trạng thái
                status_text = f"Kết quả VAD:\nCó tiếng khóc: {is_baby_crying}\nRMS: {rms:.4f}\nZCR: {zcr:.4f}\nF0: {f0:.4f} Hz"
                self.status_label.config(text=status_text)
                
                # Pad hoặc cắt âm thanh đã xử lý
                y_vad = pad_or_trim_audio(y_vad, sr * 3)  # 3 giây
                
                # Lưu âm thanh đã xử lý
                self.audio_data['processed']['audio'] = y_vad
                self.audio_data['processed']['sr'] = sr
                
                # Tạo file tạm thời để phát
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_file.close()
                sf.write(temp_file.name, y_vad, sr)
                self.audio_data['processed']['path'] = temp_file.name
                
                # Cập nhật UI
                self.play_processed_button.config(state=tk.NORMAL)
                
                # Vẽ dạng sóng đã xử lý
                self.plot_waveform(y_vad, sr, "Âm thanh sau VAD", is_processed=True)
                
            except Exception as e:
                self.error_label.config(text=f"Lỗi khi xử lý âm thanh: {str(e)}")
    
    def play_audio(self, audio_type):
        """Phát âm thanh"""
        def _play():
            try:
                pygame.mixer.music.load(self.audio_data[audio_type]['path'])
                pygame.mixer.music.play()
            except Exception as e:
                self.error_label.config(text=f"Lỗi khi phát âm thanh: {str(e)}")
        
        if audio_type == 'original' and self.audio_data['original']['path']:
            _play()
        elif audio_type == 'processed' and self.audio_data['processed']['path']:
            _play()
    
    def plot_waveform(self, y, sr, title, is_processed=False):
        """Vẽ dạng sóng âm thanh"""
        fig_index = 1 if not is_processed else 2
        fig = Figure(figsize=(7, 2))
        ax = fig.add_subplot(111)
        ax.plot(np.linspace(0, len(y)/sr, len(y)), y)
        ax.set_title(title)
        ax.set_xlabel('Thời gian (s)')
        ax.set_ylabel('Biên độ')
        
        # Xóa đồ thị trước đó nếu có
        for widget in self.plot_frames[fig_index-1].winfo_children():
            widget.destroy()
        
        # Tạo canvas
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frames[fig_index-1])
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_widgets(self, root):
        """Tạo các widget cho giao diện"""
        # Tạo các frame
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(fill=tk.X)
        
        self.plot_frames = [
            ttk.Frame(root, padding=10),
            ttk.Frame(root, padding=10)
        ]
        for frame in self.plot_frames:
            frame.pack(fill=tk.BOTH, expand=True)
        
        status_frame = ttk.Frame(root, padding=10)
        status_frame.pack(fill=tk.X)
        
        # Các phần tử điều khiển
        load_button = ttk.Button(control_frame, text="Tải Âm Thanh", command=self.load_audio)
        load_button.pack(side=tk.LEFT, padx=5)
        
        self.process_button = ttk.Button(control_frame, text="Áp Dụng VAD", command=self.process_audio, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        self.play_original_button = ttk.Button(
            control_frame, 
            text="Phát Âm Thanh Gốc", 
            command=lambda: self.play_audio('original'),
            state=tk.DISABLED
        )
        self.play_original_button.pack(side=tk.LEFT, padx=5)
        
        self.play_processed_button = ttk.Button(
            control_frame, 
            text="Phát Âm Thanh Đã Xử Lý", 
            command=lambda: self.play_audio('processed'),
            state=tk.DISABLED
        )
        self.play_processed_button.pack(side=tk.LEFT, padx=5)
        
        # Các phần tử trạng thái
        self.file_label = ttk.Label(status_frame, text="Chưa tải file nào")
        self.file_label.pack(anchor=tk.W)
        
        self.status_label = ttk.Label(status_frame, text="")
        self.status_label.pack(anchor=tk.W)
        
        self.error_label = ttk.Label(status_frame, text="", foreground="red")
        self.error_label.pack(anchor=tk.W)
    
    def run(self):
        """Khởi chạy giao diện"""
        # Tạo cửa sổ chính
        root = tk.Tk()
        root.title("Kiểm Tra VAD Tiếng Khóc Trẻ Em")
        root.geometry("800x600")
        
        # Tạo các widget
        self.create_widgets(root)
        
        # Bắt đầu vòng lặp chính
        root.mainloop()
        
        # Dọn dẹp file tạm thời
        if self.audio_data['processed']['path'] and os.path.exists(self.audio_data['processed']['path']):
            try:
                os.unlink(self.audio_data['processed']['path'])
            except:
                pass

def launch_vad_tester():
    """Hàm tiện ích để khởi chạy giao diện VAD Tester"""
    tester = VADTesterGUI()
    tester.run()

if __name__ == "__main__":
    launch_vad_tester()