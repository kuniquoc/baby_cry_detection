import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# Import các hàm từ data_processing
from data_processing.preprocess import apply_vad, extract_mfcc
from data_processing.prepare_dataset import create_dataset_splits
from data_processing.split_audio import split_audio, save_segments

class AudioDataset(Dataset):
    """
    Dataset PyTorch cho dữ liệu âm thanh
    """
    def __init__(self, file_paths, labels, transform=None, augment=False):
        """
        Khởi tạo dataset
        
        Args:
            file_paths (list): Danh sách đường dẫn file âm thanh
            labels (list): Danh sách nhãn tương ứng
            transform (callable, optional): Hàm biến đổi để áp dụng cho mẫu
            augment (bool): Có áp dụng data augmentation hay không
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Tải và xử lý âm thanh
        if self.transform:
            features = self.transform(file_path, self.augment)
        else:
            features = None
            
        return features, label

class DatasetLoader:
    """
    Lớp DatasetLoader dùng để tải và xử lý dữ liệu âm thanh cho mô hình phát hiện tiếng khóc của trẻ.
    Sử dụng các hàm có sẵn từ thư mục data_processing.
    
    Attributes:
        data_dir (Path): Đường dẫn đến thư mục chứa dữ liệu
        sample_rate (int): Tần số lấy mẫu cho âm thanh (mặc định: 16000 Hz)
        n_mels (int): Số lượng mel bands cho việc trích xuất đặc trưng (mặc định: 128)
        n_fft (int): Kích thước cửa sổ FFT (mặc định: 2048)
        hop_length (int): Độ dài bước nhảy cho FFT (mặc định: 512)
        duration (float): Độ dài mỗi đoạn âm thanh tính bằng giây (mặc định: 3.0)
        classes (list): Danh sách các lớp ('cry', 'not_cry')
        label_encoder (LabelEncoder): Bộ mã hóa nhãn
    """
    
    def __init__(self, data_dir='data/dataset', sample_rate=16000, n_mels=40, n_fft=512, 
                 hop_length=160, duration=3.0):
        """
        Khởi tạo DatasetLoader với các tham số cấu hình.
        
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa dữ liệu
            sample_rate (int): Tần số lấy mẫu cho âm thanh
            n_mels (int): Số lượng mel bands cho việc trích xuất đặc trưng
            n_fft (int): Kích thước cửa sổ FFT
            hop_length (int): Độ dài bước nhảy cho FFT
            duration (float): Độ dài mỗi đoạn âm thanh tính bằng giây
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.classes = ['cry', 'not_cry']
        self.label_encoder = LabelEncoder().fit(self.classes)
        
    def load_metadata(self):
        """
        Tạo metadata từ cấu trúc thư mục dữ liệu.
        
        Returns:
            pd.DataFrame: DataFrame chứa thông tin về các file âm thanh và nhãn tương ứng
        """
        metadata = []
        
        # Duyệt qua các thư mục train, val, test
        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue
                
            # Duyệt qua các thư mục cry và not_cry
            for label in self.classes:
                label_dir = split_dir / label
                if not label_dir.exists():
                    continue
                    
                # Lấy danh sách các file wav
                for audio_file in label_dir.glob('*.wav'):
                    metadata.append({
                        'file_path': str(audio_file),
                        'label': label,
                        'split': split
                    })
        
        # Tạo DataFrame từ metadata
        df = pd.DataFrame(metadata)
        print(f"Loaded metadata with {len(df)} files")
        print(f"Split distribution: {df['split'].value_counts().to_dict()}")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def extract_features(self, audio_path, augment=False):
        """
        Trích xuất đặc trưng từ file âm thanh sử dụng các hàm từ preprocess.py.
        
        Args:
            audio_path (str): Đường dẫn đến file âm thanh
            augment (bool): Có áp dụng data augmentation hay không
            
        Returns:
            torch.Tensor: Đặc trưng đã được trích xuất
        """
        try:
            # Tải file âm thanh với librosa
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Áp dụng VAD trước
            y_vad = apply_vad(y, sr=self.sample_rate)
            
            # Nếu VAD loại bỏ quá nhiều, sử dụng lại âm thanh gốc
            if len(y_vad) < 0.5 * len(y):
                y_vad = y
                
            # Sau đó mới điều chỉnh độ dài
            target_length = int(self.duration * self.sample_rate)
            if len(y_vad) < target_length:
                # Pad nếu âm thanh ngắn hơn độ dài mục tiêu
                y_vad = np.pad(y_vad, (0, target_length - len(y_vad)), 'constant')
            else:
                # Cắt nếu âm thanh dài hơn độ dài mục tiêu
                y_vad = y_vad[:target_length]
            
            # Trích xuất MFCC từ preprocess.py
            mfccs = extract_mfcc(
                y_vad, 
                sr=self.sample_rate,
                n_mfcc=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Áp dụng data augmentation nếu được yêu cầu
            if augment:
                mfccs = self._apply_augmentation(mfccs)
                
            # Chuyển đổi sang tensor PyTorch và thêm chiều kênh
            features = torch.from_numpy(mfccs).float().unsqueeze(0)  # [1, n_mels, time]
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def prepare_dataset(self, batch_size=32, num_workers=4):
        """
        Chuẩn bị dataset PyTorch từ metadata.
        
        Args:
            batch_size (int): Kích thước batch
            num_workers (int): Số lượng worker cho DataLoader
            
        Returns:
            tuple: (train_loader, val_loader, test_loader) - Các DataLoader cho train, validation và test
        """
        # Tải metadata
        df = self.load_metadata()
        
        # Tạo các dataset riêng cho train, val và test
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']
        
        # Tạo các PyTorch dataset
        train_dataset = self._create_dataset(train_df, augment=True) if not train_df.empty else None
        val_dataset = self._create_dataset(val_df, augment=False) if not val_df.empty else None
        test_dataset = self._create_dataset(test_df, augment=False) if not test_df.empty else None
        
        # Tạo các DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        ) if train_dataset else None
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        ) if val_dataset else None
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        ) if test_dataset else None
        
        return train_loader, val_loader, test_loader
    
    def _create_dataset(self, df, augment=False):
        """
        Tạo PyTorch dataset từ DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame chứa thông tin về các file âm thanh
            augment (bool): Có áp dụng data augmentation hay không
            
        Returns:
            AudioDataset: PyTorch dataset
        """
        if df.empty:
            return None
            
        # Lấy đường dẫn file và nhãn
        file_paths = df['file_path'].values
        labels = self.label_encoder.transform(df['label'].values)
        
        # Tạo dataset
        dataset = AudioDataset(
            file_paths=file_paths,
            labels=labels,
            transform=self.extract_features,
            augment=augment
        )
        
        return dataset
    
    def _apply_augmentation(self, features):
        """
        Áp dụng data augmentation cho đặc trưng.
        Lưu ý: Đây là augmentation trên đặc trưng, khác với augment_and_save trong augment.py
        
        Args:
            features (np.ndarray): Đặc trưng đầu vào
            
        Returns:
            np.ndarray: Đặc trưng đã được augment
        """
        choice = np.random.randint(0, 3)  # Chỉ dùng 3 phương pháp phù hợp
        
        if choice == 0:  
            # Frequency Masking (Giữ lại nhưng giảm mức độ)
            freq_mask_param = 5  # Giảm số lượng tần số bị che
            freq_mask_idx = np.random.randint(0, features.shape[0] - freq_mask_param)
            features[freq_mask_idx:freq_mask_idx + freq_mask_param, :] = 0

        elif choice == 1:  
            # Time Masking (Giữ lại nhưng giảm mức độ)
            time_mask_param = 5  # Giảm số lượng thời gian bị che
            time_mask_idx = np.random.randint(0, features.shape[1] - time_mask_param)
            features[:, time_mask_idx:time_mask_idx + time_mask_param] = 0

        elif choice == 2:  
            # Frequency Shift (Chỉ dịch chuyển nhẹ)
            shift = np.random.randint(-1, 2)  # Dịch chuyển nhẹ hơn (-1, 0, 1)
            if shift > 0:
                features[shift:, :] = features[:-shift, :]
                features[:shift, :] = 0
            elif shift < 0:
                features[:shift, :] = features[-shift:, :]
                features[shift:, :] = 0
        
        return features
    
    def process_raw_data(self, raw_dir='data/raw', segmented_dir='data/segmented', dataset_dir='data/dataset'):
        """
        Xử lý dữ liệu thô từ đầu đến cuối sử dụng các hàm từ data_processing.
        
        Args:
            raw_dir (str): Đường dẫn đến thư mục chứa dữ liệu thô
            segmented_dir (str): Đường dẫn đến thư mục để lưu dữ liệu đã phân đoạn
            dataset_dir (str): Đường dẫn đến thư mục để lưu dataset đã chia
        """
        # Bước 1: Phân đoạn âm thanh từ raw_dir sang segmented_dir
        print("Step 1: Splitting audio files...")
        base_dir = Path('data')
        raw_root = base_dir / "raw"
        segmented_root = base_dir / "segmented"
        
        # Khởi tạo bộ đếm và lưu trữ đoạn
        segment_counts = defaultdict(int)
        all_segments = defaultdict(list)
        
        # Xử lý file từ thư mục raw
        for label in ["cry", "not_cry"]:
            print(f"\nProcessing {label} files from raw directory...")
            label_dir = raw_root / label
            if label_dir.exists():
                for file_name in os.listdir(label_dir):
                    if file_name.endswith(".wav"):
                        file_path = os.path.join(label_dir, file_name)
                        segments, _ = split_audio(file_path)
                        all_segments[label].extend(segments)
                        print(f"Created {len(segments)} segments from {file_name}")
        
        # Lưu các đoạn
        for label in ["cry", "not_cry"]:
            if len(all_segments[label]) > 0:
                count = save_segments(all_segments[label], segmented_root, label)
                segment_counts[label] += count
                print(f"Saved {count} {label} segments to segmented/{label}")
        
        # Bước 2: Chia dữ liệu thành train, val, test
        print("\nStep 2: Creating dataset splits...")
        create_dataset_splits(segmented_dir, dataset_dir)
        
        print("\nData processing completed!")
        
    def visualize_sample(self, split='train', label='cry', index=0):
        """
        Hiển thị mẫu âm thanh và đặc trưng.
        
        Args:
            split (str): Split để lấy mẫu ('train', 'val', 'test')
            label (str): Nhãn để lấy mẫu ('cry', 'not_cry')
            index (int): Chỉ số của mẫu
        """
        # Tải metadata
        df = self.load_metadata()
        
        # Lọc theo split và label
        filtered_df = df[(df['split'] == split) & (df['label'] == label)]
        
        if index >= len(filtered_df):
            print(f"Index {index} out of range. Only {len(filtered_df)} samples available.")
            return
            
        # Lấy đường dẫn file
        file_path = filtered_df.iloc[index]['file_path']
        
        # Tải âm thanh
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        
        # Trích xuất đặc trưng
        features = self.extract_features(file_path)
        
        if features is None:
            print(f"Failed to extract features from {file_path}")
            return
            
        # Chuyển tensor PyTorch về numpy để hiển thị
        if isinstance(features, torch.Tensor):
            features = features.squeeze(0).numpy()  # Loại bỏ chiều kênh
            
        # Hiển thị dạng sóng
        plt.figure(figsize=(10, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
        plt.title(f'Waveform - {label} ({split})')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Hiển thị dạng sóng sau VAD
        y_vad, isBabyCrying, f0 = apply_vad(y, sr=self.sample_rate)

        print("f0", f0)

        if not isBabyCrying:
            print("BabyCry:",isBabyCrying)
            return;
            
        plt.subplot(3, 1, 2)
        plt.plot(np.linspace(0, len(y_vad)/sr, len(y_vad)), y_vad)
        plt.title(f'Waveform after VAD - {label} ({split})')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Hiển thị MFCC
        plt.subplot(3, 1, 3)
        librosa.display.specshow(
            features, 
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'MFCC Features - {label} ({split})')
        
        plt.tight_layout()
        plt.show()
        
    def get_class_weights(self):
        """
        Tính toán trọng số cho các lớp để xử lý dữ liệu không cân bằng.
        
        Returns:
            torch.Tensor: Tensor chứa trọng số cho mỗi lớp
        """
        # Tải metadata
        df = self.load_metadata()
        
        # Lọc chỉ lấy dữ liệu train
        train_df = df[df['split'] == 'train']
        
        # Đếm số lượng mẫu cho mỗi lớp
        class_counts = train_df['label'].value_counts().to_dict()
        
        # Tính tổng số mẫu
        total_samples = len(train_df)
        
        # Tính trọng số cho mỗi lớp
        weights = []
        for class_name in self.classes:
            if class_name in class_counts:
                # Công thức: total_samples / (n_classes * class_count)
                weight = total_samples / (len(self.classes) * class_counts[class_name])
            else:
                weight = 1.0
            weights.append(weight)
                
        return torch.tensor(weights, dtype=torch.float)


# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo DatasetLoader
    loader = DatasetLoader(data_dir='data/dataset')
    
    # Xử lý dữ liệu thô (nếu cần)
    # loader.process_raw_data()
    
    # Tải metadata
    metadata = loader.load_metadata()
    print(metadata.head())
    
    # # Chuẩn bị dataset
    # train_loader, val_loader, test_loader = loader.prepare_dataset(batch_size=32)
    
    # Hiển thị một mẫu
    loader.visualize_sample(split='train', label='cry', index=0)
    
    # Lấy trọng số cho các lớp
    class_weights = loader.get_class_weights()
    print("Class weights:", class_weights)
    
    # # Kiểm tra một batch từ train_loader
    # if train_loader:
    #     for batch_features, batch_labels in train_loader:
    #         print(f"Batch shape: {batch_features.shape}")
    #         print(f"Labels shape: {batch_labels.shape}")
    #         break
    
