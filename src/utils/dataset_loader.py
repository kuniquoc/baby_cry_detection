import os
import sys
import numpy as np
import pandas as pd
import librosa
import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Replace relative imports with absolute imports
from src.data_processing.preprocess import apply_vad, extract_mfcc
from src.data_processing.prepare_dataset import create_dataset_splits
from src.data_processing.split_audio import split_audio, save_segments

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
    
    def __init__(self, data_dir='data/dataset', processed_dir='data/processed', sample_rate=16000, n_mels=40, n_fft=512, 
                 hop_length=160, duration=3.0):
        """
        Khởi tạo DatasetLoader với các tham số cấu hình.
        
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa dữ liệu
            processed_dir (str): Đường dẫn đến thư mục lưu đặc trưng đã xử lý
            sample_rate (int): Tần số lấy mẫu cho âm thanh
            n_mels (int): Số lượng mel bands cho việc trích xuất đặc trưng
            n_fft (int): Kích thước cửa sổ FFT
            hop_length (int): Độ dài bước nhảy cho FFT
            duration (float): Độ dài mỗi đoạn âm thanh tính bằng giây
            classes (list): Danh sách các lớp ('cry', 'not_cry')
            label_encoder (LabelEncoder): Bộ mã hóa nhãn
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.classes = ['not_cry', 'cry']  # Định nghĩa các lớp cho mô hình phân loại
        self.label_encoder = LabelEncoder().fit(self.classes)  # Chuyển nhãn text thành số (0, 1)
        
        # Tạo cấu trúc thư mục cho dữ liệu đã xử lý, tách riêng cho mỗi lớp
        for split in ['train', 'val', 'test']:
            for label in self.classes:  # Tạo thư mục cho mỗi lớp cry và not_cry
                os.makedirs(self.processed_dir / split / label, exist_ok=True)
        
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
        # print(f"Loaded metadata with {len(df)} files")
        # print(f"Split distribution: {df['split'].value_counts().to_dict()}")
        # print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
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
            
            # Trích xuất MFCC từ preprocess.py
            mfccs = extract_mfcc(
                y, 
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
    
    def get_processed_path(self, file_path):
        """
        Tạo đường dẫn đến file đã xử lý dựa trên đường dẫn file gốc.
        
        Args:
            file_path (str): Đường dẫn đến file âm thanh
            
        Returns:
            Path: Đường dẫn đến file đặc trưng đã xử lý (.npy)
        """
        # Chuyển thành đối tượng Path
        orig_path = Path(file_path)
        # Lấy tên file và vị trí (split/label)
        splits = orig_path.parts
        
        # Tìm vị trí của 'train', 'val', 'test' trong đường dẫn
        for i, part in enumerate(splits):
            if part in ['train', 'val', 'test']:
                split = part
                label = splits[i+1]  # Giả sử label là thư mục sau split
                filename = orig_path.stem  # Tên file không có phần mở rộng
                
                # Tạo đường dẫn đến file đã xử lý
                return self.processed_dir / split / label / f"{filename}.npy"
                
        # Nếu không tìm thấy cấu trúc phù hợp, tạo đường dẫn nguyên bản
        return self.processed_dir / f"{orig_path.stem}.npy"
    
    def save_features(self, features, file_path):
        """
        Lưu đặc trưng đã trích xuất vào thư mục processed.
        
        Args:
            features (torch.Tensor): Đặc trưng để lưu
            file_path (str): Đường dẫn đến file âm thanh gốc
        """
        # Tạo đường dẫn đến file đã xử lý
        processed_path = self.get_processed_path(file_path)
        
        # Tạo thư mục nếu chưa tồn tại
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Lưu đặc trưng dưới dạng numpy array
        try:
            # Convert torch tensor to numpy array if needed
            if isinstance(features, torch.Tensor):
                features_np = features.cpu().numpy()
            else:
                features_np = features
                
            np.save(processed_path, features_np)
        except Exception as e:
            print(f"Error saving feature file {processed_path}: {e}")
            # If saving fails, try to remove the potentially corrupted file
            if processed_path.exists():
                try:
                    os.remove(processed_path)
                except:
                    pass
    
    def load_features(self, file_path):
        """
        Tải đặc trưng đã lưu từ thư mục processed.
        
        Args:
            file_path (str): Đường dẫn đến file âm thanh gốc
            
        Returns:
            torch.Tensor: Đặc trưng đã trích xuất hoặc None nếu không tồn tại
        """
        # Tạo đường dẫn đến file đã xử lý
        processed_path = self.get_processed_path(file_path)
        
        # Kiểm tra xem file có tồn tại không
        if processed_path.exists():
            try:
                # Tải đặc trưng từ file numpy
                features_np = np.load(processed_path)
                # Chuyển đổi sang tensor PyTorch
                return torch.from_numpy(features_np).float()
            except Exception as e:
                print(f"Error loading numpy file {processed_path}: {e}")
                # Xóa file bị lỗi để tính toán lại
                try:
                    os.remove(processed_path)
                    print(f"Removed corrupted feature file: {processed_path}")
                except Exception as ex:
                    print(f"Could not remove file: {ex}")
                return None
        
        return None
    
    def convert_audio_to_mfcc(self, force_recompute=False):
        """
        Chuyển đổi tất cả file âm thanh trong dataset thành đặc trưng MFCC và lưu dưới dạng npy.
        Tách riêng quá trình chuyển đổi audio sang MFCC.
        
        Args:
            force_recompute (bool): Bắt buộc tính toán lại tất cả các đặc trưng
        
        Returns:
            int: Số lượng file đã được xử lý
        """
        print("Converting audio files to MFCC features...")
        
        # Tải metadata
        df = self.load_metadata()
        
        if df.empty:
            print("No audio files found in metadata.")
            return 0
        
        # Đếm số lượng file đã xử lý
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Lấy đường dẫn file
        file_paths = df['file_path'].values
        labels = df['label'].values
        splits = df['split'].values
        
        # Trích xuất đặc trưng cho từng file
        for i, (file_path, label, split) in enumerate(tqdm(zip(file_paths, labels, splits), 
                                                          total=len(file_paths), 
                                                          desc="Converting to MFCCs")):
            processed_path = self.get_processed_path(file_path)
            
            # Kiểm tra xem đặc trưng đã tồn tại chưa
            if processed_path.exists() and not force_recompute:
                skipped_count += 1
                continue
                
            # Trích xuất đặc trưng
            features = self.extract_features(file_path, augment=False)
            
            if features is not None:
                # Lưu đặc trưng vào thư mục processed
                self.save_features(features, file_path)
                processed_count += 1
            else:
                failed_count += 1
                
        print(f"MFCC conversion completed:")
        print(f"- Processed: {processed_count} files")
        print(f"- Skipped (already exists): {skipped_count} files")
        print(f"- Failed: {failed_count} files")
        
        return processed_count
    
    def _precompute_mfccs(self, df, augment=False):
        """
        Tải các đặc trưng MFCC đã được tính toán từ thư mục processed.
        
        Args:
            df (pd.DataFrame): DataFrame chứa thông tin về các file âm thanh
            augment (bool): Áp dụng data augmentation hay không
            
        Returns:
            tuple: (features_list, labels_list) - Danh sách đặc trưng và nhãn tương ứng
        """
        if df.empty:
            return [], []
        
        features_list = []
        labels_list = []
        
        # Lấy đường dẫn file và nhãn
        file_paths = df['file_path'].values
        labels = self.label_encoder.transform(df['label'].values)
        
        # Tải đặc trưng đã được tính toán từ thư mục processed
        for i, (file_path, label) in enumerate(tqdm(zip(file_paths, labels), 
                                                  total=len(file_paths), 
                                                  desc="Loading MFCCs")):
            # Tải đặc trưng đã được lưu
            features = self.load_features(file_path)
            
            if features is not None:
                # Áp dụng augmentation nếu được yêu cầu
                if augment:
                    # Chuyển về numpy để augment
                    features_np = features.numpy() if isinstance(features, torch.Tensor) else features
                    features_np = self._apply_augmentation(features_np)
                    features = torch.from_numpy(features_np).float()
                
                features_list.append(features)
                labels_list.append(label)
            else:
                print(f"Warning: Missing processed feature for {file_path}")
        
        return features_list, labels_list

    def prepare_dataset(self, batch_size=32, num_workers=4):
        """
        Chuẩn bị dataset PyTorch bằng cách tải các đặc trưng MFCC đã tính toán từ thư mục processed.
        
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
        
        print("Loading MFCC features from processed directory...")
        
        # Tải đặc trưng đã được tính toán từ thư mục processed
        train_features, train_labels = self._precompute_mfccs(train_df, augment=False)
        val_features, val_labels = self._precompute_mfccs(val_df, augment=False)
        test_features, test_labels = self._precompute_mfccs(test_df, augment=False)
        
        print(f"Loaded features - Train: {len(train_features)}, Val: {len(val_features)}, Test: {len(test_features)}")
        
        # Create TensorDatasets với các đặc trưng đã tính toán trước
        train_dataset = TensorDataset(
            torch.stack(train_features) if train_features else torch.tensor([]),
            torch.tensor(train_labels, dtype=torch.float) if train_labels else torch.tensor([])
        ) if train_features and len(train_features) > 0 else None
        
        val_dataset = TensorDataset(
            torch.stack(val_features) if val_features else torch.tensor([]),
            torch.tensor(val_labels, dtype=torch.float) if val_labels else torch.tensor([])
        ) if val_features and len(val_features) > 0 else None
        
        test_dataset = TensorDataset(
            torch.stack(test_features) if test_features else torch.tensor([]),
            torch.tensor(test_labels, dtype=torch.float) if test_labels else torch.tensor([])
        ) if test_features and len(test_features) > 0 else None
        
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
        
    def get_class_weights(self):
        """
        Tính toán trọng số cho các lớp để xử lý dữ liệu không cân bằng.
        
        Returns:
            torch.Tensor: Tensor chứa trọng số cho lớp dương (pos_weight)
        """
        # Tải metadata
        df = self.load_metadata()
        
        # Lọc chỉ lấy dữ liệu train
        train_df = df[df['split'] == 'train']
        
        # Đếm số lượng mẫu cho mỗi lớp
        class_counts = train_df['label'].value_counts().to_dict()
        
        # In ra thống kê
        print("Class counts:", class_counts)
        
        # Tính toán pos_weight cho BCEWithLogitsLoss: số mẫu lớp âm / số mẫu lớp dương
        # Các lớp được tính theo thứ tự: [cry, not_cry] trong self.classes
        # cry là lớp dương (1), not_cry là lớp âm (0)
        pos_weight = class_counts.get('not_cry', 0) / max(class_counts.get('cry', 0), 1)
        print(f"Using pos_weight={pos_weight} for BCEWithLogitsLoss")
        
        return torch.tensor([pos_weight], dtype=torch.float)


# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo DatasetLoader
    loader = DatasetLoader(data_dir='data/dataset')
    
    # Chuyển đổi tất cả file âm thanh thành MFCC features
    loader.convert_audio_to_mfcc()
    
    # Tạo dataset từ MFCC features đã được tính toán
    train_loader, val_loader, test_loader = loader.prepare_dataset(batch_size=32)
    
    # Kiểm tra một batch từ train_loader
    if train_loader:
        for batch_features, batch_labels in train_loader:
            print(f"Batch shape: {batch_features.shape}")
            print(f"Labels shape: {batch_labels.shape}")
            break

