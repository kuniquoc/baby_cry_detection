import csv
import os

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Update paths using project root
csv_file = os.path.join(project_root, 'results', 'val_predictions.csv')
output_file = os.path.join(project_root, 'delete.txt')

# Đọc file CSV và lọc các bản ghi
false_predictions = []
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Chuyển đổi các giá trị
        confidence = float(row['confidence'])
        correct = row['correct'].lower() == 'true'
        
        # Lọc theo điều kiện
        if not correct and confidence > 0.8:
            # Lấy đường dẫn đầy đủ
            full_path = os.path.join(project_root, 'data', 'dataset', 'val', row['filename'])
            false_predictions.append(full_path)

# Ghi kết quả vào file delete.txt
with open(output_file, 'w') as f:
    for path in false_predictions:
        f.write(f'"{path}"\n')