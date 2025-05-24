# import os
# import random

# def delete_random_files(directory_path, num_files=None, percentage=None):
#     """
#     Xóa ngẫu nhiên các file trong thư mục được chỉ định
    
#     Args:
#         directory_path (str): Đường dẫn đến thư mục cần xóa file
#         num_files (int, optional): Số lượng file cần xóa
#         percentage (float, optional): Phần trăm số file cần xóa (0-100)
#     """
#     if not os.path.exists(directory_path):
#         print(f"Thư mục không tồn tại: {directory_path}")
#         return

#     # Lấy danh sách tất cả các file trong thư mục
#     all_files = []
#     for root, _, files in os.walk(directory_path):
#         for file in files:
#             all_files.append(os.path.join(root, file))

#     if not all_files:
#         print("Không có file nào trong thư mục")
#         return

#     total_files = len(all_files)
    
#     # Xác định số lượng file cần xóa
#     if percentage is not None:
#         num_to_delete = int(total_files * percentage / 100)
#     elif num_files is not None:
#         num_to_delete = min(num_files, total_files)
#     else:
#         print("Vui lòng chỉ định số lượng file hoặc phần trăm file cần xóa")
#         return

#     # Chọn ngẫu nhiên các file để xóa
#     files_to_delete = random.sample(all_files, num_to_delete)
    
#     # Xác nhận từ người dùng
#     print(f"\nSẽ xóa {num_to_delete} file từ tổng số {total_files} file.")
#     print("Các file sẽ bị xóa:")
#     for file in files_to_delete:
#         print(f"- {file}")
    
#     # Xóa các file
#     deleted_count = 0
#     failed_count = 0
    
#     print(f"\nBắt đầu xóa file...")
#     for file_path in files_to_delete:
#         try:
#             os.remove(file_path)
#             print(f"Đã xóa thành công: {file_path}")
#             deleted_count += 1
#         except Exception as e:
#             print(f"Lỗi khi xóa file {file_path}: {str(e)}")
#             failed_count += 1
    
#     print(f"\nTổng kết:")
#     print(f"- Số file đã xóa thành công: {deleted_count}")
#     print(f"- Số file thất bại: {failed_count}")

# if __name__ == "__main__":
#     # === CẤU HÌNH Ở ĐÂY ===
    
#     # Đường dẫn thư mục cần xóa file
#     DIRECTORY = "D:/Git/baby_cry_detection/data/dataset/test/not_cry"
    
#     # Chọn MỘT trong hai cách dưới đây:
    
#     # Cách 1: Xóa theo số lượng file
#     NUM_FILES = 2633 - 2630  # Số lượng file cần xóa
#     delete_random_files(DIRECTORY, num_files=NUM_FILES)
    
#     # Cách 2: Xóa theo phần trăm
#     # PERCENTAGE = 20  # Phần trăm file cần xóa (0-100)
#     # delete_random_files(DIRECTORY, percentage=PERCENTAGE)