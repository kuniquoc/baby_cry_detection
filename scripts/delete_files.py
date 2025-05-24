import os

def delete_files_from_list(delete_txt_path):
    # Đọc danh sách file cần xóa từ delete.txt
    with open(delete_txt_path, 'r') as f:
        files_to_delete = [line.strip().strip('"') for line in f.readlines()]
    
    # Xóa từng file
    deleted_count = 0
    failed_count = 0
    
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Đã xóa thành công: {file_path}")
                deleted_count += 1
            else:
                print(f"File không tồn tại: {file_path}")
                failed_count += 1
        except Exception as e:
            print(f"Lỗi khi xóa file {file_path}: {str(e)}")
            failed_count += 1
    
    print(f"\nTổng kết:")
    print(f"- Số file đã xóa thành công: {deleted_count}")
    print(f"- Số file thất bại: {failed_count}")

if __name__ == "__main__":
    # Đường dẫn tới file delete.txt
    delete_txt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "delete.txt")
    
    # Kiểm tra file delete.txt có tồn tại không
    if not os.path.exists(delete_txt_path):
        print(f"Không tìm thấy file: {delete_txt_path}")
    else:
        print("Bắt đầu xóa các file...")
        delete_files_from_list(delete_txt_path)