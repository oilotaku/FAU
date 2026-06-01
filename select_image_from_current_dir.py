import os

def select_image_from_current_dir():
    current_folder = os.getcwd() 
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG')
    files = [f for f in os.listdir(current_folder) if f.endswith(valid_exts)]
    
    if not files:
        print("找不到圖片檔案。")
        return None

    for i, filename in enumerate(files):
        print(f"[{i}] {filename}")
    
    try:
        choice = input("請輸入檔案編號: ")
        idx = int(choice)
        selected_file = files[idx]
        
        # 關鍵修改：直接回傳檔名字串，不要在這裡讀取圖片
        return selected_file 
            
    except (ValueError, IndexError):
        print("選擇無效。")
        return None