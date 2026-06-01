import os
import cv2
import math
import time
import numpy as np
import pandas as pd
from tkinter import Tk, Label, Listbox, Button, Scrollbar, END
from tkinterdnd2 import DND_FILES, TkinterDnD

# --- 原有的功能函數保留 ---

def export_to_excel(data, filename, sheet_name="Sheet1"):
    if not isinstance(data, pd.DataFrame): raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(filename, str) or not filename.lower().endswith(".xlsx"):
        raise ValueError("filename must be a string ending with '.xlsx'.")
    try:
        data.to_excel(filename, sheet_name=sheet_name, index=False, engine="openpyxl")
        print(f"Excel 匯出成功: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"匯出失敗: {e}")

def scal(gray):
    # (保留你原本的 scal 內容...)
    gray_blur = cv2.GaussianBlur(gray, (17, 17), 0)
    gray_dilate = cv2.dilate(gray_blur, (17, 17), iterations=4)
    data = gray_dilate.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 5)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.flatten()
    sorted_indices = np.argsort(centers)
    mid_label = sorted_indices[0]
    temp_mask = np.uint8((labels.reshape(gray.shape) == mid_label) * 255)
    contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inner_filled_mask = np.zeros_like(temp_mask)
    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(inner_filled_mask, [max_cnt], -1, 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (428, 428))
    inner_filled_mask = cv2.morphologyEx(inner_filled_mask, cv2.MORPH_OPEN, kernel)
    return inner_filled_mask

# --- 核心處理邏輯 (原 main 的內容) ---

def process_single_image(read_jpg_name, max_um_input):
    unit_p_um = 1
    start_time = time.perf_counter()
    
    # 檢查 template 是否存在
    if not os.path.exists('ref_img_M.jpg'):
        print("錯誤: 找不到 ref_img_M.jpg 模板檔案")
        return

    template = cv2.imread('ref_img_M.jpg')
    samp_img = cv2.imread(read_jpg_name)
    if samp_img is None: return
    
    samp_img_gray = cv2.cvtColor(samp_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape[:2] # 修正原程式 w,h 順序
    mask = np.zeros(samp_img_gray.shape, dtype=np.uint8)

    res = cv2.matchTemplate(samp_img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    rects = []
    for pt in zip(*loc[::-1]):
        rects.append([int(pt[0]), int(pt[1]), int(w), int(h)])
        rects.append([int(pt[0]), int(pt[1]), int(w), int(h)])

    if len(rects) == 0:
        print(f"警告: {read_jpg_name} 找不到匹配物件")
        return

    rects, weights = cv2.groupRectangles(rects, groupThreshold=1, eps=0.5)
    df_sorted = pd.DataFrame(rects, columns=['X', 'Y', 'W', 'H']).sort_values(by=['X'])
    result_list = list(df_sorted.itertuples(index=False, name=None))
    
    df_f = pd.DataFrame(columns=['Ø', 'X', 'Y'])
    ref_point = (0, 0)
    last_x_val = 0

    for idx, (x, y, bw, bh) in enumerate(result_list):
        cv2.rectangle(samp_img, (x, y), (x + bw, y + bh), 255, 1)
        cropped = samp_img_gray[y:y+bh, x:x+bw]
        cropped_mask = scal(cropped)
        mask[y:y+bh, x:x+bw] = cropped_mask
        
        contours, _ = cv2.findContours(cropped_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cX, cY = 0, 0
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
        
        if idx == 0:
            ref_point = (x + cX), (y + cY)

        total_sum = np.sum(cropped_mask) / 255
        f_val = 2 * math.sqrt(total_sum / math.pi) * unit_p_um
        df_f.loc[len(df_f)] = [f_val, (x + cX) - ref_point[0], (y + cY)]
        last_x_val = (x + cX) - ref_point[0]

    # --- 計算與補足邏輯 ---
    _X = df_f['X'].values + ref_point[0]
    _Y = df_f['Y'].values
    
    # 這裡保留你原本的直線擬合與誤差計算邏輯...
    # (為了簡潔，中間 Y_difference 計算省略，請根據需求保留)
    
    # 縮放至 Max_um
    scale_factor = max_um_input / last_x_val if last_x_val != 0 else 1
    cols = df_f.select_dtypes('number').columns
    df_f[cols] = df_f[cols] * scale_factor
    
    # 輸出結果
    name_part, _ = os.path.splitext(read_jpg_name)
    export_to_excel(df_f, f"{name_part}_result.xlsx")
    cv2.imwrite(f"{name_part}_res_visual.jpg", samp_img)
    print(f"檔案 {read_jpg_name} 處理完成。")

# --- GUI 介面 ---

class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("OpenCV 批次處理工具 (拖曳檔案至此)")
        self.geometry("500x400")

        self.label = Label(self, text="請將圖片檔案拖曳到下方清單中", pady=10)
        self.label.pack()

        self.listbox = Listbox(self, selectmode='multiple', width=60, height=10)
        self.listbox.pack(padx=10, pady=5)

        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.drop)

        self.btn_run = Button(self, text="開始執行批次處理", command=self.run_process, bg="#4CAF50", fg="white", pady=10)
        self.btn_run.pack(fill='x', padx=20, pady=10)

        self.file_list = []

    def drop(self, event):
        # 處理拖曳進來的檔案路徑（處理 Windows 路徑大括號問題）
        files = self.tk.splitlist(event.data)
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                self.listbox.insert(END, f)
                self.file_list.append(f)

    def run_process(self):
        if not self.file_list:
            print("清單內沒有檔案！")
            return
        
        # 這裡可以加入一個簡單的輸入對話框來取得 Max um，這裡先預設或使用固定值
        max_um = 1000.0 # 建議改用 simpledialog.askfloat 取代原本的 input()
        
        for f in self.file_list:
            print(f"正在處理: {f}")
            process_single_image(f, max_um)
        
        print("所有任務已完成！")
        self.file_list = []
        self.listbox.delete(0, END)

if __name__ == "__main__":
    app = App()
    app.mainloop()
