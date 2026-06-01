import os
import cv2
import csv
import sys
import math
import time
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import messagebox
import select_image_from_current_dir


def export_to_excel(data_list, filename, sheet_name="Sheet1"):
    # 驗證檔名
    if not isinstance(filename, str) or not filename.lower().endswith(".csv"):
        raise ValueError("filename 必須是結尾為 '.csv' 的字串")

    # ✨ 自動轉換：如果傳進來的是 DataFrame，自動轉成 List of Dicts
    if isinstance(data_list, pd.DataFrame):
        data_list = data_list.to_dict(orient="records")

    # 驗證資料格式
    if not isinstance(data_list, list):
        raise TypeError("data_list 必須是包含字典(dict)的列表(list)")

    try:
        if getattr(sys, "frozen", False):
            current_dir = os.path.dirname(sys.executable)
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))

        full_path = os.path.join(current_dir, filename)

        if not data_list:
            print("⚠️ 沒有資料可以輸出")
            return

        # 擷取字典的 Key 作為表頭
        headers = data_list[0].keys()

        # 寫入 CSV 檔案（使用 utf-8-sig 防止 Excel 中文亂碼）
        with open(full_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data_list)

        print(f"✅ 檔案已成功輸出至: '{full_path}'")

    except Exception as e:
        print(f"❌ 輸出失敗: {e}")


def scal(gray):
    # 1. 讀取與預處理
    # img = cv2.imread('ref_img_C.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0) # 平滑處理減少雜訊
    gray = cv2.dilate(gray, (21, 21), iterations=3)
    # gray = cv2.dilate(gray, (9, 9), iterations=5)

    _, temp_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    temp_mask = cv2.bitwise_not(temp_mask)

    # 5. 填滿實心圓 (關鍵步驟)
    # 找到該區域的所有輪廓，並填滿最外層
    contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    inner_filled_mask = np.zeros_like(temp_mask)

    if contours:
        # 找到最大的輪廓（避免雜訊）並填滿
        max_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(inner_filled_mask, [max_cnt], -1, 255, -1) # -1 表示填充

    # 6. 套用遮罩得到結果
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (511, 511))
    # 執行開運算 (Opening)
    inner_filled_mask_ = inner_filled_mask
    inner_filled_mask = cv2.morphologyEx(inner_filled_mask, cv2.MORPH_OPEN, kernel)
    total_sum = cv2.countNonZero(inner_filled_mask)
    f = 2 * math.sqrt((total_sum) / math.pi) 
    i = 0
    eff = 0
    while f == 0 or f > 530:
        i = i+1
        try:
            kernel_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (155-eff, 155-eff))
            inner_filled_mask = cv2.morphologyEx(inner_filled_mask_, cv2.MORPH_CLOSE, kernel_)
        except:
            eff = 0
            kernel_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (150-eff, 150-eff))
            inner_filled_mask = cv2.morphologyEx(inner_filled_mask_, cv2.MORPH_CLOSE, kernel_)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (511-i, 511-i))
        # 執行開運算 (Opening)
        inner_filled_mask = cv2.morphologyEx(inner_filled_mask, cv2.MORPH_OPEN, kernel)
        total_sum = cv2.countNonZero(inner_filled_mask)
        f = 2 * math.sqrt((total_sum) / math.pi)
        print(f"Retry...{i}__{f}__{eff}")
        if f < 520 and f != 0:
            break
        if f > 520:
            eff = eff+40

    result = cv2.bitwise_and(gray, gray, mask=inner_filled_mask)
    cv2.namedWindow('Filled Inner Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Final Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Filled Inner Mask', inner_filled_mask)
    cv2.imshow('Final Result', result)
    cv2.waitKey(10)

    return inner_filled_mask

def main(path, threshold_val):
    unit_p_um = 1# 0.2451592491367
    # read_jpg_name = select_image_from_current_dir.select_image_from_current_dir()
    read_jpg_name = path
    start_time = time.perf_counter()
    template = cv2.imread('ref_img_M2.jpg')
    samp_img = cv2.imread(read_jpg_name)
    samp_img_gray = cv2.cvtColor(samp_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape
    mask = np.zeros(samp_img_gray.shape, dtype=np.uint8)

    res = cv2.matchTemplate(samp_img_gray, template_gray, cv2.TM_CCOEFF_NORMED) #匹配
    threshold = 0.85 #匹配相似度0.8 = 80%
    loc = np.where(res >= threshold) #過濾大於相似度的區塊

    # rects = []
    # for pt in zip(*loc[::-1]):
    #     rects.append([int(pt[0]), int(pt[1]), int(w), int(h)])
    #     rects.append([int(pt[0]), int(pt[1]), int(w), int(h)])
    rects = [
        [int(pt[0]), int(pt[1]), int(w), int(h)]
        for pt in zip(*loc[::-1])
        for _ in range(2)
    ]
    rects, weights = cv2.groupRectangles(rects, groupThreshold=2, eps=0.7)
    df = pd.DataFrame(rects, columns=['X', 'Y', 'W', 'H'])
    # 排序（X 降序）
    df_sorted = df.sort_values(by=['X'], ascending=[True])
    # df_sorted['group'] = df_sorted.index // 12  # 分組編號
    df_final = df_sorted.sort_values(by=['X'], ascending=[ True])
    result = list(df_final[['X', 'Y', 'W', 'H']].itertuples(index=False, name=None))
    df_f = pd.DataFrame( columns=['\u00D8', 'X', 'Y'])

    i = 0
    last = 0
    for (x, y, bw, bh) in result:
        i = i+1
        bh = bh+40
        bw = bw-90
        # cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
        cv2.rectangle(samp_img, (x, y), (x + bw, y + bh), 255, 1)

        # print(x, y, bw, bh)
        # cv2.namedWindow("samp_img", cv2.WINDOW_NORMAL)
        # cv2.imshow("samp_img",samp_img)
        cropped = samp_img_gray[y:y+bh, x:x+bw]
        cropped_mask = scal(cropped)
        mask[y:y+bh, x:x+bw] = cropped_mask
        contours, hie = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   #get 輪廓

        for cnt in contours:
            # 1. 計算輪廓的矩 (Moments)
            M = cv2.moments(cnt)

            # 2. 計算中心點座標 (cX, cY)
            # 公式：cX = M10/M00, cY = M01/M00
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
            (cX, cY), axes, angle = cv2.fitEllipseAMS(cnt)
            cv2.circle(samp_img, (int(x + cX), int(y + cY)), 1, 255, -1)
            if len(df_f) == 0:
                ref_point = (x + cX) * unit_p_um, (y + cY) *unit_p_um

            total_sum = cv2.countNonZero(cropped_mask)
            f = 2 * math.sqrt((total_sum) / math.pi) * unit_p_um 
            print(f'{i}__{f}')
            tk._default_root.update_idletasks()
            df_f.loc[len(df_f)+1] = [f, (x + cX) * unit_p_um - ref_point[0], (y + cY) *unit_p_um]
            last = x + cX - ref_point[0]
        # print(total_sum)
        # cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
        # cv2.imshow("cropped",samp_img_gray)
        # cv2.waitKey(0)

    if i == 36:
        _X = df_f['X'].values+ref_point[0]
        _Y = df_f['Y'].values
        m, b = np.polyfit(_X, _Y, 1)
        df_f['Y_'] = m * _X + b
        df_f['1-36_Y_difference'] = df_f['Y_'] - df_f['Y']

        p1_idx, p2_idx = 2, 33 
        x1, y1 = _X[p1_idx], _Y[p1_idx]
        x2, y2 = _X[p2_idx], _Y[p2_idx]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        df_f['Y_'] = m * _X + b
        df_f['3&34_Y_difference'] = df_f['Y_'] - df_f['Y']

        p1_idx, p2_idx = 0, 35 
        x1, y1 = _X[p1_idx], _Y[p1_idx]
        x2, y2 = _X[p2_idx], _Y[p2_idx]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        df_f['Y_'] = m * _X + b
        df_f['1&36_Y_difference'] = df_f['Y_'] - df_f['Y']

        # Max_um = float(input ('enter Max um: '))
        Max_um = float(threshold_val)

        cols = df_f.select_dtypes('number').columns
        df_f[cols] = df_f[cols] * Max_um / last
        df_f['X_difference'] = (np.arange(len(df)) * 127) - df_f['X'] 

        df_f['1-36 pitch error'] = np.sqrt((df_f['X_difference'] * df_f['X_difference']) + (df_f['1-36_Y_difference'] * df_f['1-36_Y_difference']))
        df_f['3&34 pitch error'] = np.sqrt((df_f['X_difference'] * df_f['X_difference']) + (df_f['3&34_Y_difference'] * df_f['3&34_Y_difference']))
        df_f['1&36 pitch error'] = np.sqrt((df_f['X_difference'] * df_f['X_difference']) + (df_f['1&36_Y_difference'] * df_f['1&36_Y_difference']))

        output_df = df_f[['\u00D8', 'X','X_difference', '1&36_Y_difference', '1-36_Y_difference','3&34_Y_difference', '1&36 pitch error', '1-36 pitch error', '3&34 pitch error']]

        print(output_df)
        print(Max_um / last)

        masked_img = cv2.bitwise_and(samp_img_gray, samp_img_gray, mask=mask)
        name_part, _ = os.path.splitext(read_jpg_name)
        output_xlsx = f"{name_part}_result.csv"
        export_to_excel(output_df, output_xlsx, sheet_name=output_xlsx)

    else:
        _X = df_f['X'].values+ref_point[0]
        _Y = df_f['Y'].values
        m, b = np.polyfit(_X, _Y, 1)
        df_f['Y_'] = m * _X + b
        df_f['1-36_Y_difference'] = df_f['Y_'] - df_f['Y']

        p1_idx, p2_idx = 2, i-3
        x1, y1 = _X[p1_idx], _Y[p1_idx]
        x2, y2 = _X[p2_idx], _Y[p2_idx]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        df_f['Y_'] = m * _X + b
        df_f['3&34_Y_difference'] = df_f['Y_'] - df_f['Y']

        p1_idx, p2_idx = 0, i-1
        x1, y1 = _X[p1_idx], _Y[p1_idx]
        x2, y2 = _X[p2_idx], _Y[p2_idx]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        df_f['Y_'] = m * _X + b
        df_f['1&36_Y_difference'] = df_f['Y_'] - df_f['Y']

        cols = df_f.select_dtypes('number').columns
        # Max_um = float(input ('enter Max um: '))
        Max_um = float(threshold_val)
        df_f[cols] = df_f[cols] * Max_um / last
        df_f['pitch_idx'] = np.round(df_f['X'] / 127).astype(int) + 1

        # 2. 建立完整的 Pitch 模板 (從 1 到最大編號)
        max_idx = df_f['pitch_idx'].max()
        df_full = pd.DataFrame({'pitch_idx': np.arange(1, max_idx + 1)})

        # 3. 將辨識資料 df_f 合併到完整模板中
        df_f = pd.merge(df_full, df_f, on='pitch_idx', how='left')

        # 4. 重新計算 X_difference，基準為 (pitch 編號 - 1) * 127
        df_f['X_difference'] = df_f['X'] - ((df_f['pitch_idx'] - 1) * 127)

        # 5. 計算各項 pitch error (NaN 自動留空)
        df_f['1-36 pitch error'] = np.sqrt((df_f['X_difference']**2) + (df_f['1-36_Y_difference']**2))
        df_f['3&34 pitch error'] = np.sqrt((df_f['X_difference']**2) + (df_f['3&34_Y_difference']**2))
        df_f['1&36 pitch error'] = np.sqrt((df_f['X_difference']**2) + (df_f['1&36_Y_difference']**2))

        # 6. 輸出最終結果
        output_df = df_f[['\u00D8', 'X','X_difference', '1-36_Y_difference', '1&36_Y_difference','3&34_Y_difference', '1-36 pitch error', '1&36 pitch error', '3&34 pitch error']]        
        print(output_df)
        print(Max_um / last)

        name_part, _ = os.path.splitext(read_jpg_name)
        output_xlsx = f"{name_part}_result.csv"
        export_to_excel(output_df, output_xlsx,)
        print('detact error')

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"程式執行耗時：{execution_time:.4f} 秒")

    # cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("masked_img", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("samp_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("cropped",masked_img)
    # cv2.imshow("masked_img",masked_img)
    # cv2.imshow("samp_img",samp_img)
    cv2.imwrite(f"{name_part}_res.jpg",samp_img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()


class AOICore:
    def __init__(self, root):
        self.root = root
        self.root.title("AOI 影像")
        self.root.geometry("700x800")
        
        # 儲存資料 {檔案路徑: Entry物件}
        self.image_items = {}
        self.setup_ui()

    def setup_ui(self):
        # 1. 拖曳區域 (維持不變)
        self.drop_label = tk.Label(self.root, text="將圖檔拖曳至此", 
                                  bg="#f0f0f0", height=12, relief="ridge")
        self.drop_label.pack(fill=tk.X, padx=10, pady=10)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.handle_drop)

        # 2. 滾動列表 (維持不變)
        self.list_frame = tk.Frame(self.root)
        self.list_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        self.canvas = tk.Canvas(self.list_frame)
        self.scrollbar = ttk.Scrollbar(self.list_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_content = tk.Frame(self.canvas)

        self.scrollable_content.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # 3. 按鈕區域 (新增清除按鈕)
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(pady=10)

        self.btn_process = tk.Button(self.btn_frame, text="執行", 
                                    command=self.process_images, bg="#4CAF50", fg="white", width=15)
        self.btn_process.pack(side=tk.LEFT, padx=5)

        # 新增：清除按鈕
        self.btn_clear = tk.Button(self.btn_frame, text="清除清單", 
                                  command=self.clear_list, bg="#f44336", fg="white", width=15)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

    def clear_list(self):
        """清除所有已加入的圖片項目"""
        # 刪除 UI 上的元件
        for widget in self.scrollable_content.winfo_children():
            widget.destroy()
        
        # 清空資料字典
        self.image_items.clear()
        
        # 重置捲軸位置到最上方
        self.canvas.yview_moveto(0)
        print("清單已清除")

    def handle_drop(self, event):
        # 解析路徑
        files = self.root.tk.splitlist(event.data)
        for f_path in files:
            if f_path.lower().endswith(('.jpg', '.png', '.bmp')) and f_path not in self.image_items:
                self.add_to_ui(f_path)

    def add_to_ui(self, f_path):
        row = tk.Frame(self.scrollable_content, pady=2)
        row.pack(fill=tk.X)
        
        lbl = tk.Label(row, text=os.path.basename(f_path), width=40, anchor="w")
        lbl.pack(side=tk.LEFT)
        
        ent = tk.Entry(row, width=25)
        ent.insert(0, "4445") 
        ent.pack(side=tk.RIGHT, padx=5)
        
        self.image_items[f_path] = ent

    def process_images(self):
        """核心：將路徑丟入 OpenCV"""
        for path, entry_box in self.image_items.items():
            threshold_val = entry_box.get()
            main(path, threshold_val)

        messagebox.showinfo("完成", f"已成功處理 {len(self.image_items)} 張影像")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = AOICore(root)
    root.mainloop()
