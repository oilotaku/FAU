import os
import cv2
import csv
import sys
import math
import time
import logging
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import messagebox
import suport_Cxx_data as Cdata
import Cxx_template_gray
import Cxx_template_

# 在程式最開頭進行基本設定（這段通常放在檔案最上方，或 __init__ 中）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
cv2.setNumThreads(0)
os.environ["OPENCV_IPP_DISABLE"] = "1"

def select_image_from_current_dir():
    current_folder = os.getcwd()
    valid_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG")
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

        return selected_file

    except (ValueError, IndexError):
        print("選擇無效。")
        return None


def export_to_excel(data, filename, sheet_name="Sheet1"):
    # Validate DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    # Validate filename
    if not isinstance(filename, str) or not filename.lower().endswith(".xlsx"):
        raise ValueError("filename must be a string ending with '.xlsx'.")

    try:
        # Export DataFrame to Excel
        data.to_excel(filename, sheet_name=sheet_name, index=False, engine="openpyxl")
        print(f"✅ Data successfully exported to '{os.path.abspath(filename)}'")
    except Exception as e:
        print(f"❌ Failed to export to Excel: {e}")


def export_to_csv(data_list, filename="output.csv"):
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
            logging.warning("⚠️ 沒有資料可以輸出")
            return

        # 擷取字典的 Key 作為表頭
        headers = data_list[0].keys()

        # 寫入 CSV 檔案（使用 utf-8-sig 防止 Excel 中文亂碼）
        with open(full_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data_list)
        logging.info(f"檔案已成功輸出至: '{full_path}'")

    except Exception as e:
        logging.warning(f"輸出失敗: {e}")


def data_36(df_measured, df_ref):
    final_list = []
    max_error = 100  # 允許實測點與基準點的 X 軸對齊誤差

    for idx, ref_row in df_ref.iterrows():
        target_cx_ = ref_row["CX"]

        # 計算第二組中所有的 X 與當前基準點的距離
        diffs = np.abs(df_measured["X"] - target_cx_)
        best_match_idx = diffs.idxmin()

        # 如果在誤差內找到配對，直接加入原本的實測資料
        if diffs[best_match_idx] <= max_error:
            matched_data = df_measured.loc[best_match_idx].to_dict()
            matched_data["Is_Filled"] = False  # 標記為原始實測資料
            final_list.append(matched_data)
        else:
            # 🚩 沒找到配對：代表這坑位空了，直接把第一組的數據補進去
            # print(f"🚩 偵測到缺項！在基準 CX_={target_cx_} 處補入數據，並標記為 Is_Filled")
            try:
                final_list.append(
                    {
                        "Ø": 0.0,  # 缺項直徑給 0
                        "X": target_cx_,  # 把第一組的 CX_ 補進第二組的 X
                        "Y": ref_row["CY"],  # 把第一組的 CY 補進第二組的 Y
                        "Is_Filled": True,  # 標記這是補數據的地方
                    }
                )
            except:
                pass

    # 4. 重新封裝回 DataFrame 並重整索引（1 到 36）
    try:
        df_final = pd.DataFrame(final_list)
        df_final.index = range(1, len(df_final) + 1)
        pd.set_option("display.max_rows", 40)
        return df_final
    except:
        return df_measured


def scal(gray):
    # 1. 讀取與預處理
    # img = cv2.imdecode('ref_img_C.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # gray = cv2.dilate(gray, (7, 7), iterations=4)
    # gray = cv2.dilate(gray, (9, 9), iterations=5)

    data = gray.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.000001)
    _, labels, centers = cv2.kmeans(
        data, 3, None, criteria, 200, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = centers.flatten()
    sorted_indices = np.argsort(centers)  # [最深, 中等, 最亮]
    mid_label = sorted_indices[2]  # 取得中等亮度的標籤

    temp_mask = np.uint8((labels.reshape(gray.shape) == mid_label) * 255)

    contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    inner_filled_mask = np.zeros_like(temp_mask)
    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(inner_filled_mask, [max_cnt], -1, 255, -1)  # -1 表示填充

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    inner_filled_mask_ = inner_filled_mask
    inner_filled_mask = cv2.morphologyEx(inner_filled_mask, cv2.MORPH_OPEN, kernel)
    total_sum = cv2.countNonZero(inner_filled_mask)

    # if total_sum == 0:
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    #     inner_filled_mask = cv2.morphologyEx(inner_filled_mask_, cv2.MORPH_OPEN, kernel)

    # 6. 套用遮罩得到結果
    # result = cv2.bitwise_and(gray, gray, mask=inner_filled_mask)
    # cv2.namedWindow('Filled Inner Mask', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Final Result', cv2.WINDOW_NORMAL)
    # cv2.imshow('Filled Inner Mask', inner_filled_mask)
    # cv2.imshow('Final Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return inner_filled_mask


def main_Cxx(path, threshold_val):

    template_gray = Cxx_template_gray.get_template_gray()
    template_ = Cxx_template_.get_template_()

    unit_p_um = 1  # 0.2451592491367
    read_jpg_name = path

    samp_img = cv2.imdecode(
        np.fromfile(read_jpg_name, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    samp_img_gray = cv2.cvtColor(samp_img, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape
    _w, _h = template_.shape
    mask = np.zeros(samp_img_gray.shape, dtype=np.uint8)
    copy = samp_img_gray.copy()

    samp_img_gray = cv2.GaussianBlur(samp_img_gray, (17,17), 0)

    res = cv2.matchTemplate(samp_img_gray, template_, cv2.TM_CCOEFF_NORMED)
    threshold = 0.2
    loc = np.where(res >= threshold)

    rects = [
        [int(pt[0]), int(pt[1]), int(w), int(h)]
        for pt in zip(*loc[::-1])
        for _ in range(2)
    ]
    rects, weights = cv2.groupRectangles(rects, groupThreshold=1, eps=0.2)
    for x, y, bw, bh in rects:
        try:
            samp_img_gray[y : y + 400, x : x + 300] = 0
        except:
            pass

    # cv2.namedWindow("samp_img", cv2.WINDOW_NORMAL)
    # cv2.imshow('samp_img', samp_img_gray)
    # cv2.waitKey(0)
    logging.info(f"匹配物件")
    res = cv2.matchTemplate(samp_img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    threshold = 0.72
    loc = np.where(res >= threshold)

    rects = [
        [int(pt[0]), int(pt[1]), int(w), int(h)]
        for pt in zip(*loc[::-1])
        for _ in range(2)
    ]
    rects, weights = cv2.groupRectangles(rects, groupThreshold=2, eps=0.7)

    df = pd.DataFrame(rects, columns=["X", "Y", "W", "H"])

    if not df.empty:
        y_median = df["Y"].median()  # 找出 Y 的中位數
        allowed_error = 120
        df = df[abs(df["Y"] - y_median) < allowed_error]

    # df = pd.DataFrame(rects, columns=['X', 'Y', 'W', 'H'])
    # 排序（Y 降序）
    df_sorted = df.sort_values(by=["X"], ascending=[True]).reset_index(drop=True)
    # df_sorted['group'] = df_sorted.index // 12  # 分組編號
    min_distance = 400

    i = 0
    while i < len(df_sorted) - 1:
        current_x = df_sorted.loc[i, "X"]
        next_x = df_sorted.loc[i + 1, "X"]

        if (next_x - current_x) < min_distance:
            df_sorted = df_sorted.drop(i + 1).reset_index(drop=True)
            # print(i)
        else:
            i += 1

    df_final = df_sorted.sort_values(by=["X"], ascending=[True])
    result = list(df_final[["X", "Y", "W", "H"]].itertuples(index=False, name=None))
    df_f = pd.DataFrame(columns=["\u00d8", "X", "Y"])
    # 3. 畫出合併後的框 =
    i = 0
    last = 0

    for x, y, bw, bh in result:
        # print(i)
        # cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
        cv2.rectangle(samp_img, (x, y), (x + bw, y + bh), 255, 1)

        # print(x, y, bw, bh)
        # cv2.namedWindow("samp_img", cv2.WINDOW_NORMAL)
        # cv2.imshow("samp_img",samp_img)
        cropped = samp_img_gray[y : y + bh, x : x + bw]
        cropped_mask = scal(cropped)
        mask[y : y + bh, x : x + bw] = cropped_mask
        contours, hie = cv2.findContours(
            cropped_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )  # get 輪廓

        for cnt in contours:
            # 1. 計算輪廓的矩 (Moments)
            # M = cv2.moments(cnt)

            # if M["m00"] != 0:
            #     cX = M["m10"] / M["m00"]
            #     cY = M["m01"] / M["m00"]

            (cX, cY), axes, angle = cv2.fitEllipseAMS(cnt)
            cv2.circle(samp_img, (int(x + cX), int(y + cY)), 1, 255, -1)
            # center_point = find_subpixel_circle_center(cropped_mask, x, y)
            # cX, cY = center_point
            if len(df_f) == 0:
                ref_point = (x + cX) * unit_p_um, (y + cY) * unit_p_um

            total_sum = cv2.countNonZero(cropped_mask)

            f = 2 * math.sqrt((total_sum) / math.pi) * unit_p_um
            df_f.loc[len(df_f) + 1] = [
                f,
                (x + cX) * unit_p_um - ref_point[0],
                (y + cY) * unit_p_um,
            ]
            last = x + cX - ref_point[0]
            i = i + 1

        # print(df_f)
        # cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
        # cv2.imshow("cropped",samp_img_gray)
        # cv2.waitKey(0)

    if len(df_f) == 36:
        logging.info(f"處理辨識物件數量正確")
        _X = df_f["X"].values + ref_point[0]
        _Y = df_f["Y"].values
        m, b = np.polyfit(_X, _Y, 1)
        df_f["Y_"] = m * _X + b
        df_f["1-36_Y_difference"] = df_f["Y_"] - df_f["Y"]

        p1_idx, p2_idx = 2, 33
        x1, y1 = _X[p1_idx], _Y[p1_idx]
        x2, y2 = _X[p2_idx], _Y[p2_idx]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        df_f["Y_"] = m * _X + b
        df_f["3&34_Y_difference"] = df_f["Y_"] - df_f["Y"]

        p1_idx, p2_idx = 0, 35
        x1, y1 = _X[p1_idx], _Y[p1_idx]
        x2, y2 = _X[p2_idx], _Y[p2_idx]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        df_f["Y_"] = m * _X + b
        df_f["1&36_Y_difference"] = df_f["Y_"] - df_f["Y"]

        cols = df_f.select_dtypes("number").columns
        Max_um = float(threshold_val)
        df_f[cols] = df_f[cols] * Max_um / last
        df_f["X_difference"] = df_f["X"] - (np.arange(len(df_f)) * 127)

        df_f["1-36 pitch error"] = np.sqrt(
            (df_f["X_difference"] * df_f["X_difference"])
            + (df_f["1-36_Y_difference"] * df_f["1-36_Y_difference"])
        )
        df_f["3&34 pitch error"] = np.sqrt(
            (df_f["X_difference"] * df_f["X_difference"])
            + (df_f["3&34_Y_difference"] * df_f["3&34_Y_difference"])
        )
        df_f["1&36 pitch error"] = np.sqrt(
            (df_f["X_difference"] * df_f["X_difference"])
            + (df_f["1&36_Y_difference"] * df_f["1&36_Y_difference"])
        )

        output_df = df_f[
            [
                "\u00d8",
                "X",
                "X_difference",
                "1&36_Y_difference",
                "1-36_Y_difference",
                "3&34_Y_difference",
                "1&36 pitch error",
                "1-36 pitch error",
                "3&34 pitch error",
            ]
        ]

        # print(output_df)
        # print(Max_um / last)

        name_part, _ = os.path.splitext(read_jpg_name)
        output_xlsx = f"{name_part}_result.xlsx"
        output_csv = f"{name_part}_result.csv"
        # export_to_excel(output_df, output_xlsx)
        # export_to_csv(output_df, output_csv)

    elif len(df_f) < 36:
        try:
            logging.info(f"辨識物件數量不足嘗試補齊")
            cdata = Cdata.main(copy)
            # print(cdata)
            # print(df_f)
            df_f = data_36(df_f, cdata)
            logging.info(f"補齊後數量{len(df_f)}")
        except:
            pass

        _X = df_f["X"].values + ref_point[0]
        _Y = df_f["Y"].values
        m, b = np.polyfit(_X, _Y, 1)
        df_f["Y_"] = m * _X + b
        df_f["1-36_Y_difference"] = df_f["Y_"] - df_f["Y"]

        try:
            p1_idx, p2_idx = 2, 33
            x1, y1 = _X[p1_idx], _Y[p1_idx]
            x2, y2 = _X[p2_idx], _Y[p2_idx]
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            df_f["Y_"] = m * _X + b
            df_f["3&34_Y_difference"] = df_f["Y_"] - df_f["Y"]
        except:
            p1_idx, p2_idx = 2, len(_Y)-2
            x1, y1 = _X[p1_idx], _Y[p1_idx]
            x2, y2 = _X[p2_idx], _Y[p2_idx]
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            df_f["Y_"] = m * _X + b
            df_f["3&34_Y_difference"] = df_f["Y_"] - df_f["Y"]

        try:
            p1_idx, p2_idx = 0, 35
            x1, y1 = _X[p1_idx], _Y[p1_idx]
            x2, y2 = _X[p2_idx], _Y[p2_idx]
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            df_f["Y_"] = m * _X + b
            df_f["1&36_Y_difference"] = df_f["Y_"] - df_f["Y"]
        except:
            p1_idx, p2_idx = 0, len(_Y)-1
            x1, y1 = _X[p1_idx], _Y[p1_idx]
            x2, y2 = _X[p2_idx], _Y[p2_idx]
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            df_f["Y_"] = m * _X + b
            df_f["1&36_Y_difference"] = df_f["Y_"] - df_f["Y"]

        cols = df_f.select_dtypes("number").columns
        Max_um = float(threshold_val)
        df_f[cols] = df_f[cols] * Max_um / last
        df_f["pitch_idx"] = np.round(df_f["X"] / 127).astype(int) + 1

        # 2. 建立完整的 Pitch 模板 (從 1 到最大編號)
        max_idx = df_f["pitch_idx"].max()
        df_full = pd.DataFrame({"pitch_idx": np.arange(1, max_idx + 1)})

        # 3. 將辨識資料 df_f 合併到完整模板中
        df_f = pd.merge(df_full, df_f, on="pitch_idx", how="left")

        # 4. 重新計算 X_difference，基準為 (pitch 編號 - 1) * 127
        df_f["X_difference"] = df_f["X"] - ((df_f["pitch_idx"] - 1) * 127)

        # 5. 計算各項 pitch error (NaN 自動留空)
        df_f["1-36 pitch error"] = np.sqrt(
            (df_f["X_difference"] ** 2) + (df_f["1-36_Y_difference"] ** 2)
        )
        df_f["3&34 pitch error"] = np.sqrt(
            (df_f["X_difference"] ** 2) + (df_f["3&34_Y_difference"] ** 2)
        )
        df_f["1&36 pitch error"] = np.sqrt(
            (df_f["X_difference"] ** 2) + (df_f["1&36_Y_difference"] ** 2)
        )

        # 6. 輸出最終結果
        output_df = df_f[
            [
                "\u00d8",
                "X",
                "X_difference",
                "1&36_Y_difference",
                "1-36_Y_difference",
                "3&34_Y_difference",
                "1&36 pitch error",
                "1-36 pitch error",
                "3&34 pitch error",
            ]
        ]
        # print(output_df)
        # print(Max_um / last)

        name_part, _ = os.path.splitext(read_jpg_name)
        output_xlsx = f"{name_part}_result.xlsx"
        output_csv = f"{name_part}_result.csv"

        # export_to_excel(output_df, output_xlsx)
        # export_to_csv(output_df, output_csv)
        # print("detact error")

    name_part, _ = os.path.splitext(read_jpg_name)
    # masked_img = cv2.bitwise_and(samp_img_gray, samp_img_gray, mask=mask)
    # cv2.namedWindow("samp_img", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("masked_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("cropped",masked_img)
    # cv2.imshow("masked_img",masked_img)
    # cv2.imshow("samp_img",samp_img)
    # cv2.imwrite(f"{name_part}_res.jpg",samp_img)
    save_path = f"{name_part}_res.jpg"
    success, encoded_img = cv2.imencode(".jpg", samp_img)

    if success:
        # 3. 使用 Python 內建的 tofile 寫入檔案（100% 支援全中文與亂碼路徑）
        encoded_img.tofile(save_path)
        # print(f" 圖片已成功儲存至：{save_path}")
    else:
        print(" 圖片編碼失敗")

    return (output_df, output_csv)


def show_auto_close_info(title, message, delay=2000):
    """顯示一個會自動關閉的提示視窗
    delay: 毫秒數，2000 代表 2 秒
    """
    # 建立一個置頂的子視窗
    popup = tk.Toplevel()
    popup.title(title)
    popup.attributes("-topmost", True)  # 確保視窗顯示在最上層

    # 視窗內顯示的文字
    label = tk.Label(
        popup, text=message, font=("Microsoft JhengHei", 11), padx=20, pady=20
    )
    label.pack()

    # 計算並設定視窗位置（使其居中顯示，可依需求調整）
    popup.update_idletasks()
    x = (popup.winfo_screenwidth() - popup.winfo_reqwidth()) // 2
    y = (popup.winfo_screenheight() - popup.winfo_reqheight()) // 2
    popup.geometry(f"+{x}+{y}")

    # 核心：到達指定時間後自動銷毀該視窗
    popup.after(delay, popup.destroy)


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
        self.drop_label = tk.Label(
            self.root, text="將圖檔拖曳至此", bg="#f0f0f0", height=12, relief="ridge"
        )
        self.drop_label.pack(fill=tk.X, padx=10, pady=10)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self.handle_drop)

        # 2. 滾動列表 (維持不變)
        self.list_frame = tk.Frame(self.root)
        self.list_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        self.canvas = tk.Canvas(self.list_frame)
        self.scrollbar = ttk.Scrollbar(
            self.list_frame, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_content = tk.Frame(self.canvas)

        self.scrollable_content.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.scrollable_content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # 3. 按鈕區域 (新增清除按鈕)
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(pady=10)

        self.btn_process = tk.Button(
            self.btn_frame,
            text="執行",
            command=self.process_images,
            bg="#4CAF50",
            fg="white",
            width=15,
        )
        self.btn_process.pack(side=tk.LEFT, padx=5)

        # 新增：清除按鈕
        self.btn_clear = tk.Button(
            self.btn_frame,
            text="清除清單",
            command=self.clear_list,
            bg="#f44336",
            fg="white",
            width=15,
        )
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
            if (
                f_path.lower().endswith((".jpg", ".png", ".bmp"))
                and f_path not in self.image_items
            ):
                self.add_to_ui(f_path)

    def add_to_ui(self, f_path):
        row = tk.Frame(self.scrollable_content, pady=2)
        row.pack(fill=tk.X)

        lbl = tk.Label(row, text=os.path.basename(f_path), width=40, anchor="w")
        lbl.pack(side=tk.LEFT)

        ent = tk.Entry(row, width=10)
        ent.insert(0, "4445")
        ent.pack(side=tk.RIGHT, padx=5)

        self.image_items[f_path] = ent

    def process_images(self):
        """將路徑丟入 OpenCV"""
        start_time = time.perf_counter()
        for path, entry_box in self.image_items.items():
            threshold_val = entry_box.get()
            try:
                logging.info(f"處理影像 {path} 中......")
                output_df, output_csv = main_Cxx(path, threshold_val)
                export_to_csv(output_df, output_csv)
            except:
                logging.warning(f"影像 {path} 辨識失敗......")

        # self.status_var.set("執行完成！")
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        logging.info(
            f"處理 {len(self.image_items)} 張影像, 程式執行耗時：{execution_time:.4f} 秒"
        )
        messagebox.showinfo(
            "完成",
            f"處理 {len(self.image_items)} 張影像, 程式執行耗時：{execution_time:.4f} 秒",
        )
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = AOICore(root)
    root.mainloop()
