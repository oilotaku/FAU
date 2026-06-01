import os
import subprocess
import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from datetime import datetime

# --- 核心數據比對邏輯：100% 嚴格還原原始腳本算法 ---
def run_comparison(file_list, anomaly_threshold=0.1):
    df_dict = {}
    for i, f in enumerate(file_list):
        file_label = f"F{i+1}"
        file_ext = os.path.splitext(f)[1].lower()

        # 根據副檔名進行分流讀取
        if file_ext in [".xlsx", ".xls"]:
            try:
                df_dict[file_label] = pd.read_excel(f)
            except Exception as e:
                raise ValueError(
                    f"❌ Excel 檔案【{os.path.basename(f)}】無法讀取，可能檔案損壞或被佔用。\n原因: {str(e)}"
                )
        else:
            # CSV 檔案格式讀取 (支援中文路徑與多種編碼)
            for encoding in ["utf-8-sig", "utf-8", "cp950", "gbk"]:
                try:
                    with open(f, mode="r", encoding=encoding) as fh:
                        df_dict[file_label] = pd.read_csv(fh)
                    break
                except (UnicodeDecodeError, PermissionError):
                    continue
            else:
                raise ValueError(f"❌ CSV 檔案【{os.path.basename(f)}】無法讀取。")
    # 取得基礎變數
    f1_df = df_dict["F1"]
    f2_df = df_dict["F2"]
    f3_df = df_dict["F3"]

    exclude_cols = ["X", "Ø", "Source"]
    data_cols = [col for col in f1_df.columns if col not in exclude_cols]

    all_summaries = []
    # 100% 還原原本的明細開頭結構
    detailed_error_chunks = [f1_df[["X"]].copy()]
    anomaly_records = []

    # 核心迴圈：逐一深度剖析每個欄位 (完全複製您原本的數學公式與變數)
    for col in data_cols:
        f1, f2, f3 = f1_df[col], f2_df[col], f3_df[col]

        # --- A. 點對點基礎與進階微觀誤差計算 ---
        diff_12 = (f1 - f2).abs()
        diff_23 = (f2 - f3).abs()
        diff_13 = (f1 - f3).abs()

        # 三檔橫向統計 (嚴格採用無 Keys 的列表合併，確保 axis=1 計算完全正確)
        combined = pd.concat([f1, f2, f3], axis=1)
        mean_val = combined.mean(axis=1)
        std_val = combined.std(axis=1)
        max_range = combined.max(axis=1) - combined.min(axis=1)

        # 變異係數 CV%
        cv_percent = np.where(
            mean_val.abs() > 0.05, (std_val / mean_val).abs() * 100, np.nan
        )

        # 彙整此欄位的微觀明細 (完全還原原本的欄位名稱與順序)
        col_df = pd.DataFrame(
            {
                f"{col}_F1_Value": f1,
                f"{col}_F2_Value": f2,
                f"{col}_F3_Value": f3,
                f"{col}_Diff_F1_F2": diff_12,
                f"{col}_Diff_F2_F3": diff_23,
                f"{col}_Diff_F1_F3": diff_13,
                f"{col}_Max_Range": max_range,
                f"{col}_CV%": cv_percent,
            }
        )
        detailed_error_chunks.append(col_df)

        # --- B. 尋找該欄位的最大誤差極端點位 ---
        max_range_idx = max_range.idxmax()
        max_x_location = f1_df.loc[max_range_idx, "X"]
        worst_f1 = f1.loc[max_range_idx]
        worst_f2 = f2.loc[max_range_idx]
        worst_f3 = f3.loc[max_range_idx]

        # --- C. 篩選超出工程門檻的異常點 ---
        anomalies = f1_df[max_range > anomaly_threshold]["X"].tolist()
        anomaly_count = len(anomalies)

        # --- D. 收集全域總體統計摘要 ---
        all_summaries.append(
            {
                "數據欄位": col,
                "F1與F2平均差": diff_12.mean(),
                "F2與F3平均差": diff_23.mean(),
                "F1與F3平均差": diff_13.mean(),
                "平均最大波動": max_range.mean(),
                "誤差中位數(50%)": max_range.median(),
                "高變動區落點(75%)": max_range.quantile(0.75),
                "極端誤差": max_range.max(),
                "最大誤差發生 X 座標": max_x_location,
                "最大誤差點數值(F1/F2/F3)": f"{worst_f1:.4f} / {worst_f2:.4f} / {worst_f3:.4f}",
                f"超標點數(>{anomaly_threshold})": anomaly_count,
            }
        )

        # 記錄異常清單
        for idx in f1_df[max_range > anomaly_threshold].index:
            anomaly_records.append(
                {
                    "異常欄位": col,
                    "X座標": f1_df.loc[idx, "X"],
                    "File_1 數值": f1.loc[idx],
                    "File_2 數值": f2.loc[idx],
                    "File_3 數值": f3.loc[idx],
                    "當點最大波動": max_range.loc[idx],
                }
            )

    # 4. 資料集大整合
    summary_df = pd.DataFrame(all_summaries)
    detailed_df = pd.concat(detailed_error_chunks, axis=1)
    anomaly_df = (
        pd.DataFrame(anomaly_records)
        if anomaly_records
        else pd.DataFrame(columns=["提示"])
    )

    # 5. 自動輸出高階多頁 Excel 報告
    current_time = datetime.now().strftime("%m%d_%H%M")
    output_filename = f"{current_time}_三個檔案全欄位誤差報告.xlsx"
    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="1.全欄位品管摘要", index=False)
        if not anomaly_df.empty:
            anomaly_df.to_excel(writer, sheet_name="2.製程異常超標點追蹤", index=False)
        detailed_df.to_excel(writer, sheet_name="3.全坐標點微觀明細", index=False)

    return output_filename, len(anomaly_records)


# --- GUI 介面設計 ---
class DropApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 全欄位 3 檔案數據深度比對系統 (演算完全還原版)")
        self.root.geometry("600x450")
        self.root.style = ttk.Style()
        self.root.style.theme_use("clam")

        self.files_path_list = []

        title_lbl = ttk.Label(
            root,
            text="請依序或同時選取「3個」CSV檔案拖入下方區域：",
            font=("Microsoft JhengHei", 11, "bold"),
        )
        title_lbl.pack(pady=10)

        self.drop_zone = tk.Label(
            root,
            text="\n\n📥\n\n將 3 個 CSV 檔案拖曳至此處\n(支援中文路徑、100% 原始數據對齊機制)",
            font=("Microsoft JhengHei", 12),
            bg="#EDF2F7",
            fg="#4A5568",
            bd=2,
            relief="groove",
            width=55,
            height=8,
        )
        self.drop_zone.pack(pady=10, padx=20)

        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind("<<Drop>>", self.handle_drop)

        list_lbl = ttk.Label(
            root,
            text="📋 待比對檔案清單 (依序為 F1、F2、F3)：",
            font=("Microsoft JhengHei", 10, "bold"),
        )
        list_lbl.pack(anchor="w", padx=25)

        self.file_listbox = tk.Listbox(
            root, height=5, width=70, font=("Microsoft JhengHei", 9)
        )
        self.file_listbox.pack(pady=5, padx=20)

        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(pady=15, fill="x", padx=25)

        ttk.Label(
            ctrl_frame, text="異常判定門檻: ", font=("Microsoft JhengHei", 10)
        ).pack(side="left")
        self.threshold_entry = ttk.Entry(ctrl_frame, width=8)
        self.threshold_entry.insert(0, "0.1")
        self.threshold_entry.pack(side="left", padx=5)

        self.run_btn = ttk.Button(
            ctrl_frame,
            text="🚀 開始比對分析",
            command=self.process_files,
            state="disabled",
        )
        self.run_btn.pack(side="right", padx=5)

        self.clear_btn = ttk.Button(
            ctrl_frame, text="🧹 清除重新選擇", command=self.clear_all
        )
        self.clear_btn.pack(side="right", padx=5)

    def handle_drop(self, event):
        files = self.root.tk.splitlist(event.data)

        for f_path in files:
            f_path = os.path.abspath(f_path)
            # 取得檔案副檔名並轉成小寫
            file_ext = os.path.splitext(f_path)[1].lower()
            
            # 檢查副檔名是否為 CSV 或 Excel，且該檔案還沒被加入過
            if file_ext in [".csv", ".xlsx", ".xls"] and f_path not in self.files_path_list:
                if len(self.files_path_list) < 3:
                    self.files_path_list.append(f_path)
                    self.file_listbox.insert(
                        tk.END,
                        f" 📂 F{len(self.files_path_list)}: {os.path.basename(f_path)}",
                    )

        # 嚴格控管必須剛好等於 3 個檔案，確保原始 3 檔邏輯不崩潰
        if len(self.files_path_list) == 3:
            self.run_btn.config(state="normal")
            self.drop_zone.config(
                bg="#E6FFFA",
                fg="#234E52",
                text="\n\n✅\n\n3 個檔案已全數載入！\n可以點擊下方按鈕開始精確比對",
            )
        else:
            self.run_btn.config(state="disabled")
            self.drop_zone.config(
                text=f"\n\n⚠️\n\n目前已有 {len(self.files_path_list)} 個檔案\n請補足或確保拖入「剛好 3 個」檔案。"
            )

    def clear_all(self):
        self.files_path_list.clear()
        self.file_listbox.delete(0, tk.END)
        self.run_btn.config(state="disabled")
        self.drop_zone.config(
            bg="#EDF2F7",
            fg="#4A5568",
            text="\n\n📥\n\n將 3 個 CSV 檔案拖曳至此處\n(支援中文路徑、100% 原始數據對齊機制)",
        )

    def process_files(self):
        try:
            threshold = float(self.threshold_entry.get())
        except ValueError:
            messagebox.showerror("錯誤", "門檻值欄位請輸入正確的數字（例如 0.1）")
            return

        try:
            output_file, anomaly_count = run_comparison(self.files_path_list, threshold)

            msg = f"✨ 數據比對完成！\n\n產出檔案：{output_file}\n異常超標點數：{anomaly_count} 點"
            messagebox.showinfo("分析成功", msg)

            if sys.platform == "win32":
                os.startfile(output_file)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", output_file])
            else:
                subprocess.Popen(["xdg-open", output_file])

        except Exception as e:
            messagebox.showerror("系統錯誤", f"執行過程中發生未預期錯誤：\n{str(e)}")


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = DropApp(root)
    root.mainloop()
