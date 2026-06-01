import cv2
import numpy as np


def analyze_dynamic_dimension_error(img1_path, img2_path, img3_path, threshold_val=127):
    paths = [img1_path, img2_path, img3_path]
    widths = []
    heights = []
    areas = []

    print("📊 ======== 動態尺寸誤差量化報告 ========")

    for i, path in enumerate(paths):
        # 1. 讀取灰階圖
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ 圖片 {path} 讀取失敗！")
            return

        print(f"📸 圖 {i+1} 原始解析度: {img.shape[1]} x {img.shape[0]} 像素")

        # 2. 二值化（自動分離物件與背景，請根據您的影像調整閥值）
        _, thresh = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY_INV)

        # 3. 尋找物件輪廓
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            print(f"⚠️ 圖 {i+1} 找不到任何物件輪廓，請檢查二值化閥值 (threshold_val)！")
            continue

        # 4. 抓取面積最大的物件（假設為主體）
        max_contour = max(contours, key=cv2.contourArea)

        # 5. 計算外接矩形 (Bounding Box) 的像素長寬
        x, y, w, h = cv2.boundingRect(max_contour)
        area = cv2.contourArea(max_contour)

        widths.append(w)
        heights.append(h)
        areas.append(area)

    if len(widths) < 3:
        print("❌ 無法完成 3 張圖的比對，請確認影像物件是否皆有被偵測到。")
        return

    # 6. 統計誤差量化計算
    w_arr, h_arr, a_arr = np.array(widths), np.array(heights), np.array(areas)

    metrics = {
        "物件寬度 (Width)": w_arr,
        "物件高度 (Height)": h_arr,
        "物件面積 (Area)": a_arr,
    }

    print("-" * 50)
    for name, data in metrics.items():
        mean_v = np.mean(data)
        std_v = np.std(data, ddof=1)  # 樣本標準差
        max_err = np.max(data) - np.min(data)  # 最大全距誤差
        unit = "平方像素" if "面積" in name else "像素 (pixel)"

        print(f"【{name}】分析：")
        print(f"  各圖數據: {data.tolist()}")
        print(f"  ➔ 平均值 (Mean): {mean_v:.2f} {unit}")
        print(f"  ➔ 不穩定度/標準差 (Std Dev): {std_v:.4f} {unit} 👈 核心誤差")
        print(f"  ➔ 最大極端誤差 (Max-Min): {max_err:.2f} {unit}")
        print("-" * 50)


# 📌 使用時請替換成您實際的圖片檔名
analyze_dynamic_dimension_error("A024-001.jpg", "A024-002.jpg", "A024-003.jpg")
