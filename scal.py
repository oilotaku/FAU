# import cv2
# import numpy as np

# def fit_circle_lls(points):
#     """
#     線性最小平方法 (LLS) 圓形擬合副程式
#     輸入: points (N, 2) 的點集
#     回傳: (圓心座標, 半徑)
#     """
#     if len(points) < 3:
#         return (0, 0), 0
    
#     x = points[:, 0]
#     y = points[:, 1]
    
#     # 建立線性方程組 Ax + By + C = -(x^2 + y^2)
#     M = np.column_stack([x, y, np.ones_like(x)])
#     Y = -(x**2 + y**2)
    
#     try:
#         # 求解最小平法
#         A, B, C = np.linalg.lstsq(M, Y, rcond=None)[0]
#         a = -A / 2
#         b = -B / 2
#         radius = np.sqrt((A**2 + B**2) / 4 - C)
#         return (int(a), int(b)), int(radius)
#     except Exception:
#         return (0, 0), 0

# def analyze_fiber_section(src_img, cladding_scale=1.0, core_scale=1.0, show_process=True):
#     """
#     光纖截面分析副程式
#     src_img: 原始 BGR 影像
#     cladding_scale: 包層圓調整倍率
#     core_scale: 纖芯圓調整倍率
#     show_process: 是否顯示中間過程遮罩
    
#     回傳: (final_result, cladding_data, core_data, masks)
#     """
#     gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)

#     # --- 1. 提取包層 (Cladding) ---
#     data1 = gray.reshape((-1, 1)).astype(np.float32)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 5)
#     _, labels1, centers1 = cv2.kmeans(data1, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
#     sorted_idx1 = np.argsort(centers1.flatten())
#     mid_label = sorted_idx1[1] # 取中等亮度
    
#     mask1_raw = np.uint8((labels1.reshape(gray.shape) == mid_label) * 255)
#     cnts1, _ = cv2.findContours(mask1_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     cladding_mask = np.zeros_like(gray)
#     c1, r1 = (0, 0), 0
#     if cnts1:
#         max_cnt1 = max(cnts1, key=cv2.contourArea)
#         c1, r1 = fit_circle_lls(max_cnt1.reshape(-1, 2))
#         r1_adj = int(r1 * cladding_scale)
#         cv2.circle(cladding_mask, c1, r1_adj, 255, -1)
    
#     res_cladding = cv2.bitwise_and(src_img, src_img, mask=cladding_mask)

#     # --- 2. 提取纖芯 (Core) ---
#     gray2 = cv2.cvtColor(res_cladding, cv2.COLOR_BGR2GRAY)
#     data2 = gray2.reshape((-1, 1)).astype(np.float32)
#     _, labels2, centers2 = cv2.kmeans(data2, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
#     sorted_idx2 = np.argsort(centers2.flatten())
#     bright_label = sorted_idx2[1] # 取最亮
    
#     mask2_raw = np.uint8((labels2.reshape(gray2.shape) == bright_label) * 255)
#     cnts2, _ = cv2.findContours(mask2_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     core_mask = np.zeros_like(gray)
#     c2, r2 = (0, 0), 0
#     if cnts2:
#         max_cnt2 = max(cnts2, key=cv2.contourArea)
#         c2, r2 = fit_circle_lls(max_cnt2.reshape(-1, 2))
#         r2_adj = int(r2 * core_scale)
#         cv2.circle(core_mask, c2, r2_adj, 255, -1)
    
#     final_res = cv2.bitwise_and(src_img, src_img, mask=core_mask)
    
#     # 封裝結果
#     cladding_info = {"center": c1, "radius": r1}
#     core_info = {"center": c2, "radius": r2}
#     masks = {"cladding": cladding_mask, "core": core_mask}
    
#     if show_process:
#         cv2.imshow('Cladding Mask', cladding_mask)
#         cv2.imshow('Core Mask', core_mask)
#         cv2.imshow('Final Detection', final_res)
        
#     return final_res, cladding_info, core_info, masks

# if __name__ == "__main__":
#     test_img = cv2.imread(r'.\smp_img\14.jpg')
#     if test_img is not None:
#         # 直接呼叫副程式，並設定縮放參數
#         result, clad, core, m = analyze_fiber_section(test_img, cladding_scale=1, core_scale=0.99)
        
#         print(f"包層數據: {clad}")
#         print(f"纖芯數據: {core}")
        
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # 1. 建立黑色畫布 (500x500)
# height, width = 500, 500
# mask_circle = np.zeros((height, width), dtype="uint8")
# mask_rect = np.zeros((height, width), dtype="uint8")

# # 2. 繪製圓形與矩形 (填滿白色 255)
# # 圓形：圓心(250, 250), 半徑 100
# cv2.circle(mask_circle, (250, 250), 100, 255, -1)
# # 矩形：左上(200, 200), 右下(400, 300) -> 這個矩形會蓋住圓形的一部分
# cv2.rectangle(mask_rect, (200, 200), (400, 300), 255, -1)

# # 3. 核心邏輯：從圓形中移除矩形
# # 先將矩形遮罩反轉（矩形變黑，背景變白）
# mask_rect_inv = cv2.bitwise_not(mask_rect)

# # 將圓形遮罩與「反轉後的矩形」做 AND 運算
# # 只有同時是圓形且「非矩形」的區域會被保留
# result = cv2.bitwise_and(mask_circle, mask_rect_inv)

# # 顯示結果
# cv2.imshow("Result - Circle protruding from Rect", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# 1. 讀取圖片 (建議轉成灰階以縮短數據長度)
img = cv2.imread('ref_Cxx_1.jpg', cv2.IMREAD_GRAYSCALE)

# 2. 如果圖片太大，建議先縮小，否則程式碼會變超長
# img = cv2.resize(img, (20, 20)) 

# 3. 輸出成程式碼格式
if img is None:
    print("錯誤：找不到圖片，請檢查檔案名稱與路徑！")
else:
    # 2. 直接寫入成一個獨立的 Python 檔案
    output_filename = "stored_image_data.py"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("import numpy as np\n\n")
        f.write(f"# 原始圖片尺寸為: {img.shape}\n")
        f.write("raw_data = np.array([\n")
        
        # 逐行寫入像素，不經過螢幕顯示，避開環境截斷限制
        for row in img:
            f.write(f"    {row.tolist()},\n")
            
        f.write(f"], dtype=np.uint8)\n")
        
    print(f"🎉 成功！完整原圖數據已寫入：{output_filename}")
    print(f"圖片尺寸為: {img.shape}，沒有經過任何縮小。")

print(f"\n# 圖片尺寸為: {img.shape}")