import cv2
import numpy as np
import os
import pandas as pd  # 引入 pandas

# 全域變數
points = []
mode = 0  
unit_p_um = 0.2738666365381
measurement_count = 0  
last_pts_count = -1 
# 建立一個空的 DataFrame 來存放數據
df_results = pd.DataFrame(columns=['No', '\u00D8', 'X', 'Y'])

def fit_circle_least_squares(pts):
    pts = np.array(pts, dtype=np.float32)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    B = x**2 + y**2
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res
    xc, yc = a / 2, b / 2
    radius = np.sqrt(c + xc**2 + yc**2)
    return (xc, yc), (int(xc), int(yc)), radius

def fit_line_and_get_dist(pts):
    pts = np.array(pts, dtype=np.float32)
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    projections = [(p - x0) * vx + (p - y0) * vy for p in pts]
    dist_px = np.max(projections) - np.min(projections)
    return float(dist_px), (vx, vy, x0, y0)

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

def main():
    global points, mode, measurement_count, last_pts_count, df_results
    img_path = 'test3.jpg' 
    save_csv = 'measurements.csv'
    
    if not os.path.exists(img_path):
        print(f"錯誤: 找不到 {img_path}"); return
        
    img = cv2.imread(img_path)
    cv2.namedWindow("Precision Measurement Lab", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Precision Measurement Lab", mouse_callback)

    print("\n輸出格式")
    print("No      \u00D8      X          Y")

    while True:
        display_img = img.copy()
        for p in points:
            cv2.drawMarker(display_img, p, (0, 0, 255), cv2.MARKER_CROSS, 8, 1)

        # 預設當前計算數據
        current_row = None 

        if len(points) >= 2:
            try:
                if mode == 0 and len(points) >= 3:
                    (xc, yc), center_int, radius = fit_circle_least_squares(points)
                    real_dia = radius * 2 * unit_p_um
                    current_row = {'\u00D8': f"{real_dia:.4f}", 'X': f"{xc:.4f}", 'Y': f"{yc:.4f}"}
                    
                    if len(points) != last_pts_count:
                        print(f" --     {current_row['\u00D8']:<10} {current_row['X']:<10} {current_row['Y']:<10}")
                        last_pts_count = len(points)
                    cv2.circle(display_img, center_int, int(radius), (0, 255, 0), 1)

                elif mode == 1:
                    dist_px, _ = fit_line_and_get_dist(points)
                    real_dist = dist_px * unit_p_um
                    current_row = {'\u00D8': f"{real_dist:.4f}", 'X': "-", 'Y': "-"}
                    
                    if len(points) != last_pts_count:
                        print(f" --     {current_row['\u00D8']:<10} {'-':<10} {'-':<10}")
                        last_pts_count = len(points)
            except: pass

        cv2.imshow("Precision Measurement Lab", display_img)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('m'):
            mode = 1 - mode
            points = []; last_pts_count = -1; measurement_count = 0
            print(f"\n🔄 模式切換/計數重設\nNo      \u00D8      X          Y")
        elif key == ord('r'):
            points = []; last_pts_count = -1
            print("🔄 已重設選點")
        elif key == ord('s'):
            if current_row:
                measurement_count += 1
                # 將序號加入數據字典
                current_row['No'] = measurement_count
                
                # 使用 pandas 的 concat 功能新增一行
                new_df = pd.DataFrame([current_row])
                df_results = pd.concat([df_results, new_df], ignore_index=True)
                
                # 印出確認
                print(f"{measurement_count:<7} {current_row['\u00D8']:<10} {current_row['X']:<10} {current_row['Y']:<10} *PANDAS SAVED*")
                
                # 將 DataFrame 寫入 CSV (每次 S 都更新檔案)
                df_results.to_csv(save_csv, index=False, encoding='utf-8-sig')
            else:
                print("⚠️ 數據不足")
        elif key == ord('q'):
            # 離開前印出最終結果
            print("\n--- 最終測量數據表 ---")
            print(df_results)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()












    