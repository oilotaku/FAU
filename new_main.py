import cv2
import numpy as np
import os
import pandas as pd
import select_image_from_current_dir
import time

# 全域變數
points = []
mode = 0  
unit_p_um = 0.2738666365381
measurement_count = 0  
last_pts_count = -1 
# 欄位清單
cols = ['No', 'Ø', 'Global_X', 'Global_Y', 'Relative_X', 'Error_Y']
df_results = pd.DataFrame(columns=cols)

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

def calculate_metrics(df):
    """計算 Relative_X (相對於第一點) 與 Error_Y (最小平方法擬合偏差)"""
    if len(df) < 1:
        return df
    
    # 確保資料為浮點數
    coords = df[['Global_X', 'Global_Y']].values.astype(float)
    pts_x = coords[:, 0]
    pts_y = coords[:, 1]

    # 1. 計算 Relative_X: 每個 X 減去第一個點的 X
    ref_x = pts_x[0]
    df['Relative_X'] = [f"{(x - ref_x):.6f}" for x in pts_x]

    # 2. 計算 Error_Y (最小平方法擬合直線 Y = mX + c)
    if len(df) >= 2:
        A = np.vstack([pts_x, np.ones(len(pts_x))]).T
        m, c = np.linalg.lstsq(A, pts_y, rcond=None)[0]
        errors = pts_y - (m * pts_x + c)
        df['Error_Y'] = [f"{e:.6f}" for e in errors]
    else:
        df['Error_Y'] = "0.000000"
        
    return df

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

def main():
    global points, mode, measurement_count, last_pts_count, df_results
    save_csv = 'measurements.csv'
    
    print(f"選擇目標圖片(JPG)")
    read_jpg = select_image_from_current_dir.select_image_from_current_dir()
    print(f"選擇參考圖片(JPG)")
    template_jpg = select_image_from_current_dir.select_image_from_current_dir()
    
    template = cv2.imread(template_jpg)
    samp_img = cv2.imread(read_jpg)
    samp_img_gray = cv2.cvtColor(samp_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape

    res = cv2.matchTemplate(samp_img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.9)
    rects = [[int(pt_x), int(pt_y), w, h] for pt_x, pt_y in zip(*loc[::-1])]
    rects, _ = cv2.groupRectangles(rects + rects, groupThreshold=1, eps=0.5)
    
    df_rects = pd.DataFrame(rects, columns=['X', 'Y', 'W', 'H'])
    df_sorted = df_rects.sort_values(by=['X']).reset_index(drop=True)
    result_list = list(df_sorted[['X', 'Y', 'W', 'H']].itertuples(index=False, name=None))

    win_crop, win_full = "ROI Measurement", "Full Image Preview"
    cv2.namedWindow(win_crop, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_full, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_crop, mouse_callback)

    saved_shapes, roi_to_no = {}, {}
    target_idx = 0
    total_targets = len(result_list)

    print("\nNo   Ø          Global_X    Global_Y    Rel_X       Error_Y")
    print("-" * 80)

    while 0 <= target_idx < total_targets:
        roi_x, roi_y, bw, bh = result_list[target_idx]
        cropped = samp_img[roi_y:roi_y+bh, roi_x:roi_x+bw]
        points, last_pts_count = [], -1
        
        while True:
            display_crop, display_full = cropped.copy(), samp_img.copy()
            cv2.rectangle(display_full, (roi_x, roi_y), (roi_x+bw, roi_y+bh), (0, 255, 255), 3)
            
            for s_idx, shape in saved_shapes.items():
                cv2.circle(display_full, shape['center'], shape['radius'], (255, 0, 0), 2)
                cv2.putText(display_full, str(shape['no']), (shape['center'][0]+5, shape['center'][1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 0), 2)

            current_row = None
            if len(points) >= 3:
                try:
                    (xc, yc), center_int, radius = fit_circle_least_squares(points)
                    real_dia = radius * 2 * unit_p_um
                    g_x_um, g_y_um = (xc + roi_x) * unit_p_um, (yc + roi_y) * unit_p_um
                    g_center_px = (int(xc + roi_x), int(yc + roi_y))
                    
                    cv2.circle(display_crop, center_int, int(radius), (0, 255, 0), 1)
                    cv2.circle(display_full, g_center_px, int(radius), (0, 255, 0), 2)
                    
                    current_row = {'Ø': f"{real_dia:.6f}", 'Global_X': f"{g_x_um:.6f}", 'Global_Y': f"{g_y_um:.6f}"}
                    raw_radius, raw_g_center = int(radius), g_center_px
                    
                    if len(points) != last_pts_count:
                        # 即時列印預覽
                        print(f" --  {current_row['Ø']:<10} {current_row['Global_X']:<11} {current_row['Global_Y']:<11} (量測中)", end='\r')
                        last_pts_count = len(points)
                except: pass

            for p in points: cv2.drawMarker(display_crop, p, (0, 0, 255), cv2.MARKER_CROSS, 8, 1)

            cv2.imshow(win_crop, display_crop)
            cv2.imshow(win_full, display_full)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('s') and current_row:
                if target_idx in roi_to_no:
                    this_no = roi_to_no[target_idx]
                else:
                    measurement_count += 1
                    this_no = measurement_count
                    roi_to_no[target_idx] = this_no

                current_row['No'] = this_no
                # 更新 DataFrame
                df_results = df_results[df_results['No'] != this_no]
                df_results = pd.concat([df_results, pd.DataFrame([current_row])], ignore_index=True)
                df_results = df_results.sort_values(by=['No']).reset_index(drop=True)
                
                # 更新指標
                df_results = calculate_metrics(df_results)
                df_results.to_csv(save_csv, index=False, encoding='utf-8-sig')
                
                # 重新抓取該筆資料以進行列印
                r = df_results[df_results['No'] == this_no].iloc[0]
                print(f"{int(r['No']):<4} {r['Ø']:<10} {r['Global_X']:<11} {r['Global_Y']:<11} {r['Relative_X']:<11} {r['Error_Y']:<10} *SAVED*")
                
                saved_shapes[target_idx] = {'no': this_no, 'center': raw_g_center, 'radius': raw_radius}
            
            elif key == ord('r'): points = []; last_pts_count = -1; print("\n重設選點")
            elif key == ord('n'): target_idx += 1; break
            elif key == ord('b'):
                if target_idx > 0: target_idx -= 1; break
            elif key == ord('q'): 
                cv2.destroyAllWindows()
                return

    print("\n--- 測量完成 ---")
    print(df_results)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"程式執行耗時：{execution_time:.4f} 秒")