import cv2
import numpy as np
import pandas as pd
import math
import os
import select_image_from_current_dir 
import time

def fit_circle_lls(points):
    """
    線性最小平方法 (LLS) 圓形擬合副程式
    輸入: points (N, 2) 的點集
    回傳: (圓心座標, 半徑)
    """
    if len(points) < 3:
        return (0, 0), 0
    
    x = points[:, 0]
    y = points[:, 1]
    
    # 建立線性方程組 Ax + By + C = -(x^2 + y^2)
    M = np.column_stack([x, y, np.ones_like(x)])
    Y = -(x**2 + y**2)
    
    try:
        # 求解最小平法
        A, B, C = np.linalg.lstsq(M, Y, rcond=None)[0]
        a = -A / 2
        b = -B / 2
        radius = np.sqrt((A**2 + B**2) / 4 - C)
        return (int(a), int(b)), int(radius)
    except Exception:
        return (0, 0), 0

def analyze_fiber_section(src_img, cladding_scale=1.0, core_scale=1.0, 
                          cladding_level=1, core_level=2, show_process=True):
    """
    光纖截面分析副程式
    cladding_level: 包層亮度索引 (0:最暗, 1:中, 2:最亮)
    core_level: 纖芯亮度索引 (0:最暗, 1:中, 2:最亮)
    """
    # 預處理
    if len(src_img.shape) == 3:
        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = src_img.copy()
        
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # --- 1. 提取包層 (Cladding) ---
    data1 = gray_blur.reshape((-1, 1)).astype(np.float32)
    _, labels1, centers1 = cv2.kmeans(data1, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 使用 argsort 根據亮度排序，並依參數選擇 level
    sorted_idx1 = np.argsort(centers1.flatten())
    target_label1 = sorted_idx1[cladding_level] 
    
    mask1_raw = np.uint8((labels1.reshape(gray.shape) == target_label1) * 255)
    cnts1, _ = cv2.findContours(mask1_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cladding_mask = np.zeros_like(gray)
    c1, r1 = (0, 0), 0
    if cnts1:
        max_cnt1 = max(cnts1, key=cv2.contourArea)
        c1, r1 = fit_circle_lls(max_cnt1.reshape(-1, 2))
        r1_adj = int(r1 * cladding_scale)
        cv2.circle(cladding_mask, (int(c1[0]), int(c1[1])), r1_adj, 255, -1)
    
    res_cladding = cv2.bitwise_and(src_img, src_img, mask=cladding_mask)

    # --- 2. 提取纖芯 (Core) ---
    # 改進：只取包層遮罩內的像素進行分類，增加準確度
    in_cladding_pixels = gray_blur[cladding_mask > 0].reshape((-1, 1)).astype(np.float32)
    
    core_mask = np.zeros_like(gray)
    c2, r2 = (0, 0), 0
    
    if in_cladding_pixels.size > 0:
        _, labels2, centers2 = cv2.kmeans(in_cladding_pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        sorted_idx2 = np.argsort(centers2.flatten())
        target_label2 = sorted_idx2[core_level]
        
        # 還原遮罩
        temp_core_mask = np.zeros_like(gray)
        temp_core_mask[cladding_mask > 0] = (labels2.flatten() == target_label2) * 255
        
        cnts2, _ = cv2.findContours(temp_core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts2:
            max_cnt2 = max(cnts2, key=cv2.contourArea)
            c2, r2 = fit_circle_lls(max_cnt2.reshape(-1, 2))
            r2_adj = int(r2 * core_scale)
            cv2.circle(core_mask, (int(c2[0]), int(c2[1])), r2_adj, 255, -1)
    
    final_res = cv2.bitwise_and(src_img, src_img, mask=core_mask)
    
    # 封裝結果 (維持原樣)
    cladding_info = {"center": c1, "radius": r1}
    core_info = {"center": c2, "radius": r2}
    masks = {"cladding": cladding_mask, "core": core_mask}
    
    if show_process:
        cv2.imshow('Cladding Mask', cladding_mask)
        cv2.imshow('Core Mask', core_mask)
        cv2.imshow('Final Detection', final_res)
        
    return final_res, cladding_info, core_info, masks



def export_to_excel(data, filename, sheet_name="Sheet1"):
    """
    Export a pandas DataFrame to an Excel .xlsx file.
    
    :param data: pandas DataFrame
    :param filename: Output Excel file path (must end with .xlsx)
    :param sheet_name: Name of the Excel sheet
    """
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
    


def main():
    unit_p_um_y = 0.2173170474639
    unit_p_um_x = 0.219
    read_jpg = select_image_from_current_dir.select_image_from_current_dir()
    start_time = time.perf_counter()
    template = cv2.imread('test.jpg')
    samp_img = cv2.imread(read_jpg)
    samp_img_gray = cv2.cvtColor(samp_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape
    mask = np.zeros(samp_img_gray.shape, dtype=np.uint8)

    res = cv2.matchTemplate(samp_img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    threshold = 0.9
    loc = np.where(res >= threshold)

    
    rects = []
    for pt in zip(*loc[::-1]):
        rects.append([int(pt[0]), int(pt[1]), int(w), int(h)])
        rects.append([int(pt[0]), int(pt[1]), int(w), int(h)])

    rects, weights = cv2.groupRectangles(rects, groupThreshold=1, eps=0.5)
    
    df = pd.DataFrame(rects, columns=['X', 'Y', 'W', 'H'])
    # 排序（Y 降序）
    df_sorted = df.sort_values(by=['Y'], ascending=[True])
    df_sorted['group'] = df_sorted.index // 12  # 分組編號

    df_final = df_sorted.sort_values(by=['group','X'], ascending=[True, True])
    result = list(df_final[['X', 'Y', 'W', 'H']].itertuples(index=False, name=None))
    df_f = pd.DataFrame( columns=['\u00D8', 'X', 'Y'])
    i = 1
    # 3. 畫出合併後的框
    for (x, y, bw, bh) in result:
        # cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
        cv2.rectangle(samp_img, (x, y), (x + bw, y + bh), 255, 1)

        # print(x, y, bw, bh)
        cv2.namedWindow("samp_img", cv2.WINDOW_NORMAL)
        cv2.imshow("samp_img",samp_img)

        cropped = samp_img_gray[y:y+h, x:x+w]
        # result, clad, core, m = analyze_fiber_section(cropped, cladding_scale=1, core_scale=0.99)
        # cv2.imwrite(f'{i}.jpg', cropped)
        cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
        cv2.imshow("cropped",cropped)
        cv2.waitKey(10)
        i = i+1
        
        mask[y:y+h, x:x+w] = cropped
        contours, hie = cv2.findContours(cropped, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)   #get 輪廓
        
        for cnt in contours:
        # 1. 計算輪廓的矩 (Moments)
            M = cv2.moments(cnt)

        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
        

        total = pd.DataFrame(cropped)
        total_sum = total.sum().sum()/255
        f = 2 * math.sqrt((total_sum) / math.pi) * unit_p_um_x *1.008
        
        print(f)
        df_f.loc[len(df_f)+1] = [f, (x + cX) * unit_p_um_x , (y + cY) *unit_p_um_x]
        
    
    # 1. 確保座標資料完整
    X = df_f['X'].values
    Y = df_f['Y'].values

    # 2. 找出矩形的四個角落 (基於排序後的索引或座標極值)
    # 假設 df_f 已經照 group(排) 和 X(行) 排序：
    # 第一排：前 12 顆；第三排：最後 12 顆
    p1 = (X[0], Y[0])    # 左上 (第一排第一個)
    p2 = (X[11], Y[11])  # 右上 (第一排最後一個)
    p3 = (X[-12], Y[-12])# 左下 (最後一排第一個)
    p4 = (X[-1], Y[-1])  # 右下 (最後一排最後一個)

    # 3. 計算矩形中心點 (四個角的幾何中心)
    center_x = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
    center_y = (p1[1] + p2[1] + p3[1] + p4[1]) / 4

    print(f"矩形中心點座標: ({center_x:.2f}, {center_y:.2f})")

    # 4. 算出每個圓心相對於中心點的絕對值座標
    # 這裡直接更新 DataFrame 的 X, Y 欄位為相對值
    df_f['Rel_X_Abs'] = abs(df_f['X'] - center_x)
    df_f['Rel_Y_Abs'] = abs(df_f['Y'] - center_y)

    # 重新整理 DataFrame 輸出格式
    df_export = df_f[['Ø', 'X', 'Y', 'Rel_X_Abs', 'Rel_Y_Abs']]
    
    print(df_export)


    print(df_f)
    masked_img = cv2.bitwise_and(samp_img_gray, samp_img_gray, mask=mask)
    _, thresh = cv2.threshold(masked_img, int(total_sum+5), 255, 1)    #二元化 1=bw 0=wb

    thresh = cv2.erode(thresh, ( 2, 2), iterations=9)
    thresh = cv2.dilate(thresh, ( 2, 2), iterations=2)
    thresh = cv2.erode(thresh, ( 3, 3), iterations=2)
    thresh = cv2.dilate(thresh, ( 3, 3), iterations=3)

    name_part, _ = os.path.splitext(read_jpg)
    output_xlsx = f"{name_part}_result.xlsx"
    export_to_excel(df_f, output_xlsx)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"程式執行耗時：{execution_time:.4f} 秒")

    cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
    cv2.imshow("cropped",masked_img)
    cv2.waitKey(10)

    cv2.namedWindow("masked_img", cv2.WINDOW_NORMAL)
    cv2.imshow("masked_img",masked_img)
    cv2.waitKey(0)

    cv2.namedWindow("samp_img", cv2.WINDOW_NORMAL)
    cv2.imshow("samp_img",samp_img)
    cv2.waitKey(0)
    cv2.imwrite('res.jpg',thresh)

if __name__ == "__main__":
   main()




# img_rgb = cv.imread('mario.png')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# template = cv.imread('mario_coin.png',0)
# w, h = template.shape[::-1]

# res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where( res >= threshold)

# for pt in zip(*loc[::-1]):
#     cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

# cv.imwrite('res.png',img_rgb)