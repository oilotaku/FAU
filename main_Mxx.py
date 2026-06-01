import os
import cv2
import math
import time
import numpy as np
import pandas as pd
import select_image_from_current_dir


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
        print(f"Data successfully exported to '{os.path.abspath(filename)}'")
    except Exception as e:
        print(f"Failed to export to Excel: {e}")



def scal(gray):
# 1. 讀取與預處理
    # img = cv2.imread('ref_img_C.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0) # 平滑處理減少雜訊
    gray = cv2.GaussianBlur(gray, (21, 21), 0) # 平滑處理減少雜訊
    gray = cv2.dilate(gray, (21, 21), iterations=3)
    # gray = cv2.dilate(gray, (9, 9), iterations=5)

# 2. K-means 自動分三層 (背景、內圈、外圈)
    data = gray.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 5)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 3. 排序亮度：找出中等亮度的 Index
    centers = centers.flatten()
    sorted_indices = np.argsort(centers) # [最深, 中等, 最亮]
    mid_label = sorted_indices[0]         # 取得中等亮度的標籤

# 4. 建立初步遮罩
    temp_mask = np.uint8((labels.reshape(gray.shape) == mid_label) * 255)

# 5. 填滿實心圓 (關鍵步驟)
# 找到該區域的所有輪廓，並填滿最外層
    contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inner_filled_mask = np.zeros_like(temp_mask)

    if contours:
        # 找到最大的輪廓（避免雜訊）並填滿
        max_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(inner_filled_mask, [max_cnt], -1, 255, -1) # -1 表示填充

    # 6. 套用遮罩得到結果
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (515, 515))
    # 執行開運算 (Opening)
    inner_filled_mask_ = inner_filled_mask
    cv2.namedWindow('befor opening', cv2.WINDOW_NORMAL)
    cv2.imshow('befor opening', inner_filled_mask)
    cv2.waitKey(0)
    inner_filled_mask = cv2.morphologyEx(inner_filled_mask, cv2.MORPH_OPEN, kernel)
    total = pd.DataFrame(inner_filled_mask)
    total_sum = total.sum().sum()/255
    f = 2 * math.sqrt((total_sum) / math.pi) 
    i = 0
    while f == 0:
        i = i+1
        print(f"Retry...{i}")
        kernel_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (150, 150))
        inner_filled_mask = cv2.morphologyEx(inner_filled_mask_, cv2.MORPH_CLOSE, kernel_)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (515-i, 515-i))
        # 執行開運算 (Opening)
        inner_filled_mask = cv2.morphologyEx(inner_filled_mask, cv2.MORPH_OPEN, kernel)
        total = pd.DataFrame(inner_filled_mask)
        total_sum = total.sum().sum()/255
        f = 2 * math.sqrt((total_sum) / math.pi) 

    result = cv2.bitwise_and(gray, gray, mask=inner_filled_mask)
    cv2.namedWindow('Filled Inner Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Final Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Filled Inner Mask', inner_filled_mask)
    cv2.imshow('Final Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return inner_filled_mask

def main():
    unit_p_um = 1# 0.2451592491367
    read_jpg_name = select_image_from_current_dir.select_image_from_current_dir()
    start_time = time.perf_counter()
    template = cv2.imread('ref_img_M2.jpg')
    samp_img = cv2.imread(read_jpg_name)
    samp_img_gray = cv2.cvtColor(samp_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape
    mask = np.zeros(samp_img_gray.shape, dtype=np.uint8)

    res = cv2.matchTemplate(samp_img_gray, template_gray, cv2.TM_CCOEFF_NORMED) #匹配
    threshold = 0.8 #匹配相似度0.8 = 80%
    loc = np.where(res >= threshold) #過濾大於相似度的區塊

    rects = []
    for pt in zip(*loc[::-1]):
        rects.append([int(pt[0]), int(pt[1]), int(w), int(h)])
        rects.append([int(pt[0]), int(pt[1]), int(w), int(h)])

    rects, weights = cv2.groupRectangles(rects, groupThreshold=2, eps=0.7) #移除重複框選部分
    
    df = pd.DataFrame(rects, columns=['X', 'Y', 'W', 'H'])
    # 排序（X 降序）
    df_sorted = df.sort_values(by=['X'], ascending=[True])
    # df_sorted['group'] = df_sorted.index // 12  # 分組編號
    df_final = df_sorted.sort_values(by=['X'], ascending=[ True])
    result = list(df_final[['X', 'Y', 'W', 'H']].itertuples(index=False, name=None))
    df_f = pd.DataFrame( columns=['\u00D8', 'X', 'Y'])
    # 3. 畫出合併後的框 = 
    i = 0
    last = 0
    for (x, y, bw, bh) in result:
        i = i+1
        bh = bh+40
        bw = bw-110
        # cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
        cv2.rectangle(samp_img, (x, y), (x + bw, y + bh), 255, 1)

        # print(x, y, bw, bh)
        # cv2.namedWindow("samp_img", cv2.WINDOW_NORMAL)
        # cv2.imshow("samp_img",samp_img)
        cropped = samp_img_gray[y:y+bh, x:x+bw]
        cropped_mask = scal(cropped)
        mask[y:y+bh, x:x+bw] = cropped_mask
        contours, hie = cv2.findContours(cropped_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)   #get 輪廓

        for cnt in contours:
        # 1. 計算輪廓的矩 (Moments)
            M = cv2.moments(cnt)

        # 2. 計算中心點座標 (cX, cY)
        # 公式：cX = M10/M00, cY = M01/M00
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
        if len(df_f) == 0:
            ref_point = (x + cX) * unit_p_um, (y + cY) *unit_p_um

        total = pd.DataFrame(cropped_mask)
        total_sum = total.sum().sum()/255
        f = 2 * math.sqrt((total_sum) / math.pi) * unit_p_um
        print(f'{i}__{f}')
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
    
        Max_um = float(input ('Enter Max um: '))

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
        output_xlsx = f"{name_part}_result.xlsx"
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
        Max_um = float(input ('enter Max um: '))
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
        output_xlsx = f"{name_part}_result.xlsx"
        export_to_excel(output_df, output_xlsx,)
        print('detact error')


    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"程式執行耗時：{execution_time:.4f} 秒")

    cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
    cv2.namedWindow("masked_img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("samp_img", cv2.WINDOW_NORMAL)
    cv2.imshow("cropped",masked_img)
    cv2.imshow("masked_img",masked_img)
    cv2.imshow("samp_img",samp_img)    
    cv2.imwrite(f"{read_jpg_name}_res.jpg",samp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
   main()