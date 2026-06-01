import cv2
import numpy as np
import pandas as pd
import math
import os
import select_image_from_current_dir 
import time


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


def scal(gray):
    # --- 1. 預處理 ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_enhanced, (7, 7), 0)
    blurred = cv2.dilate(blurred, (7, 7), iterations=4)
    

    def get_max_contour_mask(img_gray, target_level_idx):
        data = img_gray.reshape((-1, 1)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = centers.flatten()
        sorted_indices = np.argsort(centers)
        target_label = sorted_indices[target_level_idx]
        
        mask = np.uint8((labels.reshape(img_gray.shape) == target_label) * 255)
        
        # [處理空洞]
        kernel_close = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask)
        
        if contours:
            max_cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(filled_mask, [max_cnt], -1, 255, -1) 

            # --- 關鍵修正：解決圓太大 ---
            # 1. 先用中值濾波平滑邊緣 (中值濾波也會稍微改變尺寸，這裡調小一點)
            filled_mask = cv2.medianBlur(filled_mask, 5)

            # 2. [收縮處理：侵蝕] 
            # 使用圓形核心 (MORPH_ELLIPSE) 讓縮小時保持圓潤，不會變方
            # iterations 越高，圓就縮得越小
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            filled_mask = cv2.erode(filled_mask, kernel_erode, iterations=2) 
            
            # 3. [開運算] 最後才做開運算，移除可能因為侵蝕而產生的細碎突起
            filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_erode)
            
        return filled_mask

    # --- 2. 第一階段 ---
    first_mask = get_max_contour_mask(blurred, 1)
    first_result = cv2.bitwise_and(gray_enhanced, gray_enhanced, mask=first_mask)

    # --- 3. 第二階段 ---
    second_blur = cv2.GaussianBlur(first_result, (5, 5), 0)
    final_mask = get_max_contour_mask(second_blur, 2)

    # --- 4. 顯示 ---
    cv2.imshow('Final Mask (Eroded for Precision)', final_mask)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    
    return final_mask


def main():
    unit_p_um_y = 0.2173170474639
    unit_p_um_x = 0.2193627169091
    read_jpg = select_image_from_current_dir.select_image_from_current_dir()
    start_time = time.perf_counter()
    template = cv2.imread('ref_img1.jpg')
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
    # 3. 畫出合併後的框
    for (x, y, bw, bh) in result:
        # cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
        cv2.rectangle(samp_img, (x, y), (x + bw, y + bh), 255, 1)

        # print(x, y, bw, bh)
        # cv2.namedWindow("samp_img", cv2.WINDOW_NORMAL)
        # cv2.imshow("samp_img",samp_img)
        cropped = samp_img_gray[y:y+h, x:x+w]
        cropped_mask = scal(cropped)
        mask[y:y+h, x:x+w] = cropped_mask
        contours, hie = cv2.findContours(cropped_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)   #get 輪廓

        for cnt in contours:
        # 1. 計算輪廓的矩 (Moments)
            M = cv2.moments(cnt)

        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]

        total = pd.DataFrame(cropped_mask)
        total_sum = total.sum().sum()/255
        f = 2 * math.sqrt((total_sum) / math.pi) * unit_p_um_x *unit_p_um_y
        print(f)
        df_f.loc[len(df_f)+1] = [f, (x + cX) * unit_p_um_x , (y + cY) *unit_p_um_y]
        # cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
        # cv2.imshow("cropped",cropped)
        # cv2.waitKey(0)
    
    # 計算平均中心點
    # cx = df_f['X'].mean()
    # cy = df_f['Y'].mean()

    # # 減去中心點得到相對座標
    # df_f['X'] = abs(df_f['X'] - cx)
    # df_f['Y'] = abs(df_f['Y'] - cy)

    X = df_f['X'].values
    Y = df_f['Y'].values

    # 1. 最小平方法擬合 (得到斜率 m 和 截距 b)
    m, b = np.polyfit(X, Y, 1)

    # 2. 找出數據的算術中心 (這就是擬合線穿過的中心點)
    center_x = np.mean(X)
    center_y = np.mean(Y) # 也可以用 m * center_x + b 計算，結果相同

    # 3. 計算每一點相對於這個 LLS 中心點的偏差 (選配)
    df_f['X'] = abs(df_f['X'] - center_x )
    df_f['Y'] = abs(df_f['Y'] - center_y )



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