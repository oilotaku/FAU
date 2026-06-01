import cv2
import numpy as np
import pandas as pd
import math
import os
import select_image_from_current_dir
import time

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
    gray = cv2.GaussianBlur(gray, (7, 7), 0) # 平滑處理減少雜訊
    gray = cv2.dilate(gray, (7, 7), iterations=4)

# 2. K-means 自動分三層 (背景、內圈、外圈)
    data = gray.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 5)
    _, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 3. 排序亮度：找出中等亮度的 Index
    centers = centers.flatten()
    sorted_indices = np.argsort(centers) # [最深, 中等, 最亮]
    mid_label = sorted_indices[2]         # 取得中等亮度的標籤

# 4. 建立初步遮罩
    temp_mask = np.uint8((labels.reshape(gray.shape) == mid_label) * 255)

# 找到該區域的所有輪廓
    contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    inner_filled_mask = np.zeros_like(temp_mask)
    if contours:
        # 找到最大的輪廓（避免雜訊）並填滿
        max_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(inner_filled_mask, [max_cnt], -1, 255, -1) # -1 表示填充

    # result = cv2.bitwise_and(img, img, mask=inner_filled_mask)
    # cv2.namedWindow('Filled Inner Mask', cv2.WINDOW_NORMAL)
    # # cv2.namedWindow('Final Result', cv2.WINDOW_NORMAL)
    # cv2.imshow('Filled Inner Mask', inner_filled_mask)
    # # cv2.imshow('Final Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return inner_filled_mask

def main():
    # unit_p_um = 0.2738666365381
    unit_p_um = 1
    # unit_p_um = float(input ('um/pixel'))

    read_jpg_name = select_image_from_current_dir.select_image_from_current_dir() #取得檔案名稱

    start_time = time.perf_counter()

    template = cv2.imread('ref_TGP.jpg')    #匹配樣本
    samp_img = cv2.imread(read_jpg_name)    #被配對影像
    samp_img_gray = cv2.cvtColor(samp_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    w, h = template_gray.shape
    mask = np.zeros(samp_img_gray.shape, dtype=np.uint8)

    res = cv2.matchTemplate(samp_img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    threshold = 0.6
    loc = np.where(res >= threshold)

    
    rects = []
    for pt in zip(*loc[::-1]):
        rects.append([int(pt[0]), int(pt[1]), int(w), int(h)])
        rects.append([int(pt[0]), int(pt[1]), int(w), int(h)])

    rects, weights = cv2.groupRectangles(rects, groupThreshold=1, eps=0.5) #過濾重複
    
    df = pd.DataFrame(rects, columns=['ch-ch pitch', 'Y', 'W', 'H'])
    # 排序
    df_sorted = df.sort_values(by=['ch-ch pitch'], ascending=[True])
    # df_sorted['group'] = df_sorted.index // 12  

    df_final = df_sorted.sort_values(by=['ch-ch pitch'], ascending=[ True])
    result = list(df_final[['ch-ch pitch', 'Y', 'W', 'H']].itertuples(index=False, name=None))
    df_f = pd.DataFrame( columns=['\u00D8', 'ch-ch pitch', 'Y'])
    # 3. 畫出合併後的框
    i = 0
    last = 0
    for (x, y, bw, bh) in result:
        i = i+1
        # cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
        cv2.rectangle(samp_img, (x, y), (x + bw, y + bh), 255, 2)

        # print(x, y, bw, bh)
        # cv2.namedWindow("samp_img", cv2.WINDOW_NORMAL)
        # cv2.imshow("samp_img",samp_img)
        cropped = samp_img_gray[y:y+h, x:x+w] #擷取
        cropped_mask = scal(cropped)    #擷取後取圓
        mask[y:y+h, x:x+w] = cropped_mask
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
        print(f)
        df_f.loc[len(df_f)+1] = [f, (x + cX) * unit_p_um - ref_point[0], (y + cY) *unit_p_um]
        last = x + cX - ref_point[0]
        print(total_sum)
        cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
        cv2.imshow("cropped",cropped)
        cv2.namedWindow("cropped_mask", cv2.WINDOW_NORMAL)
        cv2.imshow("cropped_mask",cropped_mask)
        cv2.waitKey(50)

    _X = df_f['ch-ch pitch'].values+ref_point[0]
    _Y = df_f['Y'].values
    m, b = np.polyfit(_X, _Y, 1)
    df_f['Y_line'] = m * _X + b
    df_f['△Y'] = df_f['Y_line'] - df_f['Y']
    Max_um = float(input ('Max um'))

    cols = df_f.select_dtypes('number').columns
    print(last)
    df_f[cols] = df_f[cols] * Max_um / last
    df_f['ch-ch pitch_result'] = df_f['ch-ch pitch'] - (np.arange(len(df)) * 127)
    output_df = df_f[['\u00D8', 'ch-ch pitch','ch-ch pitch_result', '△Y']]
    # print(df[['ch-ch pitch', 'Y', 'Delta_Y_lineLLS']].head())
    print(output_df)
    masked_img = cv2.bitwise_and(samp_img_gray, samp_img_gray, mask=mask)

    name_part, _ = os.path.splitext(read_jpg_name)
    output_xlsx = f"{name_part}_result.xlsx"
    export_to_excel(output_df, output_xlsx,)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"程式執行耗時：{execution_time:.4f} 秒")
    cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
    cv2.imshow("cropped",masked_img)
    cv2.namedWindow("masked_img", cv2.WINDOW_NORMAL)
    cv2.imshow("masked_img",masked_img)
    cv2.namedWindow("samp_img", cv2.WINDOW_NORMAL)
    cv2.imshow("samp_img",samp_img)
    cv2.waitKey(0)
    cv2.imwrite('res.jpg',samp_img)


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