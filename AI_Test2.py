import cv2
import numpy as np
import pandas as pd
import math


def get_inner_mask(gray_crop):
    blur = cv2.GaussianBlur(gray_crop, (7, 7), 0)
    data = blur.reshape((-1, 1)).astype(np.float32)
    
    # 減少 K-means 迭代次數以加速
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 3, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    # 找出亮度中間的那一層 (通常是目標圓圈)
    centers = centers.flatten()
    mid_label = np.argsort(centers)[1] # 0:深, 1:中, 2:亮 (根據你原本邏輯取1或2)
    
    mask = np.uint8((labels.reshape(gray_crop.shape) == mid_label) * 255)
    
    # 填充孔洞：只保留最大輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled_mask, [max_cnt], -1, 255, -1)
    kernel = np.ones((2, 2), np.uint8)
    filled_mask = cv2.erode(filled_mask, kernel, iterations=1)
    
    return filled_mask

def main():
    # 參數設定
    UNIT_P_UM = 0.219 * 1.125 *1.111
    MATCH_THRESHOLD = 0.6
    
    template = cv2.imread('ref_TGP.jpg', 0) # 直接讀取灰階
    samp_img = cv2.imread('20260320-TGP-800X-A1-02-Fiber-c2.jpg')
    if template is None or samp_img is None:
        print("Error: Image not found.")
        return

    samp_gray = cv2.cvtColor(samp_img, cv2.COLOR_BGR2GRAY)
    h, w = template.shape
    
    # 1. 模板匹配
    res = cv2.matchTemplate(samp_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= MATCH_THRESHOLD)
    
    # 2. 框組整合 (NMS 概念)
    rects = [[int(pt[0]), int(pt[1]), w, h] for pt in zip(*loc[::-1])]
    # cv2.groupRectangles 至少需要兩個重疊框，所以複製一份
    rects, _ = cv2.groupRectangles(rects + rects, groupThreshold=1, eps=0.5)
    
    # 排序：由左至右 (X 軸)
    rects = sorted(rects, key=lambda x: x[0])
    
    results_list = []
    final_mask = np.zeros_like(samp_gray)
    ref_point = None

    # 3. 處理每個偵測到的目標
    for (x, y, bw, bh) in rects:
        # 繪製偵測框到原圖
        cv2.rectangle(samp_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        
        # 取得局部遮罩
        cropped_gray = samp_gray[y:y+bh, x:x+bw]
        inner_mask = get_inner_mask(cropped_gray)
        
        final_mask[y:y+bh, x:x+bw] = inner_mask
        
        
        # 計算質心與面積
        M = cv2.moments(inner_mask)
        if M["m00"] > 0:
            cX_local = M["m10"] / M["m00"]
            cY_local = M["m01"] / M["m00"]
            
            abs_x_um = (x + cX_local) * UNIT_P_UM
            abs_y_um = (y + cY_local) * UNIT_P_UM
            
            if ref_point is None:
                ref_point = (abs_x_um, abs_y_um)
            
            # 直徑計算 (基於面積)
            area = np.sum(inner_mask > 0)
            diameter = 2 * math.sqrt((area - 125) / math.pi) * UNIT_P_UM
            
            results_list.append({
                'F': diameter,
                'X_um': abs_x_um - ref_point[0],
                'Y_um': abs_y_um,
                'Raw_X': abs_x_um
            })

    # 4. 數據分析 (LLS 直線擬合)
    if results_list:
        df = pd.DataFrame(results_list)
        # 進行線性擬合 Y = mX + b
        m, b = np.polyfit(df['Raw_X'], df['Y_um'], 1)
        df['Y_pred'] = m * df['Raw_X'] + b
        df['Delta_Y'] = df['Y_pred'] - df['Y_um']
        
        print("\n--- 測量結果 ---")
        print(df[['F', 'X_um', 'Y_um', 'Delta_Y']])
    
    # 5. 顯示結果
    cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Detection Result", samp_img)
    masked_res = cv2.bitwise_and(samp_gray, samp_gray, mask=final_mask)
    cv2.namedWindow("Masked Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Masked Image", masked_res)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('res_optimized.jpg', final_mask)

if __name__ == "__main__":
    main()