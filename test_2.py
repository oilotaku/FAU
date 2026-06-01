import cv2
import numpy as np
import pandas as pd
import math

def scal(gray):
# 1. 讀取與預處理
    # img = cv2.imread('ref_img_C.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0) # 平滑處理減少雜訊

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

# 5. 填滿實心圓 (關鍵步驟)
# 找到該區域的所有輪廓，並填滿最外層
    contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    inner_filled_mask = np.zeros_like(temp_mask)
    if contours:
        # 找到最大的輪廓（避免雜訊）並填滿
        max_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(inner_filled_mask, [max_cnt], -1, 255, -1) 

    # 6. 套用遮罩得到結果
    # result = cv2.bitwise_and(img, img, mask=inner_filled_mask)
    cv2.namedWindow('Filled Inner Mask', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Final Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Filled Inner Mask', inner_filled_mask)
    # cv2.imshow('Final Result', result)
    cv2.waitKey(50)
    cv2.destroyAllWindows()
    return inner_filled_mask

def main():
    unit_p_um = 0.219 *1.122

    template = cv2.imread('ref_img_C.jpg')
    samp_img = cv2.imread('C006.jpg')
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

    rects, weights = cv2.groupRectangles(rects, groupThreshold=1, eps=0.5)
    
    df = pd.DataFrame(rects, columns=['X', 'Y', 'W', 'H'])
    # 排序（Y 降序）
    df_sorted = df.sort_values(by=['X'], ascending=[True])
    # df_sorted['group'] = df_sorted.index // 12  # 分組編號

    df_final = df_sorted.sort_values(by=['X'], ascending=[ True])
    result = list(df_final[['X', 'Y', 'W', 'H']].itertuples(index=False, name=None))
    df_f = pd.DataFrame( columns=['\u00D8', 'X', 'Y'])
    # 3. 畫出合併後的框 = 
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
        if len(df_f) == 0:
            ref_point = (x + cX) * unit_p_um, (y + cY) *unit_p_um

        total = pd.DataFrame(cropped_mask)
        total_sum = total.sum().sum()/255
        f = 2 * math.sqrt((total_sum+180) / math.pi) * unit_p_um
        print(f)
        df_f.loc[len(df_f)+1] = [f, (x + cX) * unit_p_um - ref_point[0], (y + cY) *unit_p_um]
        # print(total_sum)
        # cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
        # cv2.imshow("cropped",samp_img_gray)
        # cv2.waitKey(0)

    _X = df_f['X'].values+ref_point[0]
    _Y = df_f['Y'].values*0.9993733
    m, b = np.polyfit(_X, _Y, 1)
    df_f['Y_'] = m * _X + b
    df_f['Delta_Y'] = df_f['Y_'] - df_f['Y']

    # print(df[['X', 'Y', 'Delta_Y_LLS']].head())
    print(df_f)
    masked_img = cv2.bitwise_and(samp_img_gray, samp_img_gray, mask=mask)

    cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
    cv2.imshow("cropped",masked_img)
    cv2.waitKey(10)
    _, thresh = cv2.threshold(masked_img, int(total_sum-1), 255, 0)    #二元化 1=bw 0=wb

    thresh = cv2.erode(thresh, ( 2, 2), iterations=9)
    thresh = cv2.dilate(thresh, ( 2, 2), iterations=2)
    thresh = cv2.erode(thresh, ( 3, 3), iterations=2)
    thresh = cv2.dilate(thresh, ( 3, 3), iterations=3)

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