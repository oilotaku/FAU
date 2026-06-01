import cv2

# 1. 讀取兩張圖片
img1 = cv2.imread('merged_output.jpg')
img2 = cv2.imread('A024-003.jpg')

# 2. 獲取兩張圖片的尺寸 (高度, 寬度, 通道)
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# 3. 比較面積，找出最大圖，並將小圖放大至大圖尺寸
if (w1 * h1) >= (w2 * h2):
    # img1 是大圖，將 img2 放大到 img1 的尺寸
    target_width, target_height = w1, h1
    img2 = cv2.resize(img2, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
else:
    # img2 是大圖，將 img1 放大到 img2 的尺寸
    target_width, target_height = w2, h2
    img1 = cv2.resize(img1, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

# 4. 進行權重半透明融合
alpha = 0.8  # 第一張圖的權重 (0.0 ~ 1.0)
beta = 0.5   # 第二張圖的權重 (0.0 ~ 1.0)
gamma = 0    # 綜合亮度調整值

result = cv2.addWeighted(img1, alpha, img2, beta, gamma)

# 5. 顯示與儲存結果
cv2.imshow('Merged Large Image', result)
cv2.imwrite('merged_output.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()