import cupy as cp

# 1. 檢查 GPU 是否可用
print("CuPy 是否抓到 GPU:", cp.is_available())

# 2. 建立一個 GPU 上的矩陣
x = cp.array([1, 2, 3, 4, 5])
print("陣列位置:", x.device)  # 預期輸出 <CUDA Device 0>
