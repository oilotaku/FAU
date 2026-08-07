# FAU

工業影像量測與校正工具集，透過影像模板比對取得量測特徵，並以線性回歸模型進行系統性校正，搭配 Tkinter 圖形化介面操作。

Industrial image-based measurement and calibration toolkit. It uses OpenCV template matching to extract measurement features from part images, applies a scikit-learn linear regression model to correct systematic measurement bias, and provides a Tkinter GUI for day-to-day operation.

## 功能 / Features

- **影像模板比對量測**：以 `Cxx_template_.py`、`Cxx_template_gray.py` 定義的模板，對輸入影像進行特徵定位與量測（如 X/Y 差值、pitch error 等）。
  Template-based measurement using image templates defined in `Cxx_template_.py` / `Cxx_template_gray.py` to locate features and compute values such as X/Y offsets and pitch error.
- **多套量測流程**：`main_Cxx.py`、`main_Mxx.py`、`main_M00_windows.py`、`main_TGP.py`、`main_x-x.py` 對應不同機台/軸別的量測主程式。
  Multiple measurement pipelines for different machine/axis configurations.
- **GUI 操作介面**：`GUI_WINDOW_Cxx.py`、`GUI_WINDOW_Mxx.py` 以 Tkinter + `tkinterdnd2` 提供拖放檔案、參數設定與結果匯出的圖形介面。
  Tkinter-based GUI (with drag-and-drop support via `tkinterdnd2`) for loading images, setting parameters, and exporting results.
- **量測校正模型**：`calibration.py` 使用 `sklearn` 的線性回歸，依據實測值與標準真值訓練校正權重，修正系統性偏差。
  `calibration.py` trains a linear regression model against reference ground-truth data to correct systematic measurement bias.
- **輔助工具**：`JPG_DIFF.py`（影像差異比對）、`comperter.py`（比較工具）、`sharpening.py`（影像銳化）、`scal.py`、`select_image_from_current_dir.py`（互動選圖）等。
  Supporting utilities for image diffing, comparison, sharpening, scaling, and interactive image selection.

## 技術 / Tech Stack

Python、OpenCV (`cv2`)、NumPy、pandas、scikit-learn、Tkinter、`tkinterdnd2`

## 檔案結構 / File Overview

| 檔案 / File | 說明 / Description |
| --- | --- |
| `main_*.py` | 各版本/機台量測主程式 Entry points for different measurement pipelines |
| `GUI_WINDOW_*.py` | 圖形化操作介面 Tkinter GUI applications |
| `Cxx_template_*.py` | 量測用模板資料 Template data used for matching |
| `calibration.py` | 校正模型訓練 Calibration model training |
| `stored_image_data.py`, `suport_Cxx_data.py` | 內建資料/支援資料 Bundled/support data |
| `JPG_DIFF.py`, `comperter.py`, `sharpening.py`, `scal.py` | 影像處理輔助工具 Image processing utilities |
| `test_*.py`, `AI_Test2.py`, `123.py` | 測試/實驗腳本 Test and experimental scripts |

## 使用方式 / Usage

安裝相依套件後，依需求執行對應的 `main_*.py` 或 `GUI_WINDOW_*.py`：

Install the dependencies below, then run the desired `main_*.py` or `GUI_WINDOW_*.py` entry point:

```bash
pip install opencv-python numpy pandas scikit-learn tkinterdnd2
python GUI_WINDOW_Cxx.py
```

> 本專案為內部量測工具，部分腳本內含特定機台/資料路徑設定，使用前請依實際環境調整。
> This is an internal measurement tool; some scripts contain machine- or environment-specific paths that should be adjusted before use.
