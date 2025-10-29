# Spam Email Classifier (SVM Baseline & Visualization)

本專案提供一個以 SVM 為基礎的垃圾郵件分類器，並包含互動式資料視覺化與即時推論功能（Colab/Notebook 版）。

## 主要功能
- 資料總覽圖像（類別分布、訊息長度分布）
- Top Tokens by Class（可互動調整 N）
- 模型效能圖像（混淆矩陣、ROC、Threshold sweep）
- Live Inference（手動輸入或一鍵產生範例，顯示預測與機率）

## 快速開始
1. 下載 `spam_email_svm_visualization.ipynb` 並上傳至 [Google Colab](https://colab.research.google.com/)
2. 依序執行各區塊

## 依賴套件
請參考 `requirements.txt`。

## 資料集
- [sms_spam_no_header.csv](https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv)

---

本專案可作為 AIoT-DA2025 HW3 的 baseline 及視覺化展示。