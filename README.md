# Spam Email Classifier (AIoT-DA2025 HW3)

[Open in Streamlit](REPLACE_WITH_YOUR_STREAMLIT_APP_URL)

本專案是一個可重現的 spam/ham 郵件分類 pipeline，結合 scikit-learn 與 OpenSpec，並支援 Streamlit 互動展示。

---

## 專案簡介
本專案基於 Packt《Hands-On Artificial Intelligence for Cybersecurity》第三章的資料集與範例，擴充了前處理步驟、豐富的視覺化（步驟輸出、指標、CLI/Streamlit 視圖），並以 OpenSpec 管理規格與變更。

- 預處理報告：`docs/PREPROCESSING.md`
- OpenSpec 變更提案：`openspec/changes/add-spam-email-classifier/`
- 來源參考：https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity.git
- Demo Site: https://2025spamemail.streamlit.app/

---

## 安裝與執行

### 環境安裝
建議於全新虛擬環境下安裝：
```bash
pip install -r requirements.txt
```

### 資料集
- 原始資料集（無標頭2欄CSV）：`datasets/sms_spam_no_header.csv`
- 清理後資料集（自動產生）：`datasets/processed/sms_spam_clean.csv`

### 指令範例
#### 預處理（可選，儲存每步輸出）
```bash
python scripts/preprocess_emails.py \
	--input datasets/sms_spam_no_header.csv \
	--output datasets/processed/sms_spam_clean.csv \
	--no-header --label-col-index 0 --text-col-index 1 \
	--output-text-col text_clean \
	--save-step-columns \
	--steps-out-dir datasets/processed/steps
```
#### 訓練
```bash
python scripts/train_spam_classifier.py \
	--input datasets/processed/sms_spam_clean.csv \
	--label-col col_0 --text-col text_clean
```
#### 單筆預測
```bash
python scripts/predict_spam.py --text "Free entry in 2 a wkly comp to win cash"
```
#### 批次預測
```bash
python scripts/predict_spam.py \
	--input datasets/processed/sms_spam_clean.csv \
	--text-col text_clean \
	--output predictions.csv
```

---

## 筆記
- 訓練產生的 artifacts 會儲存於 `models/`（vectorizer, model, label mapping）。
- 詳細前處理步驟與範例請見 `docs/PREPROCESSING.md`。
- OpenSpec 驗證指令：
	```bash
	openspec validate add-spam-email-classifier --strict
	```
- 推薦訓練參數（Precision ≧ 0.90, Recall ≧ 0.93）：
	```bash
	python scripts/train_spam_classifier.py \
		--input datasets/processed/sms_spam_clean.csv \
		--label-col col_0 --text-col text_clean \
		--class-weight balanced \
		--ngram-range 1,2 \
		--min-df 2 \
		--sublinear-tf \
		--C 2.0 \
		--eval-threshold 0.50
	```
	實測（held-out）：Precision ≈ 0.923, Recall ≈ 0.966, F1 ≈ 0.944。

---

## 視覺化
產生視覺化報告（輸出於 `reports/visualizations/`）：

- 類別分布
	```bash
	python scripts/visualize_spam.py \
		--input datasets/processed/sms_spam_clean.csv \
		--label-col col_0 \
		--class-dist
	```
- Token 頻率（每類前 20）
	```bash
	python scripts/visualize_spam.py \
		--input datasets/processed/sms_spam_clean.csv \
		--label-col col_0 --text-col text_clean \
		--token-freq --topn 20
	```
- 混淆矩陣、ROC、PR（需已訓練模型）
	```bash
	python scripts/visualize_spam.py \
		--input datasets/processed/sms_spam_clean.csv \
		--label-col col_0 --text-col text_clean \
		--models-dir models \
		--confusion-matrix --roc --pr
	```
- Threshold sweep（CSV+圖）
	```bash
	python scripts/visualize_spam.py \
		--input datasets/processed/sms_spam_clean.csv \
		--label-col col_0 --text-col text_clean \
		--models-dir models \
		--threshold-sweep
	```

---

## Streamlit App
啟動互動式儀表板：
```bash
streamlit run app/streamlit_app.py
```

### 主要功能
- 資料集與欄位選擇
- 類別分布、Top tokens by class
- 混淆矩陣、ROC/PR 曲線（需已訓練模型）
- 閾值滑桿（即時 precision/recall/f1）
- Live Inference：即時輸入訊息預測 spam/ham 與機率條
- 快速測試：一鍵填入 spam/ham 範例

### 雲端部署
1. 將本 repo 推送至 GitHub（已完成）。
2. 前往 https://share.streamlit.io，點選 “New app”，選擇：
	 - Repository: DotRabbit74/2025ML-HW3--spamEmail
	 - Branch: main
	 - Main file: app/streamlit_app.py
3. 等待建置完成。App 會自動載入 models/ 內的 artifacts。

---

## Project Status
Phases 1–4 are complete and archived. See the final summary: `docs/FinalReport.md`.