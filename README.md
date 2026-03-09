# AOIRiskGate Studio

影像分類風險閘門（Risk Gate）實作專案，聚焦完整流程交付：
**Train → Eval → Calibrate Gate → Report**。

---

## 1) 專案定位

### 背景問題（Problem）
在影像分類任務中，模型即使有不錯的整體 accuracy，實務上仍可能出現高風險錯誤：
- 錯誤樣本被直接自動放行，造成品質或決策風險
- 僅看總體指標時，無法判斷哪些樣本應送人工複核

### 產品目標（Product Goal）
建立可操作的風險管理流程：
1. 訓練分類模型
2. 以 validation 資料評估模型表現
3. 產生 risk score 並完成 Gate 校正（trade-off）
4. 輸出可閱讀報表（review rate / escape rate / top-k 案例）

### 成功標準（Success Metrics）
- 穩定產生以下可驗證產物：
  - `results/runs/<run>/`：訓練歷程與 checkpoint
  - `results/evals/<run>/`：混淆矩陣、誤判案例、embeddings
  - `results/gate/<run>/`：trade-off、`scores.csv`、`report.json`
- 可透過 threshold 或 target-review 明確控制人工複核比例

---

## 2) 系統設計（SASD / Architecture 概念）

### 系統流程（High-level）
`Train -> Eval -> Calibrate Gate -> Report -> (WebUI 觀察與調參)`

### 核心模組
- `train.py`：模型訓練，輸出 `last.pt` / `best.pt`
- `eval.py`：產出混淆矩陣、誤判樣本與 `val_embeddings.pt`
- `gate_calibrate.py`：計算 risk score、建立 trade-off、推薦 threshold
- `gate_report.py`：輸出報表與 Top-K 案例（可選擇複製圖片）
- `app.py`：Gradio 介面，串接完整流程

### 設計取捨（Trade-off）
- 優先強化可解釋性與可展示性，而非只追 benchmark
- 提供風險分數可視化（per-class histogram、trade-off curve）
- 同時提供 CLI 與 WebUI，兼顧開發重現與跨角色溝通

---

## 3) 專案目錄（Tree）

```text
my_practice/
├─ docs/
│  ├─ webui_refactor_prd.md
│  └─ webui_refactor_task_breakdown.md
├─ results/
│  ├─ runs/                  # 訓練輸出：metrics、loss 曲線、checkpoints
│  ├─ evals/                 # 驗證輸出：cm、誤判案例、embeddings
│  └─ gate/                  # gate 輸出：scores、tradeoff、report
├─ src/
│  ├─ app.py                 # Gradio 入口
│  ├─ core/
│  │  ├─ config.py
│  │  ├─ data.py
│  │  ├─ models.py
│  │  ├─ train.py
│  │  ├─ engine.py
│  │  ├─ eval.py
│  │  ├─ gate.py
│  │  ├─ gate_calibrate.py
│  │  ├─ gate_report.py
│  │  ├─ checkpoints.py
│  │  └─ viz.py
│  └─ webui/
│     ├─ ui_train.py
│     ├─ ui_eval.py
│     ├─ ui_calibrate.py
│     ├─ ui_gate_dashboard.py
│     ├─ ui_report.py
│     └─ ui_paths.py
├─ requirements.txt
└─ README.md
```

---

## 4) 快速開始（Quick Start）

### 4.1 環境安裝
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2 資料準備
請確認資料夾結構可對應 `Config.train_dir` / `Config.val_dir`（預設位於 `src/core/config.py`）。

### 4.3 使用資料集：NEU-DET（表面缺陷檢測）
本專案目前以 **NEU-DET** 作為主要示範資料集。NEU-DET 是工業電腦視覺的經典公開資料集，聚焦於熱軋鋼帶表面缺陷的分類與定位問題。

**資料來源（Source）**
- 原始資料庫：Northeastern University (NEU) Surface Defect Database
  - http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm
- 主要參考文獻：
  - Song, K., & Yan, Y. (2013). *A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects*. Applied Surface Science, 285, 858-864. https://doi.org/10.1016/j.apsusc.2013.09.002
- 若資料由第三方鏡像（例如 Kaggle）取得，資料著作權與學術引用仍應回溯至 NEU 原始資料庫與上述文獻。

**致謝（Acknowledgement）**
- 本專案示範與實驗使用之鋼材表面缺陷影像資料，來自 NEU Surface Defect Database。感謝 Kechen Song、Yunhui Yan 與相關研究團隊公開資料集，促進工業視覺與缺陷檢測研究發展。

**資料集核心規格**
- 類別數量：6 類缺陷
  - Crazing (Cr)：龜裂
  - Inclusion (In)：夾雜
  - Patches (Pa)：斑塊
  - Pitted Surface (PS)：麻點
  - Rolled-in Scale (RS)：壓入氧化皮
  - Scratches (Sc)：劃痕
- 數據量：共 1,800 張影像（每類 300 張）
- 影像屬性：200×200 灰階圖像（Grayscale）

**本專案中的使用方式（建議）**
1. 先依類別建立資料夾，並切分為 train / validation。
2. 將資料整理為影像分類常見結構（每個類別一個子資料夾）。
3. 於訓練指令中明確指定 `--train-dir` 與 `--val-dir`。

範例結構：
```text
dataset/NEU-DET/
├─ train/
│  └─ images/
│     ├─ Cr/
│     ├─ In/
│     ├─ Pa/
│     ├─ PS/
│     ├─ RS/
│     └─ Sc/
└─ validation/
   └─ images/
      ├─ Cr/
      ├─ In/
      ├─ Pa/
      ├─ PS/
      ├─ RS/
      └─ Sc/
```

**為什麼這個資料集有挑戰性**
- 類內差異大（Intra-class variation）：同一缺陷在不同光照、材質或紋理下外觀可能差很多。
- 類間相似度高（Inter-class similarity）：不同缺陷（例如 Cr 與 RS）在視覺上可能非常接近，容易誤判。

這也是本專案加入 Risk Gate 校正流程的原因：不只追求整體 accuracy，也要把高風險樣本攔下來交由人工複核。

---

## 5) CLI 使用流程

### Step 1: 訓練
```bash
python src/core/train.py \
  --train-dir ./dataset/NEU-DET/train/images \
  --val-dir ./dataset/NEU-DET/validation/images \
  --model resnet18 \
  --epochs 20 \
  --run-name baseline
```

主要輸出：
- `results/runs/baseline/metrics.json`
- `results/runs/baseline/checkpoints/{last.pt,best.pt}`
- `results/runs/baseline/loss.png`

### Step 2: 驗證
```bash
python src/core/eval.py \
  --name baseline \
  --model resnet18 \
  --copy-misclassified
```

主要輸出：
- `results/evals/baseline/cm.png`
- `results/evals/baseline/cm_norm.png`
- `results/evals/baseline/val_embeddings.pt`

### Step 3: Gate 校正
```bash
python src/core/gate_calibrate.py \
  --run-name baseline \
  --model resnet18 \
  --method pca_recon_pred_class \
  --target-review 0.10
```

主要輸出：
- `results/gate/baseline/scores.csv`
- `results/gate/baseline/gate_tradeoff_curve.png`
- `results/gate/baseline/recommended.json`

### Step 4: 產生報表
```bash
python src/core/gate_report.py \
  --run-name baseline \
  --target-review 0.10 \
  --top-k 50 \
  --copy-images
```

主要輸出：
- `results/gate/baseline/report.json`
- `results/gate/baseline/review_imgs/`
- `results/gate/baseline/escape_imgs/`

---

## 6) Web UI（Demo 入口）

```bash
python src/app.py --host 127.0.0.1 --port 7860
```

啟動後可在單一介面完成：
- 訓練參數設定與執行
- 驗證結果檢視（CM / misclassified）
- Gate 校正與 threshold 調整
- 報表生成與案例瀏覽

---

## 7) 專案特點

1. **流程完整**：不只涵蓋模型訓練，亦包含驗證、風險控管與人工複核策略，可支援製造／品管／營運場景的決策流程。
2. **可驗證、可追溯**：每一步皆有固定輸出檔與路徑，結果可重現、可回查。
3. **工程可交付**：提供 CLI + WebUI 雙入口，兼顧研發效率與跨角色溝通需求。
4. **具 trade-off 彈性調整能力**：以 review rate / escape rate 平衡人力成本與風險，而非僅依賴單一 accuracy 指標。

---

## 8) 實驗條件（baseline）

為了讓結果可重現，baseline 實驗條件如下：

### 8.1 訓練設定
- Dataset：NEU-DET（train / validation）
- 輸入尺寸：`img_size = 200`
- 模型：`resnet18`（`pretrained = true`）
- Backbone：`train_freeze_backbone = true`
- Epochs：`20`
- Batch size：`128`
- Optimizer 參數：
  - learning rate：`0.001`
  - weight decay：`0.01`
  - momentum：`0.9`
- Seed：`114514`

### 8.2 Gate 設定
- Gate 方法：`pca_recon_pred_class`
- 風險分數：`abs(z-score_normalized)`
- 校正資料：validation set
- 報表設定：`target_review = 0.05`（5% 人工複核）

---

## 9) 關鍵實驗結果（baseline）

以下結果整理自 `results/runs/baseline/` 與 `results/gate/baseline/`：

### 9.1 分類指標（Accuracy / F1）
- 驗證集樣本數：360
- Accuracy：**92.22%**（332 / 360）
- Macro F1：**91.88%**
- 總錯判數：28

> 指標來源：`scores.csv`（`y_true` / `y_pred`）與 `metrics.json`。

### 9.2 Gate 前後差異（以 target review = 5% 為例）
- **Gate 前（不啟用複核）**
  - review rate：0%
  - escape rate：**7.78%**（28 / 360）
  - 錯判攔截率（error capture rate）：0%
- **Gate 後（target-review = 5%）**
  - review rate：**5.00%**（18 / 360）
  - automation rate：95.00%
  - escape rate：**2.92%**（10 / 342）
  - 錯判攔截率：**64.29%**（18 / 28）

補充：若將 review rate 提升到 **8%**，目前 trade-off 顯示 escape rate 可降至 **0%**（在此驗證集上可攔截全部錯判）。

### 9.3 Demo 可操作性（CLI / WebUI 使用情境）
- **CLI（工程重現 / 批次實驗）**
  - 適合做 baseline 比較、固定參數重跑、結果版本化（run-name 對齊）。
  - 可直接串接 shell script / CI，快速完成 Train → Eval → Gate → Report。
- **WebUI（展示 / 協作溝通）**
  - 適合面試 demo、跨角色溝通（工程師/PM/主管）與即時調參。
  - 可視覺化觀察混淆矩陣、錯判案例、trade-off 曲線與 Top-K 高風險樣本。

---

## 10) 已知限制與下一步

### 已知限制
- 目前主要聚焦單一資料集流程。
- `requirements.txt` 需依硬體環境（CPU/GPU）調整對應的 torch 安裝方式。
- 訓練流程目前以從頭訓練為主，不支援載入既有 checkpoint 後接續訓練（resume / fine-tune）。
- 新資料加入時通常需要重跑完整訓練，迭代成本較高。
- **資料分佈改變時，既有 gate 不一定可直接套用**：
  - 輸入分佈漂移（Input Distribution Shift）：光照、材質、拍攝條件改變後，模型信心與錯誤型態可能改變。
  - risk 分數分佈漂移（Risk Score Shift）：原先校正好的 threshold 在新批次資料上可能不再對應相同的 review / escape 行為。
  - 線上抽樣標註後 escape_rate 漂移：上線後抽樣驗證可能發現實際 escape_rate 高於離線估計。

### 下一步計劃
- 加入 `--resume-ckpt`，支援中斷續訓與增量訓練。
- 加入微調模式（如 freeze/unfreeze 策略與 finetune learning rate）。
- 補上核心指標測試（review rate / escape rate），避免後續改版時指標計算失真。
- 增加資料版本紀錄檔（例如資料來源、切分時間、樣本數），提升結果可追溯性。
- 建立 **漂移監控與再校正流程**：
  - 監控輸入分佈與 risk score 分佈（例如 PSI / KL / KS 指標）。
  - 週期性以近期資料重估 threshold，必要時重新 calibrate gate。
  - 導入線上抽樣標註機制，持續追蹤真實 escape_rate 與錯判攔截率。
  - 設定告警門檻（例如 escape_rate 超過 SLA）並觸發人工複核比例上調或模型重訓。


---

## 11) 總結

**AOIRiskGate Studio 是一個將模型效能轉為可落地風險決策的工程化練習專案。**
