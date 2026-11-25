# AICUP 2026 圍棋研究（中文說明）

[English README](README.md)

這個 repo 收錄 AI CUP 2026 圍棋競賽兩個題目的研究程式與記錄。兩題的標籤、
評分方式、模型與推論流程都不同，分數不能互相比較，所以分成兩個目錄：

| 目錄 | 題目 | 目前官方 Public 分數 |
| --- | --- | ---: |
| [`competition1_player_identification/`](competition1_player_identification/) | 玩家辨識：從 400 或 600 位候選人中找出下這幾盤棋的人 | **0.8961831** |
| [`competition2_rank_prediction/`](competition2_rank_prediction/) | 棋力預測：預測沒見過的玩家的登記級數（10 級） | **0.7599623** |

Private 分數尚未公布。兩個最終提交、權重與分數檔整理在
[`final_20261125/`](final_20261125/README.md)（權重放在 GitHub release 附件，不進 Git）。

## 分數演進

### 比賽 1：玩家辨識

| 日期 | 做法 | Public |
| --- | --- | ---: |
| 2026-11-11 | 原題庫模型凍結後套用到新題庫 | 0.5236602 |
| 2026-11-13 | 題庫內訓練與開放式方法融合 | 0.5640548 |
| 2026-11-13 | 逐手 HumanSL 內部向量 + 風格模型（v4） | 0.8592604 |
| 2026-11-14 | 加上重排模型 | 0.8637413 |
| 2026-11-14 | 改用逐手比對 + CSLS 校正（v5） | 0.8801077 |
| 2026-11-14 | 原題庫也改用同一套方法（v6） | 0.8819787 |
| 2026-11-15 | 新題庫再加一個 960 維特徵的風格模型，兩者分數相加（v9，最終版） | **0.8961831** |

之後把 1,200 位驗收玩家也加進訓練重訓（v11 / v12 / v13），分數是 0.8922 / 0.8833 /
0.8906，都沒有超過 v9。重訓本身就會讓官方分數上下跳約 ±0.01，所以最終版維持 v9。

### 比賽 2：棋力預測

| 日期 | 做法 | Public |
| --- | --- | ---: |
| 2026-11-06 | KataGo / HumanSL 逐手特徵 + 序數 Transformer + ResNet + 對手側模型 | 0.7227214 |
| 2026-11-14 | 加上「認出對手是誰」的證據 | 0.7586411 |
| 2026-11-15 | 用兩個風格模型一起認對手（v47，最終版） | **0.7599623** |

## 比賽 1 的做法：玩家辨識

核心想法是把每一盤棋變成一個「風格向量」，同一個人的棋會落在相近的位置。

1. **逐手取特徵**：把玩家自己的每一手（每盤最多 160 手）前的盤面丟進凍結
   的 HumanSL 網路（b18c384，不微調），取落子點的 384 維內部向量；另外加上
   KataGo 與 HumanSL 輸出的 41 個數值（意外度、策略排名、勝率、地盤變化等）。
2. **風格模型**：4 層 Transformer 把一盤棋的所有著手壓成 256 維的單位向量。
   訓練資料是 6,807 位訓練玩家的 24.8 萬盤棋，損失函數是 ArcFace（玩家分類）
   加上監督式對比學習（同一人靠近、不同人拉開）。訓練過程沒有用到任何官方
   測試玩家。
3. **比對**：候選人的參考棋局取平均當作代表向量，和題目的平均向量算餘弦
   相似度。
4. **逐手比對**：題目中的每一手，到候選人的參考棋局裡找最像的一手，再取
   平均。這能抓到「整盤平均」看不出來的局部習慣。
5. **CSLS 校正**：有些候選人和每一題都很像，會被過度選中；CSLS 會扣掉這種
   「跟誰都像」的分數。
6. 取分數最高的 5 位候選人作答。
7. **兩個風格模型相加（最終版 v9）**：另外訓練一個逐手特徵更豐富的風格模型
   （960 維：落子點的最終層向量、周圍 3×3 的平均、第 12 層與第 6 層的向量），
   新題庫用兩個模型各自算出的三種分數相加後再做 CSLS。單獨使用時兩者差不多，
   但錯的地方不同，相加後官方分數從 0.8820 提升到 0.8962。

原題庫（400 人、每人 100 盤參考棋局）和新題庫（600 人、每人 10 盤）使用
同一個風格模型，只是融合權重不同。原題庫已接近滿分，剩餘的空間都在新題庫。

詳細記錄與重現方式：
[trunk style 記錄](competition1_player_identification/docs/public_v2_trunk_style_20261113.md)、
[`competition1_player_identification/README.md`](competition1_player_identification/README.md)。
架構流程圖請見[英文 README](README.md)。

## 比賽 2 的做法：棋力預測

答案是玩家的「登記級數」，而不是實際下出來的棋力，兩者常常對不上，這是
這一題最主要的困難。

### 原本的流程（Public 0.7227）

每盤取最多 96 個決策點，計算 KataGo 策略與 HumanSL 在 5 級、1 段、5 段三種
設定下的意外度，交給序數 Transformer 與 ResNet；再加上只看對手著手的模型，
以及題目層級的修正，最後用期望分數選答案。詳見
[`competition2_rank_prediction/CURRENT_BEST.md`](competition2_rank_prediction/CURRENT_BEST.md)。

### 認出對手（Public 0.7586）

觀察到的事實：測試題裡被問的玩家是新面孔，但他們的對手大約七成是訓練
資料裡已經標好級數的玩家，而配對系統傾向把級數相近的人排在一起。

1. 把每盤棋「對手那一方」的著手丟進比賽 1 的風格模型，和 6,807 位已知玩家
   比對，得到對手是每個人的機率（單盤認對約 42%）。
2. 依每位玩家的登記級數加總，得到「對手是幾級」的機率，並保留「對手不在
   名單內」的份量。
3. 學一張小表：對手是某級時，本人是各級的機率。平手、讓先且對手執白、
   讓先且對手執黑各一張。
4. 一題有 5 到 20 盤，每盤的證據相乘累積。
5. 和原本流程的機率相乘後，用期望分數選答案。

新增的部分約 45 個參數，沒有使用任何測試答案。在 2,000 位驗收玩家上（把
這些玩家從名單中移除，模擬測試情境）的分數是 0.7585，與官方的 0.7586 一致。

最終版（v47）把比賽 1 的兩個風格模型認對手的分數平均：兩者單盤認出的對手只有
36% 相同，平均後驗收集 0.7585 → 0.7632，官方 0.7586 → 0.7600。

之後又檢查了幾個可能的額外資訊來源，都沒有增益：對手是比賽 1 題庫玩家、題目內
棋局順序、「誰也跟這種對手下過」、強制各級題數相等、同一玩家被拆成多題、測試
棋局出現在訓練資料裡、測試玩家共用對手。數字見下方連結的記錄。

這個做法依賴資料集的組成方式（測試題的對手多半是訓練玩家）。完整數字、
沒有增益的變體（加入勝負、擴大名單、多模型平均、找風格相近的玩家）與
注意事項見
[opponent identity 結果記錄](competition2_rank_prediction/experiments/opponent_identity/RESULTS.md)。

## 定案後的負面審查（2026-11-17 至 11-20）

最終提交之後，又在獨立分支上做了九個棋力預測的審查，都沒有改變任何提交。
這些分支已在 2026-11-25 刪除，commit 保留在下列標籤。

| 標籤 | 主題 | 結論 |
| --- | --- | --- |
| [`archive/experiment-strength-score-v2`](https://github.com/richard880502/AICUP2026_GO/tree/archive/experiment-strength-score-v2) | Strength Score v2、Tier A 與全量訓練覆蓋 | 全部負面 |
| [`archive/experiment-rank-context-v1`](https://github.com/richard880502/AICUP2026_GO/tree/archive/experiment-rank-context-v1) | 級數情境（讓子與執色通道） | 官方測試集裡讓子通道是空的 |
| [`archive/audit-shared-opponent-recurrence`](https://github.com/richard880502/AICUP2026_GO/tree/archive/audit-shared-opponent-recurrence) | 測試玩家之間共用的未知對手 | 先審查，結論是不值得做 |
| [`archive/audit-identity-matchmaking-graph`](https://github.com/richard880502/AICUP2026_GO/tree/archive/audit-identity-matchmaking-graph) | 身分與配對圖、跨模型殘差 | 事先登記的門檻沒過，認人這條路到此為止 |
| [`archive/analysis-hard-boundary-feature-atlas`](https://github.com/richard880502/AICUP2026_GO/tree/archive/analysis-hard-boundary-feature-atlas) | 1D/2D 難分界線的特徵圖譜 | 和隨機排列對照後沒有可用訊號 |
| [`archive/research-rankmle`](https://github.com/richard880502/AICUP2026_GO/tree/archive/research-rankmle) | 用十個 HumanSL 級數設定取代三個（前提審查） | 方向穩定但樣本不足 |
| [`archive/experiment-ten-profile-production`](https://github.com/richard880502/AICUP2026_GO/tree/archive/experiment-ten-profile-production) | 十設定版的正式 Transformer | 單獨 +0.0139，但疊上 C1+C2 或 v47 後增益消失 |
| [`archive/experiment-conditional-complementarity-v1`](https://github.com/richard880502/AICUP2026_GO/tree/archive/experiment-conditional-complementarity-v1) | 依對手把握度分組的十設定互補性 | 篩選沒過，前提不成立 |
| [`archive/experiment-opponent-rank-posterior-audit`](https://github.com/richard880502/AICUP2026_GO/tree/archive/experiment-opponent-rank-posterior-audit) | 認對手到級數後驗的真值審查 | 已知/未知對照；v47 的代理指標不能直接部署 |

## 使用的外部模型（最終報告用）

官方 Q&A 規定外部模型必須在報告中列出來源、版本、授權、用途與消融實驗。
沒有使用任何外部資料集，訓練資料全部來自官方比賽資料。

| 模型 | 來源與版本 | 授權（2026-11-16 查核） | 用途 | 消融 |
| --- | --- | --- | --- | --- |
| KataGo 引擎 | [lightvector/KataGo](https://github.com/lightvector/KataGo) v1.18.2，git `fd0723fd` | MIT（repo 的 `LICENSE`） | 執行下面兩個網路，產生逐手的引擎數值 | - |
| KataGo 網路 `kata1-b18c384nbt-s9996604416-d4316597426` | [katagotraining.org](https://katagotraining.org/)，`.bin.gz` SHA256 `9d7a6afe...`；PyTorch `.zip` SHA256 `72a5bc35...` | KataGo Neural Network License（MIT 類，[授權頁](https://katagotraining.org/network_license/)） | 兩題的引擎數值（策略意外度與排名、勝率、目差、地盤變化）。PyTorch 版只用在自我對弈特徵實驗，沒有進任何提交 | 用它取代 HumanSL 的逐手向量：比賽 1 驗收 0.642，HumanSL 為 0.712 |
| HumanSL 網路 `b18c384nbt-humanv0` | [KataGo v1.15.0 release](https://github.com/lightvector/KataGo/releases/tag/v1.15.0)，`.bin.gz` SHA256 `637746e4...` 與 `.ckpt`（323,752,318 bytes） | release 檔案沒有附授權文字；repo 是 MIT，官方網路採用上面的 MIT 類授權。報告若需明確授權聲明，建議向主辦單位確認 | 凍結、不微調。比賽 1：落子點的內部向量（384 維，v7 為 960 維）。比賽 2：C1+C2 中各級數設定下的著手機率 | 比賽 1 加入內部向量前（只用引擎數值）最佳 Public 0.5640548；第一版內部向量 0.7187986 |

## 這個 repo 放了什麼、沒放什麼

有放：原始碼、啟動腳本、設定範本、小型 JSON 報告、研究文件。

沒有放：

- 有授權限制的 SGF / CSV 比賽資料；
- KataGo、HumanSL 與官方教學的模型權重；
- 訓練出來的 `.pt` 檢查點與 `.npz` 特徵快取；
- 記錄檔、虛擬環境、提交檔與本機備份。

這些檔案留在研究用的機器上，要重現 GPU 推論必須另外還原。`data/`、
`runtime/`、`production/` 三個目錄都被 Git 忽略。

## 開發檢查

Python 變更必須通過 `pylint` 與 `mypy`。clone 之後請啟用 repo 內建的 hook：

```bash
git config core.hooksPath .githooks
```
