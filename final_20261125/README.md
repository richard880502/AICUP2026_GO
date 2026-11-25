# 最終提交（2026-11-25 整理）

兩個比賽各自 Public 最高分的提交，以及重現它們所需的權重與分數檔。

| 比賽 | 版本 | 官方 Public | 提交檔 |
| --- | --- | ---: | --- |
| 比賽 1：玩家辨識 | v9（v4 + v7 兩個風格模型的分數相加） | **0.8961831** | `submissions/competition1_player_v9_public0.8961831.csv` |
| 比賽 2：棋力預測 | v47（C1+C2 + 用兩個風格模型認對手） | **0.7599623** | `submissions/competition2_rank_v47_public0.7599623.csv` |

所有檔案的 SHA256 在 [`MANIFEST.sha256`](MANIFEST.sha256)。比賽 1 提交檔的 SHA256 是
`ee527fe2170240c8c4ca6e4506ca8053b0ac2a68d4ddc220814c0b1db507e42f`，和 2026-11-15 上傳的檔案相同。

## 內容

| 路徑 | 內容 | 大小 |
| --- | --- | ---: |
| `submissions/` | 兩個最終提交檔 | 126 KB |
| `competition1/weights/style_v4.pt`、`style_v7.pt` | 風格模型（384 維與 960 維逐手特徵，6,807 類） | 各約 18 MB |
| `competition1/weights/style_v4_match_head.pt`、`style_v7_match_head.pt` | 逐手比對的微調頭（含整個編碼器） | 各約 18 MB |
| `competition1/weights/rich_pca.npz` | 960 維特徵的 PCA 投影 | 0.9 MB |
| `competition1/scores/` | 新題庫 pub2 / priv2 的六種分數、候選人與題目名單、原題庫 v6 的答案 | 約 37 MB |
| `competition2/c1c2_v1.1/` | 棋力 C1+C2 凍結版（release v1.1_2026-10-22）的權重、驗收機率與擬合參數 | 約 730 MB |
| `competition2/opponent_identity/` | v47 的輸入：holdout3 與官方對手的類別餘弦、C1+C2 的題目機率、題目對照表 | 約 475 MB |

不包含：官方比賽資料（有授權限制）、KataGo 與 HumanSL 的網路權重（第三方，來源見 repo 根目錄 README 的「使用的外部模型」）、逐手特徵快取（約 150 GB，留在研究機器 `/mnt/share_data_77/richard/AICUP2026_GO/`）。

## 從保留的檔案重現

在研究機器的 repo 根目錄執行，`PYTHONPATH=.:competition1_player_identification`：

- 比賽 1：`python -m competition1_player_identification.experiments.trunk_style.ensemble_v9`
  只讀分數檔，不需要 GPU 或特徵。2026-11-25 用 CPU 重算，新題庫 800 題答案與提交檔完全相同。
- 比賽 2：`python -m competition2_rank_prediction.experiments.opponent_identity.h3_final_v47`
  會在 holdout3 上重新擬合約 45 個參數。2026-11-25 在 CPU 上重跑，800 題中 791 題相同；
  另外 9 題是 CPU 與原本 GPU 浮點誤差造成的邊界題，平均最高機率 0.651（原 0.652）。

從棋譜重頭跑（抽特徵、訓練）的步驟見
`competition1_player_identification/docs/public_v2_trunk_style_20261113.md` 與
`competition2_rank_prediction/experiments/opponent_identity/RESULTS.md` 的重現章節。

## 放上 GitHub 的方式

權重不進 Git：根目錄 `.gitignore` 已排除 `*.pt`、`*.npz`、`*.npy`、`*.pkl`、`*.csv`，
這個資料夾的 `.gitignore` 也排除 `release_assets/`。會被 commit 的只有說明、MANIFEST 與設定檔。

GitHub 的限制（2026-11-25 查官方文件）：

| 方式 | 限制 | 適不適合 |
| --- | --- | --- |
| 直接 commit | 單檔超過 100 MiB 會被拒；建議 repo 小於 1 GB | 不適合：C1+C2 有 7 個 ResNet 各 100.7 MiB |
| Git LFS（免費帳號） | 單檔 2 GB；儲存與每月流量各 10 GiB，超過就不能再推 | 可以，但流量會被下載吃掉 |
| **Release 附件** | 單檔小於 2 GiB；每個 release 最多 1,000 個；總大小與流量不限 | **建議**：不佔 repo 大小，也不吃 LFS 額度 |

三個壓縮檔已上傳到 Release
[`final-20261125`](https://github.com/richard880502/AICUP2026_GO/releases/tag/final-20261125)
（2026-11-25），附 `SHA256SUMS.txt`；上傳後比對過 GitHub 上的 SHA256 與本機相同，
研究機器上的 `release_assets/` 已刪除，內容仍以這個資料夾為準。

| 檔案 | 大小 |
| --- | ---: |
| `go_final_competition1_v9.tar` | 100 MB |
| `go_final_competition2_c1c2_v1.1.tar` | 731 MB |
| `go_final_competition2_opponent_identity_v47.tar` | 475 MB |

下載與解開：

```bash
gh release download final-20261125 --repo richard880502/AICUP2026_GO --dir go_final && cd go_final && shasum -a 256 -c SHA256SUMS.txt && for f in *.tar; do tar -xf "$f"; done
```

注意：repo 目前是私人的（2026-11-25 查詢），release 附件也只有有權限的人看得到。
提交檔（`submissions/`）是用官方題目算出來的答案，repo 原本的政策是不放提交檔，
壓縮檔中 `competition1_v9` 與 `opponent_identity_v47` 各附了一份。之後若把 repo 改成公開，
建立 release 前先把提交檔從壓縮檔中拿掉。
