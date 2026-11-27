# AI CUP 2026 圍棋競賽 GPU 環境

本映像以 w77 的兩個競賽環境為依據。兩者使用 Python 3.11.11、PyTorch 2.5.1+cu121、CUDA 12.1 和 cuDNN 9.1.0。映像內的 KataGo 1.18.2 CUDA 執行檔與 w77 的版本雜湊相同，另包含 HumanSL v1.15.0 的上游 Python 原始碼。Dockerfile 和 Python 依賴鎖定檔可用來重新建置。

## 拉取與驗證

使用 Linux x86_64、NVIDIA GPU、與 CUDA 12.1 相容的主機驅動及 NVIDIA Container Toolkit。映像驗證指令以合成輸入執行兩個競賽模型的 GPU 前向、反向與優化器更新，無需下載棋譜或權重。

```bash
image=ghcr.io/richard880502/aicup2026-go:gpu-w77-v1
docker pull "$image"
docker run --rm --gpus 1 "$image"
docker run --rm --gpus 1 "$image" katago version
```

從包含本目錄的 Git 提交重建映像時，在 repo 根目錄執行 `repro/gpu/build.sh`。建置內容來自 `git archive HEAD`，因此未追蹤的資料、權重、快取及本機環境不會進入映像。映像的系統套件來自 Debian Bookworm 套件庫，日後重建時可能包含更新版系統套件。

## 外掛資料和權重

以下是容器內的路徑。目錄名稱可在主機任意設定，掛載時對到右欄即可。

| 主機內容 | 容器掛載點 |
| --- | --- |
| 有授權的 AI CUP 原始資料，保留 README 所列的子目錄 | `/workspace/data`，唯讀 |
| KataGo 主網路與 HumanSL 引擎版的 `.bin.gz` 檔 | `/assets/katago`，唯讀 |
| `b18c384nbt-humanv0.ckpt` | 掛載其所在目錄至 `/assets/human_trunk`，唯讀 |
| 競賽一逐手特徵及訓練輸出 | `/workspace/production`，可寫入 |
| 競賽二逐手特徵 | `/workspace/competition2_rank_prediction/features`，擷取時可寫入 |
| 競賽二訓練輸出 | `/workspace/competition2_rank_prediction/results`，可寫入 |
| 主辦單位教學程式中含 `network.py` 的目錄，供 ResNet 使用 | `/tmp2/richard/aicup_tutorial_review`，唯讀 |

KataGo 和 HumanSL 權重請依[專案根目錄](../../README.zh-TW.md)列出的來源取得。ResNet 需要[主辦單位教學專案](https://github.com/AILAB-NDHU/AICup-2026-Tutorial)的 `network.py`，w77 使用的檔案來自提交 `359e06fcaf811f9e87179f12f71425dd18dbabd1`。

## 訓練入口

競賽一使用已準備好、結構與 v4 訓練紀錄一致的逐手特徵。設定 `C1_PRODUCTION` 為主機上的 production 根目錄，其下包含 `trunk_style_20261113` 及相關資料。

```bash
docker run --rm --gpus 1 --shm-size=4g \
  -v "$C1_PRODUCTION:/workspace/production" "$image" \
  python -m competition1_player_identification.experiments.trunk_style.train \
  --all-moves --extra --engine-dropout 0.3 --epochs 30 --name style_v4_retrain
```

競賽二以已擷取的 d96 訓練與驗證特徵為例。設定 `C2_FEATURES` 和 `C2_RESULTS` 為主機上的特徵與結果目錄。

```bash
docker run --rm --gpus 1 --shm-size=4g \
  -v "$C2_FEATURES:/workspace/competition2_rank_prediction/features:ro" \
  -v "$C2_RESULTS:/workspace/competition2_rank_prediction/results" \
  -w /workspace/competition2_rank_prediction "$image" \
  python transformer_rank.py --train features/feat96/train --val features/feat96/val \
  --device cuda:0 --tag d96_retrain
```

使用主機帳號寫入掛載目錄時，可加上 `--user "$(id -u):$(id -g)"`，並確保該帳號有目錄寫入權限。若採用歷史腳本中的 `cuda:1` 預設值，請依容器可見的 GPU 調整。從原始棋譜重跑所有步驟仍需依[競賽一實驗紀錄](../../competition1_player_identification/docs/public_v2_trunk_style_20261113.md)與[競賽二實驗紀錄](../../competition2_rank_prediction/experiments/opponent_identity/RESULTS.md)準備資料與中間檔。最終權重與既有分數檔請見[最終提交說明](../../final_20261125/README.md)。
