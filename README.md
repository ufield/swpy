# swpy

太陽風データを使用した磁気嵐（Magnetic Storm）予測システムです。DST指数とOMNI太陽風データを用いて、磁気嵐の発生を予測します。

*このREADMEはDevin AIによって作成されました。*

## 概要

このリポジトリは、太陽風データを処理し、機械学習モデルを使用して磁気嵐を予測するためのPythonライブラリを提供します。主に4時間先の磁気嵐状況を予測するように設計されています。

## 特徴

- DST（Disturbance Storm Time）指数データの取得と処理
- OMNI太陽風データの取得と処理
- 磁気嵐イベントのデータセット作成
- トレーニングデータとテストデータの分割機能
- 複数の予測モデル（MLP、LSTM、ガウス過程）
- モデルトレーニング機能
- 評価と可視化ツール

## 主要なパラメータ

予測に使用される主な物理量：
- `flow_speed`: 太陽風の流速
- `BZ_GSM`: 惑星間空間磁場のZ成分（GSM座標系）
- `F`: 磁場の全強度
- `proton_density`: 太陽風のプロトン密度

## 依存関係

このプロジェクトを実行するには以下のライブラリが必要です：

- PyTorch: ニューラルネットワークモデル用
- GPyTorch: ガウス過程モデル用
- pandas, numpy: データ処理用
- matplotlib: 可視化用
- pyspedas: 宇宙物理データアクセス用

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/ufield/swpy.git
cd swpy

# 必要なパッケージのインストール
pip install torch numpy pandas matplotlib pyspedas gpytorch
```

## 使用方法

### データセットの作成

DST指数とOMNI太陽風データから磁気嵐イベントのデータセットを作成します：

```bash
export ROOT_DATA_DIR="/path/to/your/data/directory/"
python swpy/dataset/create_storm_dataset.py --targets flow_speed,BZ_GSM,F,proton_density
```

### トレーニングとテストデータの分割

```bash
python swpy/dataset/create_train_test.py --targets flow_speed,BZ_GSM,F,proton_density
```

### モデルのトレーニング

各種モデル（MLP、LSTM、ガウス過程）を使用して磁気嵐予測のトレーニングを行います。

### 評価と可視化

予測結果を評価し、可視化するためのツールが提供されています。

## リポジトリ構造

- `swpy/dataset/`: データセット作成と処理関連のスクリプト
- `swpy/models/`: 予測モデル（MLP、LSTM、ガウス過程）の実装
- `swpy/eval/`: 評価と可視化のためのスクリプト
- `swpy/utils/`: ユーティリティ関数

## 参照

このプロジェクトはGruet_2018の研究に基づいています。

- [ガウス過程と機械学習 (機械学習プロフェッショナルシリーズ)](https://www.amazon.co.jp/%E3%82%AC%E3%82%A6%E3%82%B9%E9%81%8E%E7%A8%8B%E3%81%A8%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92-%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%97%E3%83%AD%E3%83%95%E3%82%A7%E3%83%83%E3%82%B7%E3%83%A7%E3%83%8A%E3%83%AB%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA-%E6%8C%81%E6%A9%8B-%E5%A4%A7%E5%9C%B0/dp/4061529269)
