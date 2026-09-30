# 9月30日 Generative AI — Vision Models 授業案

## まず結論

20個全部を順番に実演するより、**6個を軸に、残りは補助・学生の自習用**にする。
問いは「機械は画像をどう数にし、数から何を読み取り、数から画像をどう作るか」。
毎回「予想を挙手 → パラメータを1つ変える → なぜそうなったか学生に説明してもらう」。

- 今日の公開シラバス: https://ryosuzuki.notion.site/Week-6-e1fc3a073d31839aa35d01a8ab490cdf
- 今日のデッキ: https://docs.google.com/presentation/d/1LQrIC8Mn7mmLrtPDnyOisCvBy0dRoqgwl670WHqDmO4/edit
- 過去録画の確認範囲: **2026年春のWeek 6の文字起こし**。2025年録画を確認したとは言わない。文字起こしの対象区間とローカル98ページPDFの内容を照合した。動画ファイルはクラウド上でローカル読込が失敗したため、全編再生や画面確認は未実施。
- スライド番号は公開中の現行98ページデッキによる。公開PDFを再取得し、ローカルPDFからの全文抽出と一致することを確認した。

## 150分の進行案（12:35–15:05）

| 時間 | 内容・デモ | 学生への問い |
|---|---|---|
| 12:35–12:45 | 導入、先週のAgentsから「見る」へ。**#1 画素** | 同じ絵でも明るさを変えると入力はどう変わる？ |
| 12:45–13:10 | CNN。**#4 畳み込み**、必要に応じ#6 ReLU / #7 Pooling | この3×3窓の出力は正か負か？エッジをどう探す？ |
| 13:10–13:25 | **#9 パッチ→トークン**、#10 Attention | パッチを半分の幅にするとトークン数は何倍？ |
| 13:25–13:40 | YOLO/SAM。#11 IoU、#13 領域選択。実際のSAMを補足 | 「何か」「どこか」「どの画素か」は別問題？ |
| 13:40–13:50 | 休憩 | |
| 13:50–14:10 | **#14 CLIP空間**、#15 Contrastive / #16 Zero-shot | 新しいラベルを候補に足すとスコアが変わるのはなぜ？ |
| 14:10–14:25 | Multimodal LLM。実画像を1枚、同じ画像への別質問 | 画像のラベル付けだけで画像について会話できる？ |
| 14:25–14:45 | **#17 Forward diffusion** と **#18 Noise prediction** | ノイズを足すだけなら簡単。逆向きには何を学習する必要がある？ |
| 14:45–14:55 | #19 Guidance、#20 Latent。Diffusion Explainerへ接続 | 強く誘導すれば常に良い画像になる？ |
| 14:55–15:05 | まとめ・Exit ticket | CNN/ViT/CLIP/Diffusionの役割を1文ずつ。 |

太字の6個が核。#1は短い導入、#11等は補足として使い、20個すべての説明を義務にしない。

## 前回に実際に出てきたデモ（録画文字起こしの時刻）

| 時刻 | 確認した内容 | 今日の改善 |
|---|---|---|
| 00:37:40 | 手書き数字の入力・推論。直後に誤分類に言及 | 「必ず正解」を期待せず、入力のズレやモデルの限界を問いにする |
| 00:48:40–00:50:06 | CNN Explainer、コーヒーカップ、RGB・畳み込みの可視化 | 先に#4で9個の掛け算を理解→大きいネットワークへ |
| 01:03:45–01:06:07 | YOLO/MediaPipe探索とカメラ検出。探す待ち時間、物体誤認識 | タブを事前に開く。カメラ権限・外部サイトが使えなければ#11–13で継続 |
| 01:17:20–01:18:01 | SAMのセグメンテーション実演 | 実モデルと、#13の色ベース領域選択を混同させない |
| 01:38:32–01:40:05 | ChatGPTの画像理解への導入 | CLIPは文章を生成するモデルではないことを補足 |
| 02:19:36–02:20:24 | Diffusion Explainer、ステップごとの画像生成 | 前回は後半に集中。今日は#17–18用に20分を確保 |

## 20個の操作カード

| # | 実験 | 操作と見どころ |
|---|---|---|
| 1 | Pixels are numbers | 画素をクリック、コントラストを変更 |
| 2 | A neuron makes a decision | 重みを変更、加重和とsigmoidが更新 |
| 3 | Feature detectors | エッジを回転、縦横の応答を比較 |
| 4 | Slide a convolution kernel | Blur/Edge/Sharpen、窓を移動、9項の和を見る |
| 5 | Stride and output size | ストライド1→3、出力サイズの変化 |
| 6 | ReLU changes the signal | 入力をシフト、負値が0になる |
| 7 | Max pooling | 領域サイズ変更、最大値が残る |
| 8 | A growing receptive field | 層を増やし、見える範囲が拡大 |
| 9 | Images become tokens | パッチサイズを変更、トークン数が変化 |
| 10 | Attention across patches | クエリをクリック、温度で重みを集中・分散 |
| 11 | Boxes and IoU | 予測ボックスを移動、重なりを計算 |
| 12 | Remove duplicate detections | NMSの閾値を変更、残る箱を比較 |
| 13 | A prompt selects a region | クリック位置と許容差で連結領域を選ぶ |
| 14 | Image and text share a space | ベクトルを回転、cosineの変化 |
| 15 | Contrastive learning | 正しいペアを強化、対角線と損失を見る |
| 16 | Zero-shot labels compete | ラベル集合と温度を変える |
| 17 | Forward diffusion | 信号保持率、固定ノイズと再シード |
| 18 | Predict noise, recover signal | ノイズ予測の誤差と復元MSE |
| 19 | Guidance changes direction | 無条件・条件付き・誘導後のベクトル |
| 20 | A latent space of shapes | 1次元コードを滑らかに変えて形状を補間 |

## 3Blue1Brownとの関係

既存スライドが参照しているのは次の動画。今回の実験は、これらの「具体例・数値・直接操作で理解する」説明を参考にした**独自実装**。映像の完全再現や3D版とは呼ばない。

- Neural networks: https://www.youtube.com/watch?v=aircAruvnKk （スライド参照187秒、349秒）
- Convolutions: https://www.youtube.com/watch?v=KuXjwB4LzSA （スライド参照538秒）
- 本物の3D CNNの補足: https://adamharley.com/nn_vis/cnn/3d.html

行列・画素・固定テキストの読めることを優先し、今日の新規デモは2D。投影や回転で数式を読みにくくしない。

## 外部デモ（今日のNotionで確認したリンク）

- MNIST: https://ryosuzuki.github.io/lecture-demo/05-mnist/
- CNN Explainer: https://poloclub.github.io/cnn-explainer/
- 3D CNN: https://adamharley.com/nn_vis/cnn/3d.html
- YOLO: https://shaqian.github.io/tfjs-yolo-demo/
- SAM: https://segment-anything.com/
- MediaPipe: https://mediapipe-studio.webapps.google.com/
- Diffusion Explainer: https://poloclub.github.io/diffusion-explainer/

## 教え方の注意点

- CNNの畳み込みは通常カーネル反転なしのcross-correlation。本デモもその計算。
- #13はflood fillであり、SAMの推論ではない。
- CLIPのベクトルは説明用。CLIPそのものが画像説明文を生成するわけではない。
- #18は既知の人工ノイズを使う。実モデルが正解のノイズを最初から知っているという説明はしない。
- Stable Diffusionは元画像に戻す暗号の復号ではない。
- 20個ともAPI不要。学習済みモデルの実行と説明用計算を区別。
- UIは英語、教師用進行案は日本語。自習時には各画面のPredictの問いを先に考えてもらう。

## 授業直前の2分チェック

1. index.htmlを開く。#4→#9→#14→#17→#18を順にクリックし、Reset。
2. Full screenで投影サイズを確認。各パートの最初にスライドへ戻る。
3. 外部サイトは上記リンクだけ先に開く。カメラを使う場合は本人の操作で権限を確認。
4. 時間が押したら#5–8、#12、#15–16、#19–20を自習へ回す。
5. 前回録画中の試験日・課題締切は今回に流用しない。

## Exit ticket（英語）

- What does a convolution kernel compute at one location?
- How does changing patch size change the number of tokens?
- Why can adding a text label change CLIP-style normalized scores?
- What target is a noise-prediction diffusion model trained to estimate?
- Which of today's demos used a trained model, and which used illustrative computations?
