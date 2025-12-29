# キーワード索引

このファイルは、G検定学習リポジトリ内の重要キーワードとその記述先ファイルへのリンクを提供します。

## 使い方
- キーワードで検索し、詳細が記述されているファイルを見つける
- 類似概念の比較や混同しやすい用語の確認に活用

---

## AI定義・哲学
- **プランニング（STRIPS）**: 前提条件・行動・結果で行動計画を記述する古典的手法 → [03_ai_definition/planning.md](../03_ai_definition/planning.md)
- **幅優先探索（BFS）**: 同階層のノードを全探索後に次階層へ進む探索手法 → [03_ai_definition/search_algorithms.md](../03_ai_definition/search_algorithms.md)
- **深さ優先探索（DFS）**: 深さ方向に優先的に進む探索手法 → [03_ai_definition/search_algorithms.md](../03_ai_definition/search_algorithms.md)
- **A*探索**: ヒューリスティック関数を用いた効率的探索 → [03_ai_definition/search_algorithms.md](../03_ai_definition/search_algorithms.md)

## 機械学習
- **サポートベクターマシン（SVM）**: カーネルトリックで高次元写像し、マージン最大化で線形分離 → [05_machine_learning/supervised_learning.md](../05_machine_learning/supervised_learning.md)
- **カーネルトリック**: 明示的な高次元写像なしに内積を計算する技法 → [05_machine_learning/supervised_learning.md](../05_machine_learning/supervised_learning.md)
- **カーネル関数**: RBF、多項式、線形カーネル等でデータを高次元空間に写像 → [05_machine_learning/supervised_learning.md](../05_machine_learning/supervised_learning.md)
- **サポートベクター**: 分離超平面に最も近いデータ点 → [05_machine_learning/supervised_learning.md](../05_machine_learning/supervised_learning.md)
- **マージン最大化**: クラス間の余白を最大化して汎化性能を向上 → [05_machine_learning/supervised_learning.md](../05_machine_learning/supervised_learning.md)
- **Actor-Critic**: Actorが行動選択、Criticが行動評価する強化学習手法 → [05_machine_learning/reinforcement_learning.md](../05_machine_learning/reinforcement_learning.md)
- **Actor（行動者）**: 方策を学習し行動を選択 → [05_machine_learning/reinforcement_learning.md](../05_machine_learning/reinforcement_learning.md)
- **Critic（評価者）**: 価値関数を学習し行動を評価 → [05_machine_learning/reinforcement_learning.md](../05_machine_learning/reinforcement_learning.md)
- **強化学習**: 試行錯誤を通じて報酬を最大化する方策を学習 → [05_machine_learning/reinforcement_learning.md](../05_machine_learning/reinforcement_learning.md)
- **Q学習**: 価値ベースの強化学習、Q関数を学習 → [05_machine_learning/reinforcement_learning.md](../05_machine_learning/reinforcement_learning.md)
- **Policy Gradient**: 方策ベースの強化学習、方策を直接学習 → [05_machine_learning/reinforcement_learning.md](../05_machine_learning/reinforcement_learning.md)
- **TD誤差（Temporal Difference Error）**: Criticが計算する予測誤差 → [05_machine_learning/reinforcement_learning.md](../05_machine_learning/reinforcement_learning.md)

## 深層学習
- **ニューラルネットワーク基礎**: 入力層・中間層・出力層、活性化関数 → [06_deep_learning/neural_network_basics.md](../06_deep_learning/neural_network_basics.md)
- **Softmax関数**: 多クラス分類で各クラスの確率を出力（合計1） → [06_deep_learning/neural_network_basics.md](../06_deep_learning/neural_network_basics.md)
- **Sigmoid関数**: 2値分類の出力層で確率を計算（0～1） → [06_deep_learning/neural_network_basics.md](../06_deep_learning/neural_network_basics.md)
- **ReLU**: 中間層で最も使われる活性化関数 → [06_deep_learning/neural_network_basics.md](../06_deep_learning/neural_network_basics.md)
- **クレジット割り当て問題**: どのパラメータが誤差に寄与しているかを明らかにする問題 → [06_deep_learning/backpropagation.md](../06_deep_learning/backpropagation.md)
- **誤差逆伝播法（Backpropagation）**: クレジット割り当て問題を解決する学習アルゴリズム → [06_deep_learning/backpropagation.md](../06_deep_learning/backpropagation.md)
- **連鎖律（Chain Rule）**: 合成関数の微分法則、誤差逆伝播の数学的基礎 → [06_deep_learning/backpropagation.md](../06_deep_learning/backpropagation.md)
- **構造的クレジット割り当て**: どの重みが誤差に寄与しているかを決定 → [06_deep_learning/backpropagation.md](../06_deep_learning/backpropagation.md)
- **時間的クレジット割り当て**: 時系列データでどの時点が影響したかを決定 → [06_deep_learning/backpropagation.md](../06_deep_learning/backpropagation.md)
- **CNN（畳み込みニューラルネットワーク）**: 画像認識に特化した深層学習モデル → [06_deep_learning/cnn.md](../06_deep_learning/cnn.md)
- **ストライド（Stride）**: CNNでフィルタを移動させる幅 → [06_deep_learning/cnn.md](../06_deep_learning/cnn.md)
- **パディング（Padding）**: CNNで入力画像の周囲に余白を追加する処理 → [06_deep_learning/cnn.md](../06_deep_learning/cnn.md)
- **プーリング層**: CNNで特徴マップのダウンサンプリングを行う層 → [06_deep_learning/cnn.md](../06_deep_learning/cnn.md)

## AI応用
- **SSD（Single Shot MultiBox Detector）**: ワンステージ物体検出器、リアルタイム処理に適する → [07_ai_applications/image_recognition.md](../07_ai_applications/image_recognition.md)
- **ワンステージ検出器**: 位置とクラスを1回の順伝播で同時予測（SSD、YOLO） → [07_ai_applications/image_recognition.md](../07_ai_applications/image_recognition.md)
- **ツーステージ検出器**: 領域候補生成→分類の2段階処理（R-CNN系） → [07_ai_applications/image_recognition.md](../07_ai_applications/image_recognition.md)
- **物体検出**: 画像内の物体位置（バウンディングボックス）とクラスを特定 → [07_ai_applications/image_recognition.md](../07_ai_applications/image_recognition.md)
- **セマンティックセグメンテーション**: 画素単位でクラス識別を行う手法 → [07_ai_applications/image_recognition.md](../07_ai_applications/image_recognition.md)
- **インスタンスセグメンテーション**: ピクセル単位で個別物体を区別（Mask R-CNN等） → [07_ai_applications/image_recognition.md](../07_ai_applications/image_recognition.md)
- **FCN（Fully Convolutional Network）**: 全結合層を畳み込み層に置き換えたセグメンテーション → [07_ai_applications/image_recognition.md](../07_ai_applications/image_recognition.md)
- **U-Net**: スキップ結合を使うセグメンテーション手法、医療画像で広く使用 → [07_ai_applications/image_recognition.md](../07_ai_applications/image_recognition.md)
- **SegNet**: 最大プーリングインデックス記憶でメモリ効率化 → [07_ai_applications/image_recognition.md](../07_ai_applications/image_recognition.md)
- **DeepLab**: Atrous Convolution使用の高精度セグメンテーション → [07_ai_applications/image_recognition.md](../07_ai_applications/image_recognition.md)
- **エンコーダ-デコーダ**: 圧縮→復元の対称的ネットワーク構造（SegNet、U-Net等） → [07_ai_applications/image_recognition.md](../07_ai_applications/image_recognition.md)
- **IoU（Intersection over Union）**: セグメンテーションの評価指標、予測と正解の重なり度 → [07_ai_applications/image_recognition.md](../07_ai_applications/image_recognition.md)

## 法律・倫理
- **営業秘密**: 秘密管理性・有用性・非公知性の3要件で保護される情報 → [09_law_ethics/trade_secret.md](../09_law_ethics/trade_secret.md)
- **不正競争防止法**: 営業秘密の保護を規定する法律 → [09_law_ethics/trade_secret.md](../09_law_ethics/trade_secret.md)

---

## 更新履歴
- 2025/12/28: 初期版作成（プランニング、探索アルゴリズム、CNN、Softmax、営業秘密）
