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
- **状態空間探索**: 問題を状態と遷移で表現し解を探索 → [03_ai_definition/search_algorithms.md](../03_ai_definition/search_algorithms.md)
- **記号的AI**: ルールベースの推論・探索（探索アルゴリズム、プランニング等） → [03_ai_definition/search_algorithms.md](../03_ai_definition/search_algorithms.md)
- **サブシンボリックAI**: パターン認識・統計的学習（深層学習、機械学習） → [03_ai_definition/search_algorithms.md](../03_ai_definition/search_algorithms.md)
- **ミニマックス法（Minimax）**: 自分の最大利益と相手の最小利益を追求するゲーム探索手法 → [03_ai_definition/search_algorithms.md](../03_ai_definition/search_algorithms.md)
- **αβ枝刈り（Alpha-Beta Pruning）**: ミニマックス法の効率化、不要な枝を探索しない → [03_ai_definition/search_algorithms.md](../03_ai_definition/search_algorithms.md)
- **ゲーム木（Game Tree）**: ゲームの可能な手順を木構造で表現 → [03_ai_definition/search_algorithms.md](../03_ai_definition/search_algorithms.md)
- **ゼロサムゲーム（Zero-sum Game）**: 一方の利益が他方の損失となるゲーム → [03_ai_definition/search_algorithms.md](../03_ai_definition/search_algorithms.md)
- **評価関数（Evaluation Function）**: 盤面の有利・不利を数値化する関数 → [03_ai_definition/search_algorithms.md](../03_ai_definition/search_algorithms.md)
- **モンテカルロ木探索（MCTS）**: ランダムシミュレーションで評価するゲーム探索（AlphaGo） → [03_ai_definition/search_algorithms.md](../03_ai_definition/search_algorithms.md)
- **フレーム問題（Frame Problem）**: 行動の前後で何が変化し何が変化しないかを列挙する計算量爆発の問題 → [03_ai_definition/frame_problem.md](../03_ai_definition/frame_problem.md)
- **常識の問題（Common Sense Problem）**: 人間の一般常識をAIに習得させることの困難さ → [03_ai_definition/frame_problem.md](../03_ai_definition/frame_problem.md)
- **シンボルグラウンディング問題（Symbol Grounding Problem）**: 記号と現実世界の意味の対応付けの困難さ → [03_ai_definition/frame_problem.md](../03_ai_definition/frame_problem.md)
- **資格問題（Qualification Problem）**: 行動の前提条件を全て列挙できない問題 → [03_ai_definition/frame_problem.md](../03_ai_definition/frame_problem.md)
- **Cycプロジェクト**: 人間の常識を論理式で記述する試み（1984年～） → [03_ai_definition/frame_problem.md](../03_ai_definition/frame_problem.md)

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
- **ε-greedy方策**: 確率εでランダム行動、確率1-εで最良行動を選択する探索戦略 → [05_machine_learning/reinforcement_learning.md](../05_machine_learning/reinforcement_learning.md)
- **探索（Exploration）**: 未知の行動を試して新しい情報を得る → [05_machine_learning/reinforcement_learning.md](../05_machine_learning/reinforcement_learning.md)
- **活用（Exploitation）**: 現在の知識で最良の行動を選ぶ → [05_machine_learning/reinforcement_learning.md](../05_machine_learning/reinforcement_learning.md)
- **ε-減衰（ε-decay）**: 学習進行に応じてεを減少させる手法 → [05_machine_learning/reinforcement_learning.md](../05_machine_learning/reinforcement_learning.md)
- **探索と活用のトレードオフ**: 探索しすぎると報酬低下、活用しすぎると局所解 → [05_machine_learning/reinforcement_learning.md](../05_machine_learning/reinforcement_learning.md)
- **RMSE（Root Mean Squared Error）**: 外れ値に敏感な回帰誤差指標（誤差を2乗） → [05_machine_learning/evaluation_metrics.md](../05_machine_learning/evaluation_metrics.md)
- **MAE（Mean Absolute Error）**: 外れ値に頑健な回帰誤差指標（誤差の絶対値） → [05_machine_learning/evaluation_metrics.md](../05_machine_learning/evaluation_metrics.md)
- **精度（Accuracy）**: 全予測のうち正解した割合。不均衡データでは注意 → [05_machine_learning/evaluation_metrics.md](../05_machine_learning/evaluation_metrics.md)
- **適合率（Precision）**: Positiveと予測したもののうち実際にPositiveだった割合 → [05_machine_learning/evaluation_metrics.md](../05_machine_learning/evaluation_metrics.md)
- **再現率（Recall）**: 実際のPositiveのうち正しく検出できた割合 → [05_machine_learning/evaluation_metrics.md](../05_machine_learning/evaluation_metrics.md)
- **F1スコア**: 適合率と再現率の調和平均、バランス指標 → [05_machine_learning/evaluation_metrics.md](../05_machine_learning/evaluation_metrics.md)
- **AUC-ROC**: 閾値非依存の分類性能指標、ROC曲線下の面積 → [05_machine_learning/evaluation_metrics.md](../05_machine_learning/evaluation_metrics.md)
- **R²（決定係数）**: モデルがデータ分散を説明する割合 → [05_machine_learning/evaluation_metrics.md](../05_machine_learning/evaluation_metrics.md)
- **アンサンブル学習**: 複数モデルを組み合わせて精度向上 → [05_machine_learning/ensemble_learning.md](../05_machine_learning/ensemble_learning.md)
- **バギング（Bagging）**: 並列訓練で分散低減、ブートストラップサンプリング → [05_machine_learning/ensemble_learning.md](../05_machine_learning/ensemble_learning.md)
- **ブースティング（Boosting）**: 逐次訓練でバイアス低減、弱学習器を改善 → [05_machine_learning/ensemble_learning.md](../05_machine_learning/ensemble_learning.md)
- **ランダムフォレスト**: 複数の決定木をバギングで組み合わせる代表的アンサンブル → [05_machine_learning/ensemble_learning.md](../05_machine_learning/ensemble_learning.md)
- **XGBoost**: 勾配ブースティングの高速実装、Kaggleで頻用 → [05_machine_learning/ensemble_learning.md](../05_machine_learning/ensemble_learning.md)
- **AdaBoost**: 最初のブースティング手法、誤分類重視 → [05_machine_learning/ensemble_learning.md](../05_machine_learning/ensemble_learning.md)
- **スタッキング（Stacking）**: 異種モデルをメタ学習器で統合 → [05_machine_learning/ensemble_learning.md](../05_machine_learning/ensemble_learning.md)
- **弱学習器（Weak Learner）**: 単独では精度が低いがランダムより良いモデル → [05_machine_learning/ensemble_learning.md](../05_machine_learning/ensemble_learning.md)
- **k-分割交差検証（k-fold Cross-Validation）**: データをk個に分割し各回1つをテスト・残りを訓練として繰り返し評価 → [05_machine_learning/cross_validation.md](../05_machine_learning/cross_validation.md)
- **ホールドアウト法**: データを1回だけ訓練・テストに分割する評価手法 → [05_machine_learning/cross_validation.md](../05_machine_learning/cross_validation.md)
- **LOOCV（Leave-One-Out Cross-Validation）**: k=データ数の交差検証、小規模データ向き → [05_machine_learning/cross_validation.md](../05_machine_learning/cross_validation.md)
- **層化k-分割（Stratified k-fold）**: 各foldでクラス比率を保持する交差検証 → [05_machine_learning/cross_validation.md](../05_machine_learning/cross_validation.md)

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
- **RNN（Recurrent Neural Network）**: 系列データを処理する再帰型ニューラルネットワーク → [06_deep_learning/rnn.md](../06_deep_learning/rnn.md)
- **隠れ状態（Hidden State）**: RNNで前の時刻の情報を保持するベクトル → [06_deep_learning/rnn.md](../06_deep_learning/rnn.md)
- **勾配消失問題**: RNNで勾配が減衰し長期依存性の学習が困難になる問題 → [06_deep_learning/rnn.md](../06_deep_learning/rnn.md)
- **Encoder-Decoder**: 系列変換アーキテクチャ、機械翻訳に最適 → [06_deep_learning/rnn.md](../06_deep_learning/rnn.md)
- **Seq2Seq（Sequence-to-Sequence）**: 可変長入力→可変長出力の系列変換タスク → [06_deep_learning/rnn.md](../06_deep_learning/rnn.md)
- **コンテキストベクトル**: Encoderが生成する固定長の意味表現 → [06_deep_learning/rnn.md](../06_deep_learning/rnn.md)
- **Attention機構**: Decoderが入力の重要部分に注目する仕組み → [06_deep_learning/rnn.md](../06_deep_learning/rnn.md)
- **双方向RNN**: 順方向・逆方向の両方向から系列を処理 → [06_deep_learning/rnn.md](../06_deep_learning/rnn.md)
- **Transformer**: Self-Attentionで並列処理、長距離依存学習に優れる → [06_deep_learning/transformer.md](../06_deep_learning/transformer.md)
- **Self-Attention（自己注意機構）**: 系列内全要素間の関連度を並列計算 → [06_deep_learning/transformer.md](../06_deep_learning/transformer.md)
- **Multi-Head Attention**: 複数の視点で並列にAttentionを計算 → [06_deep_learning/transformer.md](../06_deep_learning/transformer.md)
- **位置エンコーディング（Positional Encoding）**: sin/cos関数で系列の順序情報を埋め込み → [06_deep_learning/transformer.md](../06_deep_learning/transformer.md)
- **BERT**: Transformer EncoderベースのNLPモデル、双方向学習 → [06_deep_learning/transformer.md](../06_deep_learning/transformer.md)
- **GPT**: Transformer Decoderベースの生成モデル、自己回帰 → [06_deep_learning/transformer.md](../06_deep_learning/transformer.md)
- **Vision Transformer（ViT）**: 画像をパッチ分割してTransformer適用 → [06_deep_learning/transformer.md](../06_deep_learning/transformer.md)
- **Query、Key、Value**: Self-Attentionの3つの基本ベクトル → [06_deep_learning/transformer.md](../06_deep_learning/transformer.md)
- **Masked Self-Attention**: Decoderで未来の情報を隠すマスク機構 → [06_deep_learning/transformer.md](../06_deep_learning/transformer.md)
- **Feed-Forward Network**: Transformerの各層でAttention後に適用される全結合層 → [06_deep_learning/transformer.md](../06_deep_learning/transformer.md)

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
- **音声認識（ASR）**: 音声波形→テキスト変換 → [07_ai_applications/speech_processing.md](../07_ai_applications/speech_processing.md)
- **音声合成（TTS）**: テキスト→音声生成 → [07_ai_applications/speech_processing.md](../07_ai_applications/speech_processing.md)
- **MFCC**: メル周波数ケプストラム係数、最も一般的な音響特徴量 → [07_ai_applications/speech_processing.md](../07_ai_applications/speech_processing.md)
- **CTC（Connectionist Temporal Classification）**: 音声とテキストの時間的対応付け → [07_ai_applications/speech_processing.md](../07_ai_applications/speech_processing.md)
- **WaveNet**: 1D-CNNによる音声波形生成 → [07_ai_applications/speech_processing.md](../07_ai_applications/speech_processing.md)
- **1D-CNN**: 時間方向の畳み込み、音声処理に有効 → [07_ai_applications/speech_processing.md](../07_ai_applications/speech_processing.md)
- **Whisper**: Transformer型の音声認識モデル、OpenAI開発 → [07_ai_applications/speech_processing.md](../07_ai_applications/speech_processing.md)

## 自然言語処理（NLP）
- **形態素解析（Morphological Analysis）**: 文を形態素（意味の最小単位）に分割し品詞を判別 → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **形態素（Morpheme）**: 意味を持つ最小単位（「走る」「ない」等） → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **品詞（Part-of-Speech, POS）**: 名詞・動詞・形容詞・助詞等の文法的カテゴリ → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **トークン化（Tokenization）**: 文章を単語や部分文字列に分割（品詞判別なし） → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **構文解析（Syntactic Analysis）**: 文の文法構造（主語・述語等）を解析 → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **係り受け解析（Dependency Parsing）**: 単語間の依存関係（修飾関係等）を解析 → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **意味解析（Semantic Analysis）**: 文の意味内容を理解し概念・関係を抽出 → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **固有表現抽出（Named Entity Recognition, NER）**: 人名・地名・組織名等を識別 → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **形態素解析器**: MeCab、Janome、Sudachi等のツール → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **分かち書き**: 形態素解析の分割結果を空白区切りで表記 → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **Word2Vec**: 単語を分散表現（ベクトル）に変換、Skip-gram/CBOW → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **ルールベース機械翻訳**: 文法規則と辞書を人手で作成（～1970年代後半） → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **統計的機械翻訳**: 対訳データから統計的に学習（1990年代～2010年代） → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **ニューラル機械翻訳（NMT）**: 深層学習による翻訳、Transformer使用（2015年代～） → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **Seq2Seq（Sequence-to-Sequence）**: Encoder-Decoderで系列変換、機械翻訳の基礎 → [07_ai_applications/natural_language_processing.md](../07_ai_applications/natural_language_processing.md)
- **プロンプトエンジニアリング（Prompt Engineering）**: 生成AIに望ましい出力を得るため指示文を設計・最適化 → [07_ai_applications/prompt_engineering.md](../07_ai_applications/prompt_engineering.md)
- **プロンプト（Prompt）**: AIへの入力指示文 → [07_ai_applications/prompt_engineering.md](../07_ai_applications/prompt_engineering.md)
- **Zero-shot学習**: 例示なしで指示のみで実行 → [07_ai_applications/prompt_engineering.md](../07_ai_applications/prompt_engineering.md)
- **Few-shot学習（In-context Learning）**: 少数の例示でパターン学習 → [07_ai_applications/prompt_engineering.md](../07_ai_applications/prompt_engineering.md)
- **Chain-of-Thought（CoT）プロンプティング**: 思考過程を促す手法、推論タスクで有効 → [07_ai_applications/prompt_engineering.md](../07_ai_applications/prompt_engineering.md)
- **システムプロンプト**: AIの役割・振る舞いを定義する初期設定 → [07_ai_applications/prompt_engineering.md](../07_ai_applications/prompt_engineering.md)
- **ハルシネーション（Hallucination）**: AIが事実でない情報を生成する現象 → [07_ai_applications/prompt_engineering.md](../07_ai_applications/prompt_engineering.md)
- **役割指定（Role Prompting）**: AIに専門家等の役割を与える手法 → [07_ai_applications/prompt_engineering.md](../07_ai_applications/prompt_engineering.md)
- **温度（Temperature）**: 生成のランダム性を制御するパラメータ → [07_ai_applications/prompt_engineering.md](../07_ai_applications/prompt_engineering.md)

## 法律・倫理
- **営業秘密**: 秘密管理性・有用性・非公知性の3要件で保護される情報 → [09_law_ethics/trade_secret.md](../09_law_ethics/trade_secret.md)
- **不正競争防止法**: 営業秘密の保護を規定する法律 → [09_law_ethics/trade_secret.md](../09_law_ethics/trade_secret.md)
- **AI倫理原則**: 人間中心・透明性・公平性・プライバシー・安全性・アカウンタビリティ・人間の監督 → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **透明性（Transparency）**: AIの動作原理・データ利用・判断根拠の説明可能性 → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **説明可能性（Explainability）**: 判断根拠を説明できる技術的能力（XAI） → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **分析対象者**: データが収集・分析される本人。直接ユーザーでなくても説明を受ける権利あり → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **アカウンタビリティ（Accountability）**: AI判断の結果に対する責任の所在明確化 → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **公平性（Fairness）**: 人種・性別等による不当な差別の排除 → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **XAI（Explainable AI）**: 説明可能なAI技術（LIME、SHAP等） → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **LIME**: 局所的な挙動を線形モデルで近似する説明手法 → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **SHAP**: シャープレイ値を用いた特徴量の貢献度計算 → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **説明可能性と性能のトレードオフ**: 複雑なモデルほど高性能だが説明困難 → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **リスクベースアプローチ**: 高リスク領域では説明可能性優先、低リスクは柔軟対応 → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **Ethics by Design**: 設計段階から倫理原則を組み込む開発手法 → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **Privacy by Design**: 設計段階からプライバシー保護を組み込む手法 → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **法的・倫理的検討のタイミング**: 企画・設計の初期段階から継続的に実施すべき → [09_law_ethics/ai_ethics_principles.md](../09_law_ethics/ai_ethics_principles.md)
- **カメラ画像利活用ガイドブック**: 経済産業省が策定したカメラ画像利活用の指針 → [09_law_ethics/camera_image_guidelines.md](../09_law_ethics/camera_image_guidelines.md)
- **事前告知**: カメラ撮影・利活用開始前の通知（原則） → [09_law_ethics/camera_image_guidelines.md](../09_law_ethics/camera_image_guidelines.md)
- **個人識別符号**: 顔認識データ等、個人情報保護法で定義される → [09_law_ethics/camera_image_guidelines.md](../09_law_ethics/camera_image_guidelines.md)
- **AI開発契約**: AI開発の委託契約、知的財産権・精度保証・データ権利処理を規定 → [09_law_ethics/ai_development_contract.md](../09_law_ethics/ai_development_contract.md)
- **学習済みモデルの著作権**: 契約で帰属を明示（開発者保持 or 委託者譲渡） → [09_law_ethics/ai_development_contract.md](../09_law_ethics/ai_development_contract.md)
- **ベストエフォート条項**: AI精度は確率的で完全保証困難、最善努力義務を規定 → [09_law_ethics/ai_development_contract.md](../09_law_ethics/ai_development_contract.md)
- **成果物の定義**: モデル・重み・ソースコード・ドキュメントの範囲を明確化 → [09_law_ethics/ai_development_contract.md](../09_law_ethics/ai_development_contract.md)
- **瑕疵担保責任**: プログラムのバグは瑕疵、精度未達は瑕疵でない → [09_law_ethics/ai_development_contract.md](../09_law_ethics/ai_development_contract.md)
- **職務著作**: 契約なき場合、著作権は開発者に帰属するデフォルトルール → [09_law_ethics/ai_development_contract.md](../09_law_ethics/ai_development_contract.md)

## 数学・統計
- **情報理論（Information Theory）**: 情報の量・伝達・圧縮を数学的に扱う理論 → [10_math_statistics/information_theory.md](../10_math_statistics/information_theory.md)
- **エントロピー（Entropy）**: 情報の不確実性、$H(X) = -\sum p(x) \log p(x)$ → [10_math_statistics/information_theory.md](../10_math_statistics/information_theory.md)
- **相互情報量（Mutual Information）**: 2変数の依存性、**I(X;Y)=0で独立** → [10_math_statistics/information_theory.md](../10_math_statistics/information_theory.md)
- **KLダイバージェンス（Kullback-Leibler Divergence）**: 分布間の乖離度、非対称 → [10_math_statistics/information_theory.md](../10_math_statistics/information_theory.md)
- **クロスエントロピー（Cross Entropy）**: 分類問題の損失関数 → [10_math_statistics/information_theory.md](../10_math_statistics/information_theory.md)
- **条件付きエントロピー（Conditional Entropy）**: Xを知った後のYの不確実性 → [10_math_statistics/information_theory.md](../10_math_statistics/information_theory.md)
- **情報利得（Information Gain）**: 決定木で分岐時のエントロピー減少量 → [10_math_statistics/information_theory.md](../10_math_statistics/information_theory.md)

---

## 更新履歴
- 2025/12/31: ε-greedy方策（探索と活用のトレードオフ、ε-減衰等）追加
- 2025/12/31: 情報理論（エントロピー、相互情報量、KLダイバージェンス、クロスエントロピー等）追加
- 2025/12/31: フレーム問題・常識の問題・シンボルグラウンディング問題（資格問題、Cycプロジェクト等）追加
- 2025/12/31: 機械翻訳の歴史（ルールベース、統計的、ニューラル機械翻訳、Seq2Seq等）追加
- 2025/12/31: ミニマックス法・αβ枝刈り（ゲーム木探索、評価関数、MCTS等）追加
- 2025/12/31: プロンプトエンジニアリング（Zero-shot、Few-shot、CoT、ハルシネーション等）追加
- 2025/12/31: 交差検証（k-分割、ホールドアウト法、LOOCV、層化k-分割等）追加
- 2025/12/31: 自然言語処理（形態素解析、構文解析、トークン化、係り受け解析、固有表現抽出等）追加
- 2025/12/30: AI開発契約（知的財産権、ベストエフォート条項、成果物定義、瑕疵担保責任等）追加
- 2025/12/30: Transformer（Self-Attention、Multi-Head Attention、位置エンコーディング、BERT、GPT等）追加
- 2025/12/30: 音声処理（音声認識、音声合成、MFCC、CTC、WaveNet、Whisper等）追加
- 2025/12/30: RNN（Encoder-Decoder、Seq2Seq、機械翻訳、勾配消失問題等）追加
- 2025/12/30: アンサンブル学習（バギング、ブースティング、ランダムフォレスト、XGBoost等）追加
- 2025/12/30: 法的・倫理的検討のタイミング、Ethics by Design、開発段階別チェックリスト追加
- 2025/12/30: 説明可能性と性能のトレードオフ、XAI技術（LIME、SHAP）追加
- 2025/12/30: 探索手法の適用領域（記号的AI vs サブシンボリックAI）追加
- 2025/12/30: AI倫理原則（透明性、説明可能性、分析対象者の権利等）追加
- 2025/12/30: 評価指標（RMSE、MAE、精度、適合率、再現率、F1、AUC-ROC、R²）追加
- 2025/12/28: 初期版作成（プランニング、探索アルゴリズム、CNN、Softmax、営業秘密）
