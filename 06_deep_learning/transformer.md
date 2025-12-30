# Transformer

## 要点（試験用）
- **Self-Attentionメカニズム**により、系列内の全要素間の関係を並列に計算。RNNの再帰処理を排除し、**長距離依存関係の学習**と**並列処理**を実現。
- 位置エンコーディングで系列の順序情報を付与。Multi-Head Attentionで多様な関係性を同時に捉える。
- 自然言語処理の主流アーキテクチャ（BERT、GPT等）。画像（Vision Transformer）にも適用拡大。

## 定義
**Transformer**は、再帰構造（RNN）や畳み込み（CNN）を使わず、**Attention機構のみ**で系列データを処理する深層学習アーキテクチャ。2017年にVaswaniらが提案（"Attention Is All You Need"）。

- **Self-Attention（自己注意機構）**：入力系列内の各要素が、他の全要素との関連度を計算し、文脈を考慮した表現を生成
- **Multi-Head Attention**：複数の異なる視点（Head）で並列にAttentionを計算し、結合
- **位置エンコーディング（Positional Encoding）**：系列の順序情報を明示的に埋め込み
- **Encoder-Decoder構造**：機械翻訳等ではEncoder（入力を符号化）とDecoder（出力を生成）を組み合わせる

## 重要キーワード
- **Self-Attention（自己注意機構）**：系列内の全要素間の関連性を計算。Query、Key、Valueの3つのベクトルで計算
- **Multi-Head Attention**：異なる表現部分空間で並列にAttentionを計算（通常8〜16個のHead）
- **位置エンコーディング（Positional Encoding）**：sin/cos関数等で系列の位置情報を埋め込み
- **Encoder-Decoder**：Encoderが入力を処理、Decoderが出力を生成（翻訳・要約等）
- **Feed-Forward Network（FFN）**：各層でAttention後に適用される全結合層
- **残差接続（Residual Connection）**：各サブレイヤーの入出力を加算（学習安定化）
- **Layer Normalization**：各層で正規化を実施
- **Masked Self-Attention**：Decoderで未来の情報を見ないようマスク
- **BERT（Bidirectional Encoder Representations from Transformers）**：Encoderのみ、双方向学習
- **GPT（Generative Pre-trained Transformer）**：Decoderのみ、自己回帰的生成
- **Vision Transformer（ViT）**：画像をパッチに分割してTransformer適用

## 詳細

### 背景と動機
**従来のRNN/LSTMの課題**：
- **再帰処理が必須**：前の時刻の計算完了まで次に進めない → 並列化困難、学習遅い
- **長距離依存の限界**：系列が長いと勾配消失/爆発のリスク、情報の減衰
- **計算コスト**：長い系列では計算量が線形に増加

**CNNの限界**：
- 局所的な特徴抽出に特化、長距離依存は多層積層が必要
- 系列全体の関係を直接捉えにくい

**Transformerの革新**：
- **Attention機構のみ**で系列を処理 → 全要素間の関係を一度に並列計算
- **並列処理**可能 → GPUの性能を最大活用、学習高速化
- **長距離依存**を直接モデル化 → 任意の距離の要素間も直接関連付け

### Self-Attentionの仕組み

入力系列 $\mathbf{X} = [x_1, x_2, ..., x_n]$ に対し：

1. **Query、Key、Valueの生成**：
   - $\mathbf{Q} = \mathbf{X} \mathbf{W}_Q$（Query：問い合わせ）
   - $\mathbf{K} = \mathbf{X} \mathbf{W}_K$（Key：キー）
   - $\mathbf{V} = \mathbf{X} \mathbf{W}_V$（Value：値）

2. **Attention重みの計算**：
   $$
   \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}
   $$
   - $\mathbf{Q}\mathbf{K}^T$：各要素間の関連度スコア
   - $\sqrt{d_k}$：スケーリング（勾配安定化）
   - softmax：確率分布に変換
   - $\mathbf{V}$：重み付き和で最終的な表現を生成

3. **直観**：
   - 各単語（Query）が、他の全単語（Key）との類似度を計算
   - 類似度に応じて他の単語（Value）を重み付けて集約
   - 文脈を考慮した新しい表現を生成

### Multi-Head Attentionの利点

```
[入力]
  ↓
[線形変換 × h個のHead]
  ↓
[並列にSelf-Attention × h個]
  ↓
[結合 + 線形変換]
  ↓
[出力]
```

- **異なる視点**：各Headが異なる表現部分空間で異なる関係性を学習
  - Head1：構文的関係（主語-述語）
  - Head2：意味的関係（類義語）
  - Head3：距離的関係（近接単語）
- **表現力向上**：複数の関係性を同時に捉える

### 位置エンコーディング

Attention機構には**系列の順序情報がない**ため、明示的に位置を埋め込む：

$$
\begin{align}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{align}
$$

- $pos$：単語の位置、$i$：次元インデックス、$d$：埋め込み次元数
- sin/cosの組み合わせで相対位置も学習可能

### Encoder-Decoder構造

**Encoder（符号化器）**：
```
入力 → 埋め込み + 位置エンコーディング
  ↓
[Multi-Head Self-Attention]
  ↓ 残差接続 + Layer Norm
[Feed-Forward Network]
  ↓ 残差接続 + Layer Norm
[× N層（6〜12層）]
  ↓
文脈表現
```

**Decoder（復号化器）**：
```
出力（shifted right）→ 埋め込み + 位置エンコーディング
  ↓
[Masked Multi-Head Self-Attention]（未来を見ない）
  ↓ 残差接続 + Layer Norm
[Multi-Head Attention]（Encoderの出力を参照）
  ↓ 残差接続 + Layer Norm
[Feed-Forward Network]
  ↓ 残差接続 + Layer Norm
[× N層]
  ↓
出力系列
```

## 実例

### 機械翻訳の例

**入力（英語）**："I love artificial intelligence"
**Encoderの処理**：
- 各単語が他の全単語との関係を学習
  - "love" → "I"（主語）、"intelligence"（目的語）との高い関連度
  - "artificial" → "intelligence"（修飾関係）との高い関連度
- 文脈を考慮した表現を生成

**Decoderの処理**：
- 出力を1単語ずつ生成（自己回帰）
- 「私は」→「人工」→「知能が」→「好きです」
- 各時点でEncoderの出力とこれまでの出力を参照

### 代表的なTransformerモデル

| モデル | 構造 | 用途 | 特徴 |
|--------|------|------|------|
| **BERT** | Encoderのみ | 文分類、固有表現抽出 | 双方向（前後の文脈）、Masked Language Model |
| **GPT** | Decoderのみ | 文章生成、対話 | 単方向（左→右）、自己回帰生成 |
| **T5** | Encoder-Decoder | 翻訳、要約、QA | すべてをText-to-Text問題として統一 |
| **Vision Transformer** | Encoderのみ | 画像分類 | 画像を16×16パッチに分割、系列として処理 |

### 計算量の比較

| アーキテクチャ | 時間計算量 | 並列化 | 長距離依存 |
|---------------|-----------|-------|-----------|
| **RNN** | $O(n)$（逐次） | 不可 | 困難（勾配消失） |
| **CNN** | $O(n/k)$（k:カーネルサイズ） | 可能 | 多層必要 |
| **Transformer** | $O(n^2)$（全ペア） | 可能 | 直接学習 |

- **Transformerの課題**：系列長nの2乗の計算量 → 長い系列では高コスト
- **解決策**：Sparse Attention、Linformer、Performer等の効率化手法

## 試験での問われ方

### ✅ 最も適切な特徴（正解パターン）

1. **「Self-Attentionメカニズムにより、系列内の全要素間の関係を並列に計算する」**
   - ✅ Transformerの核心的特徴

2. **「RNNと異なり、再帰的な処理を行わないため並列化が可能」**
   - ✅ RNNとの主要な違い

3. **「長距離依存関係の学習に優れ、任意の距離の要素間を直接関連付けられる」**
   - ✅ 重要な利点

4. **「位置エンコーディングで系列の順序情報を明示的に埋め込む」**
   - ✅ Attentionに順序情報がないための対策

5. **「Multi-Head Attentionで異なる視点から複数の関係性を同時に捉える」**
   - ✅ 表現力向上の仕組み

### ❌ 最も不適切な特徴（ひっかけパターン）

1. **「隠れ状態を次の時刻に順次引き継いで処理する」**
   - ❌ **RNN/LSTMの特徴**。Transformerは再帰なし

2. **「畳み込み層を主要な構成要素として局所的な特徴を抽出する」**
   - ❌ **CNNの特徴**。Transformerは畳み込み不使用

3. **「勾配消失問題を完全に解決するためGate機構を使用する」**
   - ❌ **LSTM/GRUの特徴**。Transformerは残差接続とLayer Norm使用

4. **「計算量が系列長に比例するため長い系列でも効率的」**
   - ❌ 計算量は系列長の**2乗**に比例（$O(n^2)$）。長い系列では高コスト

5. **「主に画像認識タスクに特化して設計された」**
   - ❌ 主に**自然言語処理**向け。画像はViTで後から適用

### 比較されやすい概念との違い

| 観点 | Transformer | RNN/LSTM | CNN |
|------|-------------|----------|-----|
| **処理方式** | 並列（Attention） | 逐次（再帰） | 並列（畳み込み） |
| **長距離依存** | 直接学習（全ペア） | 困難（勾配消失） | 多層積層必要 |
| **計算量** | $O(n^2)$ | $O(n)$ | $O(n)$ |
| **並列化** | 可能 | 不可 | 可能 |
| **順序情報** | 位置エンコーディング | 暗黙的（時系列） | カーネル範囲内 |
| **主な用途** | NLP、翻訳 | 系列データ全般 | 画像、音声 |

### 引っ掛けポイント

1. **「再帰的処理」**：
   - Transformerは再帰**なし** ↔ RNNは再帰**あり**

2. **「畳み込み」**：
   - Transformerは畳み込み**不使用** ↔ CNNは畳み込み**主体**

3. **「計算量」**：
   - Transformerは$O(n^2)$で長い系列では**不利** ↔ 「効率的」は誤り

4. **「Gate機構」**：
   - LSTMのGate ≠ TransformerのAttention

5. **「双方向」**：
   - BERT（Encoder）は双方向、GPT（Decoder）は単方向
   - 「Transformerは常に双方向」は誤り

6. **「画像専用」**：
   - 元々はNLP向け、ViTで画像にも適用拡大

## 補足

### 実務的観点

1. **学習コスト**：
   - 大規模データと計算資源が必要（GPT-3: 175B parameters）
   - 事前学習済みモデル（BERT、GPT）の活用が一般的

2. **系列長の制限**：
   - $O(n^2)$の計算量 → 長い文書では分割・要約が必要
   - BERTは最大512トークン、GPT-3は2048トークン

3. **ファインチューニング**：
   - 事前学習済みモデルを特定タスクに適応
   - 少量データでも高性能達成可能

4. **解釈性**：
   - Attention重みの可視化で「どの単語に注目したか」分析可能
   - ただし、複雑なモデルの完全理解は困難

### 関連トピック
- [RNN（再帰型ニューラルネットワーク）](rnn.md)：系列処理の従来手法
- [LSTM/GRU](lstm_gru.md)：長距離依存対策のRNN改良版
- [自然言語処理](../07_ai_applications/natural_language_processing.md)：Transformerの主要応用領域
- [Attention機構](neural_network_basics.md)：Transformerの基礎となる仕組み
