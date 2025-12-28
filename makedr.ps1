# ===== 基本設定 =====
$root = "g_test"

# フォルダとファイル定義
$structure = @{
    "00_index" = @(
        "README.md",
        "syllabus_map.md"
    )
    "01_ai_overview" = @(
        "what_is_ai.md",
        "strong_vs_weak_ai.md",
        "symbolic_vs_subsymbolic.md",
        "ai_vs_ml_vs_dl.md"
    )
    "02_ai_history" = @(
        "first_ai_boom.md",
        "second_ai_boom.md",
        "third_ai_boom.md",
        "ai_winter.md"
    )
    "03_ai_definition" = @(
        "turing_test.md",
        "rational_agent.md",
        "frame_problem.md",
        "chinese_room.md"
    )
    "04_knowledge_representation" = @(
        "rule_based_system.md",
        "ontology.md",
        "semantic_network.md",
        "inference.md"
    )
    "05_machine_learning" = @(
        "supervised_learning.md",
        "unsupervised_learning.md",
        "reinforcement_learning.md",
        "feature_engineering.md",
        "overfitting_underfitting.md",
        "cross_validation.md",
        "evaluation_metrics.md"
    )
    "06_deep_learning" = @(
        "neural_network_basics.md",
        "backpropagation.md",
        "cnn.md",
        "rnn.md",
        "lstm_gru.md",
        "transformer.md",
        "generative_models.md"
    )
    "07_ai_applications" = @(
        "image_recognition.md",
        "natural_language_processing.md",
        "speech_processing.md",
        "recommendation_system.md"
    )
    "08_ai_society" = @(
        "ai_business_use.md",
        "dx_and_ai.md",
        "data_utilization.md",
        "ai_limitations.md"
    )
    "09_law_ethics" = @(
        "personal_data_protection.md",
        "gdpr.md",
        "ai_ethics_principles.md",
        "bias_and_fairness.md"
    )
    "10_math_statistics" = @(
        "probability.md",
        "statistics.md",
        "linear_algebra.md",
        "optimization.md"
    )
    "90_mock_exam_notes" = @(
        "weak_points.md",
        "mistake_patterns.md",
        "final_review.md"
    )
    "99_templates" = @(
        "g_test_note_template.md"
    )
}

# ===== G検定用 Markdown テンプレ =====
$template = @"
# テーマ名

## 要点（試験用）
- 3行以内で説明できるか

## 定義
- 公式・一般的な定義

## 重要キーワード
- 用語1：
- 用語2：

## 試験での問われ方
- 比較されやすい概念：
- 引っ掛けポイント：

## 補足
- 実務的観点（任意）
"@

# ===== 作成処理 =====
New-Item -ItemType Directory -Path $root -Force | Out-Null

foreach ($folder in $structure.Keys) {
    $folderPath = Join-Path $root $folder
    New-Item -ItemType Directory -Path $folderPath -Force | Out-Null

    foreach ($file in $structure[$folder]) {
        $filePath = Join-Path $folderPath $file
        if (-not (Test-Path $filePath)) {
            $template | Out-File -FilePath $filePath -Encoding UTF8
        }
    }
}

Write-Host "G検定用フォルダ・Markdownテンプレート作成完了"
