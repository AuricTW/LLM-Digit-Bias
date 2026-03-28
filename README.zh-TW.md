# LLM Digit Bias

[English README](./README.md)

本專案公開了一套研究框架，用於分析大型語言模型在離散選擇任務中的偏差。當前版本聚焦於一個簡單但具有辨識力的設定：要求模型從 `1` 到 `9` 中選擇一個數字，且只輸出一個阿拉伯數字。

這套框架同時支援行為層與機率層分析。除了重複抽樣與輸出頻率統計之外，本地 `transformers` 路線也支援 tokenizer audit 與 audited-surface candidate probability 分析，因此可以比較模型最終輸出與其候選偏好之間的關係。

## 專案範圍

- 支援多種 prompt、排列方式與 temperature 的重複抽樣
- 嚴格解析：只有 `1` 到 `9` 的單一裸數字才算有效輸出
- 模組化 client layer，可擴充到 `transformers`、OpenAI-compatible API 與 vLLM
- 分析模組包含 frequency、invalid rate、chi-square、entropy、KL divergence 與 Jensen-Shannon divergence
- 提供 ordering 與 position 分析，用於比較 digit identity effect 與 position effect
- 提供可直接整理為研究報告的跨模型摘要與表格

## 專案結構

```text
src/                 核心框架與分析程式
prompts/             提示詞模板
configs/             代表性的實驗設定
results/processed/   本次公開版納入的聚合結果
docs/                公開說明與報告骨架
```

## 安裝

基本安裝：

```bash
python -m pip install -e .
```

可選依賴：

```bash
python -m pip install -e .[openai]
python -m pip install -e .[transformers]
python -m pip install -e .[vllm]
python -m pip install accelerate bitsandbytes
```

## 基本重現方式

框架 smoke test：

```bash
python -m src.runner.main --config configs/quickstart_mock.json
```

重建本公開版內附的聚合比較結果：

```bash
python -m src.analysis.cross_model_comparison_main
python -m src.analysis.cross_model_plots_main
python -m src.analysis.report_artifacts_main
```

## 本次公開版包含的結果

本次公開版已附上 `results/processed/` 內的聚合結果，包括：

- `cross_model_comparison.csv`
- `cross_model_comparison.md`
- `cross_model_temperature_details.csv`
- `cross_model_figures/`
- `report_ready/main_comparison_table.md`
- `report_ready/supplementary_temperature_table.md`
- `report_ready/appendix_cases_table.md`
- `report_ready/report_results_skeleton.md`

這些檔案足以呈現目前的跨模型研究結論，而不需要完整公開全部 raw run。

## 釋出範圍

本公開版提供理解與重現已釋出分析所需的程式碼、代表性設定檔與整理後結果，但不包含完整的 raw experimental dumps 與本地中間產物。

未納入本次公開版的內容包括：

- `results/raw/`
- tokenizer audit 的 raw dumps
- 本地暫存檔
- `*.egg-info` 等 build artifacts

上述界線與理由可參考 [PUBLIC_EXPORT_NOTES.md](./PUBLIC_EXPORT_NOTES.md)。

## 目前的研究結論

目前已釋出的分析支持三個較穩定的結論。

- 在這個任務中，多數模型的輸出都明顯偏離均勻隨機。
- 若可取得機率層資訊，偏差通常不只體現在最終輸出，也會體現在模型對候選數字的偏好上。
- 偏差機制具有模型家族差異：有些模型更偏向特定 digit identity，有些更受列表位置影響，也有模型呈現 mixed 或 temperature-sensitive 的型態。
