# ReadLLMAnOLLmAPI

Readability LLM Analyzer for Ollama API on Localized Resource Constained Edge (Raspberry Pi)

> **Analyze the readability and structural metrics of responses generated by small, quantized LLMs running locally via [Ollama](https://github.com/ollama/ollama) and view runtime metrics interactively with Gradio.**

---

## ✨ Project Overview

This tool enables researchers and practitioners to **evaluate the readability, complexity, and linguistic structure** of outputs from localized small-scale LLMs (quantized for edge/low-resource devices) using the Ollama inference engine.  
Metrics are calculated using [TextStat](https://github.com/textstat/textstat) and [TextDescriptives](https://hlasse.github.io/TextDescriptives/index.html) and are stored in a local SQLite database for further analysis.  
A [Gradio](https://gradio.app/) web interface allows for interactive prompt-response analysis and metrics exploration.

---

## 🚀 Features

- **Supports Small Quantized LLMs**: Works with any local model served by [Ollama](https://github.com/ollama/ollama) API.
- **Comprehensive Metrics**: Extracts classic readability indices, spaCy linguistic features, TextDescriptives metrics (including POS proportions, coherence, information theory, and more).
- **Interactive Gradio UI**: Type a prompt, select a model, and view LLM output plus all computed metrics in a clear dashboard.
- **Metrics Logging**: Every inference is logged into a local SQLite database for reproducible, longitudinal, or comparative studies.
- **Runtime Statistics**: Captures Ollama-specific runtime stats (token/s, duration, etc.) alongside text metrics.
- **Designed for Edge**: Runs efficiently on a Raspberry Pi 4B or similar edge device, and any system supported by Ollama.

---

## 🛠️ Installation

1. **Clone this repo**  
    ```bash
    git clone https://github.com/YOUR-USERNAME/llm-readability-analyzer.git
    cd llm-readability-analyzer
    ```

2. **Install dependencies**  
    We recommend using a virtual environment (e.g., `venv` or `conda`):

    ```bash
    pip install -r requirements.txt
    ```

    **`requirements.txt` contents:**
    ```
    gradio
    requests
    textstat==0.7.7
    spacy
    textdescriptives==2.8.4
    ```

3. **Download the spaCy language model** (run once):
    ```bash
    python -m spacy download en_core_web_sm
    ```

4. **Start the Ollama server and pull any quantized model you want to use**  
   (see [Ollama documentation](https://ollama.com/library)).

---

## ⚡ Usage

1. **Launch the app:**
    ```bash
    python ReadLLMAnOLLmAPI.py
    ```

2. **Visit the Gradio UI:**  
   Open [http://localhost:7860](http://localhost:7860) in your browser.

3. **Workflow:**
   - Select an available LLM model (auto-discovered from Ollama).
   - Enter a prompt.
   - Click "Analyze".
   - View the model's response, readability indices, TextDescriptives metrics, and Ollama runtime statistics.
   - All results are also logged in `llm_readability.db` (SQLite).

---

## 📊 What Metrics Are Measured?

- **Readability:** Flesch Reading Ease, SMOG, Coleman-Liau, Dale-Chall, etc.
- **Complexity & Structure:** Syllable/word/sentence counts, polysyllabic/monosyllabic word ratios, word length stats.
- **POS Proportions:** Fraction of nouns, verbs, adjectives, etc.
- **TextDescriptives:** Coherence, information theory (entropy, perplexity), quality metrics (repetition, bullets, ellipsis, OOV ratio), dependency distance, and more.
- **Ollama Runtime:** Model loading time, prompt evaluation stats, tokens per second.

---

## 📝 Example Output

- The UI displays all metrics in a tabular format, with headings.
- The SQLite DB (`llm_readability.db`) can be analyzed further with any SQL tool or exported as CSV for research.

---

## 🧑‍💻 Code Organization

- **`ReadLLMAnOLLmAPI.py`** — Main application logic (Gradio UI, metrics computation, DB logging)
- **`requirements.txt`** — List of dependencies

---

## 🔗 References

- [Ollama Documentation](https://ollama.com/)
- [TextStat](https://github.com/textstat/textstat)
- [spaCy](https://spacy.io/)
- [TextDescriptives](https://hlasse.github.io/TextDescriptives/index.html)
- [Gradio](https://gradio.app/)

---

## 📝 Notes

- This project is optimized for **localized, resource-constrained environments** (Raspberry Pi, Jetson, low-end cloud, etc.), but will run on any Python 3.8+ system.
- Make sure the Ollama server is running before starting the app.
- All metrics are calculated **locally** for data privacy and research reproducibility.

---

## 👤 Author

Maintained by [Partha Pratim Ray].  
Pull requests and issues welcome!

---

