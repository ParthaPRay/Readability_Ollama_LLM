# ReadLLMAnOLLmAPI
# Localized LLM Response Readability Analysis 
# Run Once For Before First Ever Run Follow: python -m spacy download en_core_web_sm
# Partha Pratim Ray, 19/06/2025, parthapratimray1986@gmail.com

import gradio as gr
import requests
import textstat
import sqlite3
import os
import subprocess
import spacy
import textdescriptives as td
from datetime import datetime
import json

DB_FILE = "llm_readability.db"
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_TAGS_API = "http://localhost:11434/api/tags"
OLLAMA_STOP_API = "http://localhost:11434/api/stop"

# ---- spaCy + TextDescriptives setup ----
nlp = spacy.load("en_core_web_sm")
# Add all needed components for maximum metrics coverage
if "textdescriptives/readability" not in nlp.pipe_names:
    nlp.add_pipe("textdescriptives/readability")
if "textdescriptives/descriptive_stats" not in nlp.pipe_names:
    nlp.add_pipe("textdescriptives/descriptive_stats")
if "textdescriptives/dependency_distance" not in nlp.pipe_names:
    nlp.add_pipe("textdescriptives/dependency_distance")
print(nlp.pipe_names)    
if "textdescriptives/pos_proportions" not in nlp.pipe_names:
    nlp.add_pipe("textdescriptives/pos_proportions")

if "textdescriptives/quality" not in nlp.pipe_names:
    nlp.add_pipe("textdescriptives/quality")
if "textdescriptives/coherence" not in nlp.pipe_names:
    nlp.add_pipe("textdescriptives/coherence")
if "textdescriptives/information_theory" not in nlp.pipe_names:
    nlp.add_pipe("textdescriptives/information_theory")



# ---- DB Migration ----
def migrate_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model TEXT,
                prompt TEXT,
                llm_output TEXT,
                flesch_reading_ease REAL,
                flesch_kincaid_grade REAL,
                smog_index REAL,
                coleman_liau_index REAL,
                automated_readability_index REAL,
                dale_chall_readability_score REAL,
                difficult_words INTEGER,
                linsear_write_formula REAL,
                gunning_fog REAL,
                text_standard TEXT,
                fernandez_huerta REAL,
                szigriszt_pazos REAL,
                gutierrez_polini REAL,
                crawford REAL,
                gulpease_index REAL,
                osman REAL,
                syllable_count INTEGER,
                lexicon_count INTEGER,
                sentence_count INTEGER,
                char_count INTEGER,
                letter_count INTEGER,
                polysyllabcount INTEGER,
                monosyllabcount INTEGER,
                reading_time REAL,
                total_duration INTEGER,
                load_duration INTEGER,
                prompt_eval_count INTEGER,
                prompt_eval_duration INTEGER,
                eval_count INTEGER,
                eval_duration INTEGER,
                tokens_per_second REAL,
                textdescriptives_json TEXT,

                -- TextDescriptives metrics below
                td_lix REAL,
                td_rix REAL,
                td_n_characters INTEGER,
                td_n_sentences INTEGER,
                td_n_tokens INTEGER,
                td_n_unique_tokens INTEGER,
                td_proportion_unique_tokens REAL,
                td_sentence_length_mean REAL,
                td_sentence_length_median REAL,
                td_sentence_length_std REAL,
                td_syllables_per_token_mean REAL,
                td_syllables_per_token_median REAL,
                td_syllables_per_token_std REAL,
                td_token_length_mean REAL,
                td_token_length_median REAL,
                td_token_length_std REAL,
                td_dependency_distance_mean REAL,
                td_dependency_distance_std REAL,
                td_prop_adjacent_dependency_relation_mean REAL,
                td_prop_adjacent_dependency_relation_std REAL,
                td_pos_prop_ADJ REAL,
                td_pos_prop_ADP REAL,
                td_pos_prop_ADV REAL,
                td_pos_prop_AUX REAL,
                td_pos_prop_CCONJ REAL,
                td_pos_prop_DET REAL,
                td_pos_prop_INTJ REAL,
                td_pos_prop_NOUN REAL,
                td_pos_prop_NUM REAL,
                td_pos_prop_PART REAL,
                td_pos_prop_PRON REAL,
                td_pos_prop_PROPN REAL,
                td_pos_prop_PUNCT REAL,
                td_pos_prop_SCONJ REAL,
                td_pos_prop_SYM REAL,
                td_pos_prop_VERB REAL,
                td_pos_prop_X REAL,
                td_alpha_ratio REAL,
                td_contains_lorem_ipsum REAL,
                td_doc_length INTEGER,
                td_duplicate_line_chr_fraction REAL,
                td_duplicate_ngram_chr_fraction_10 REAL,
                td_duplicate_ngram_chr_fraction_5 REAL,
                td_duplicate_ngram_chr_fraction_6 REAL,
                td_duplicate_ngram_chr_fraction_7 REAL,
                td_duplicate_ngram_chr_fraction_8 REAL,
                td_duplicate_ngram_chr_fraction_9 REAL,
                td_duplicate_paragraph_chr_fraction REAL,
                td_mean_word_length REAL,
                td_n_stop_words INTEGER,
                td_oov_ratio REAL,
                td_passed_quality_check REAL,
                td_proportion_bullet_points REAL,
                td_proportion_ellipsis REAL,
                td_symbol_to_word_ratio_hash REAL,
                td_top_ngram_chr_fraction_2 REAL,
                td_top_ngram_chr_fraction_3 REAL,
                td_top_ngram_chr_fraction_4 REAL,
                td_first_order_coherence REAL,
                td_second_order_coherence REAL,
                td_entropy REAL,
                td_per_word_perplexity REAL,
                td_perplexity REAL
            )
        """)
        conn.commit()

print(f"[DEBUG] DB Check at: {os.path.abspath(os.getcwd())}")
migrate_db()



# --- TextDescriptives td metrics Extraction ---

def extract_textdescriptives_metrics(doc):
    """Extracts all required TD metrics as a dict of {column_name: value}"""
    def get(dct, *keys):
        v = dct
        for k in keys:
            v = v.get(k, None) if isinstance(v, dict) else None
        return v

    out = dict()
    # Readability
    out['td_lix'] = get(doc._.readability, 'lix')
    out['td_rix'] = get(doc._.readability, 'rix')
    # Descriptive Statistics
    out['td_n_characters'] = get(doc._.counts, 'n_characters')
    out['td_n_sentences'] = get(doc._.counts, 'n_sentences')
    out['td_n_tokens'] = get(doc._.counts, 'n_tokens')
    out['td_n_unique_tokens'] = get(doc._.counts, 'n_unique_tokens')
    out['td_proportion_unique_tokens'] = get(doc._.counts, 'proportion_unique_tokens')
    out['td_sentence_length_mean'] = get(doc._.sentence_length, 'sentence_length_mean')
    out['td_sentence_length_median'] = get(doc._.sentence_length, 'sentence_length_median')
    out['td_sentence_length_std'] = get(doc._.sentence_length, 'sentence_length_std')
    out['td_syllables_per_token_mean'] = get(doc._.syllables, 'syllables_per_token_mean')
    out['td_syllables_per_token_median'] = get(doc._.syllables, 'syllables_per_token_median')
    out['td_syllables_per_token_std'] = get(doc._.syllables, 'syllables_per_token_std')
    out['td_token_length_mean'] = get(doc._.token_length, 'token_length_mean')
    out['td_token_length_median'] = get(doc._.token_length, 'token_length_median')
    out['td_token_length_std'] = get(doc._.token_length, 'token_length_std')
    # Dependency Distance
    out['td_dependency_distance_mean'] = get(doc._.dependency_distance, 'dependency_distance_mean')
    out['td_dependency_distance_std'] = get(doc._.dependency_distance, 'dependency_distance_std')
    out['td_prop_adjacent_dependency_relation_mean'] = get(doc._.dependency_distance, 'prop_adjacent_dependency_relation_mean')
    out['td_prop_adjacent_dependency_relation_std'] = get(doc._.dependency_distance, 'prop_adjacent_dependency_relation_std')
    
    # POS Proportions
    for pos in ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X']:
        key = f'pos_prop_{pos}'
        out[f'td_pos_prop_{pos}'] = doc._.pos_proportions.get(key)
    
    # Quality
    q = getattr(doc._, 'quality', None)
    if q is not None and hasattr(q, 'to_flat_value_dict'):
        qd = q.to_flat_value_dict()
        out['td_alpha_ratio'] = qd.get('alpha_ratio')
        out['td_contains_lorem_ipsum'] = qd.get('contains_lorem ipsum')
        out['td_doc_length'] = qd.get('doc_length')
        out['td_duplicate_line_chr_fraction'] = qd.get('duplicate_line_chr_fraction')
        out['td_duplicate_ngram_chr_fraction_10'] = qd.get('duplicate_ngram_chr_fraction_10')
        out['td_duplicate_ngram_chr_fraction_5'] = qd.get('duplicate_ngram_chr_fraction_5')
        out['td_duplicate_ngram_chr_fraction_6'] = qd.get('duplicate_ngram_chr_fraction_6')
        out['td_duplicate_ngram_chr_fraction_7'] = qd.get('duplicate_ngram_chr_fraction_7')
        out['td_duplicate_ngram_chr_fraction_8'] = qd.get('duplicate_ngram_chr_fraction_8')
        out['td_duplicate_ngram_chr_fraction_9'] = qd.get('duplicate_ngram_chr_fraction_9')
        out['td_duplicate_paragraph_chr_fraction'] = qd.get('duplicate_paragraph_chr_fraction')
        out['td_mean_word_length'] = qd.get('mean_word_length')
        out['td_n_stop_words'] = qd.get('n_stop_words')
        out['td_oov_ratio'] = qd.get('oov_ratio')
        out['td_passed_quality_check'] = qd.get('passed_quality_check')
        out['td_proportion_bullet_points'] = qd.get('proportion_bullet_points')
        out['td_proportion_ellipsis'] = qd.get('proportion_ellipsis')
        out['td_symbol_to_word_ratio_hash'] = qd.get('symbol_to_word_ratio_#')
        out['td_top_ngram_chr_fraction_2'] = qd.get('top_ngram_chr_fraction_2')
        out['td_top_ngram_chr_fraction_3'] = qd.get('top_ngram_chr_fraction_3')
        out['td_top_ngram_chr_fraction_4'] = qd.get('top_ngram_chr_fraction_4')
    # Coherence
    out['td_first_order_coherence'] = get(doc._.coherence, 'first_order_coherence')
    out['td_second_order_coherence'] = get(doc._.coherence, 'second_order_coherence')
    # Information Theory
    out['td_entropy'] = get(doc._.information_theory, 'entropy')
    out['td_per_word_perplexity'] = get(doc._.information_theory, 'per_word_perplexity')
    out['td_perplexity'] = get(doc._.information_theory, 'perplexity')
    return out


# ---- Ollama Model Management ----
def get_llm_models():
    try:
        response = requests.get(OLLAMA_TAGS_API, timeout=5)
        models = response.json()["models"]
        out = [m["model"] for m in models]
        print("[DEBUG] Models:", out)
        return out
    except Exception as e:
        print("[WARN] Model list error:", e)
        return ["qwen2:0.5b"]

def unload_model(model):
    try:
        requests.post(OLLAMA_STOP_API, json={"model": model}, timeout=5)
        print(f"[INFO] Unloaded model: {model}")
    except Exception as e:
        print(f"[WARN] Unable to unload model {model}: {e}")

def preload_model(model):
    try:
        payload = {"model": model, "prompt": "", "keep_alive": -1}
        requests.post(OLLAMA_API, json=payload, timeout=120)
        print(f"[INFO] Preloaded model: {model}")
    except Exception as e:
        print(f"[WARN] Preload failed for {model}: {e}")

# ---- Readability and TD Metrics ----
def readability_metrics(text):
    def safe(f, default=0):
        try:
            return f(text)
        except Exception:
            return default

    # TextStat metrics
    metrics = {
        "flesch_reading_ease": safe(textstat.flesch_reading_ease, 0),
        "flesch_kincaid_grade": safe(textstat.flesch_kincaid_grade, 0),
        "smog_index": safe(textstat.smog_index, 0),
        "coleman_liau_index": safe(textstat.coleman_liau_index, 0),
        "automated_readability_index": safe(textstat.automated_readability_index, 0),
        "dale_chall_readability_score": safe(textstat.dale_chall_readability_score, 0),
        "difficult_words": safe(textstat.difficult_words, 0),
        "linsear_write_formula": safe(textstat.linsear_write_formula, 0),
        "gunning_fog": safe(textstat.gunning_fog, 0),
        "text_standard": safe(textstat.text_standard, "n/a"),
        "fernandez_huerta": safe(textstat.fernandez_huerta, 0),
        "szigriszt_pazos": safe(textstat.szigriszt_pazos, 0),
        "gutierrez_polini": safe(textstat.gutierrez_polini, 0),
        "crawford": safe(textstat.crawford, 0),
        "gulpease_index": safe(textstat.gulpease_index, 0),
        "osman": safe(textstat.osman, 0),
        "syllable_count": safe(textstat.syllable_count, 0),
        "lexicon_count": safe(textstat.lexicon_count, 0),
        "sentence_count": safe(textstat.sentence_count, 0),
        "char_count": safe(textstat.char_count, 0),
        "letter_count": safe(textstat.letter_count, 0),
        "polysyllabcount": safe(textstat.polysyllabcount, 0),
        "monosyllabcount": safe(textstat.monosyllabcount, 0),
        "reading_time": safe(textstat.reading_time, 0),
    }

    # --- TextDescriptives metrics ---
    doc = nlp(text)
    print([(token.text, token.pos_) for token in doc])
    print(doc._.pos_proportions)   # Should not be None, should be a dict.
    print('POS proportions:', doc._.pos_proportions)
    print('Pipeline:', nlp.pipe_names)
    print('Tokens:', [(t.text, t.pos_) for t in doc])
    td_metrics = {}

    # Readability (TextDescriptives-only)
    for key in ["lix", "rix"]:
        val = doc._.readability.get(key)
        if val is not None:
            td_metrics[f"Readability::{key}"] = val

    # Descriptive Statistics
    for k, v in doc._.token_length.items():
        td_metrics[f"Descriptive Statistics::token_length_{k}"] = v
    for k, v in doc._.sentence_length.items():
        td_metrics[f"Descriptive Statistics::sentence_length_{k}"] = v
    for k, v in doc._.syllables.items():
        td_metrics[f"Descriptive Statistics::syllables_{k}"] = v
    for k, v in doc._.counts.items():
        td_metrics[f"Descriptive Statistics::{k}"] = v

    # Dependency Distance
    depdist = doc._.dependency_distance
    for k in ["dependency_distance_mean", "dependency_distance_std",
              "prop_adjacent_dependency_relation_mean", "prop_adjacent_dependency_relation_std"]:
        val = depdist.get(k)
        if val is not None:
            td_metrics[f"Dependency Distance::{k}"] = val

    # POS Proportions
    for pos, v in doc._.pos_proportions.items():
        td_metrics[f"POS Proportions::{pos}"] = v

    # Quality
    try:
        quality_dict = doc._.quality.to_flat_value_dict()
    except Exception:
        quality_dict = {k: getattr(doc._.quality, k)
                        for k in dir(doc._.quality)
                        if not k.startswith('_') and not callable(getattr(doc._.quality, k))}
    for k, v in quality_dict.items():
        td_metrics[f"Quality::{k}"] = v
    td_metrics["Quality::passed_quality_check"] = doc._.passed_quality_check

    # Coherence
    for k, v in doc._.coherence.items():
        td_metrics[f"Coherence::{k}"] = v

    # Information Theory
    for k, v in doc._.information_theory.items():
        td_metrics[f"Information Theory::{k}"] = v

    return metrics, td_metrics, doc

def group_all_metrics_for_display(metrics, td_metrics, ollama_metrics):
    """
    Arrange all metrics for Gradio, with headings and subheadings:
    - Readability: (TD first, then TextStat)
    - Complexity (TextStat)
    - Grade Level (TextStat)
    - All other TD metrics grouped as before.
    - Ollama metrics at the end.
    """
    display = []

    # --- Readability Section ---
    display.append(["=== Readability ===", ""])
    # TextDescriptives readability metrics (not grade-level)
    display.append(["-- Readability (TextDescriptives) --", ""])
    # TD-only readability
    for k in ["Readability::lix", "Readability::rix"]:
        if k in td_metrics:
            display.append([k.split("::")[1], td_metrics[k]])
    # TextStat readability metrics (not grade-level)
    display.append(["-- Readability (TextStat) --", ""])
    textstat_readability = [
        "flesch_reading_ease", "dale_chall_readability_score", "fernandez_huerta",
        "szigriszt_pazos", "gutierrez_polini", "crawford", "gulpease_index", "osman"
    ]
    for k in textstat_readability:
        display.append([k, metrics.get(k)])

    # --- Complexity Section ---
    display.append(["=== Complexity (TextStat) ===", ""])
    textstat_complexity = [
        "syllable_count", "lexicon_count", "sentence_count", "char_count", "letter_count",
        "polysyllabcount", "monosyllabcount", "difficult_words", "reading_time"
    ]
    for k in textstat_complexity:
        display.append([k, metrics.get(k)])

    # --- Grade Level Section ---
    display.append(["=== Grade Level (TextStat) ===", ""])
    textstat_grade = [
        "flesch_kincaid_grade", "smog_index", "coleman_liau_index",
        "automated_readability_index", "linsear_write_formula", "gunning_fog", "text_standard"
    ]
    for k in textstat_grade:
        display.append([k, metrics.get(k)])

    # --- TextDescriptives Sections ---
    td_groups = [
        "Descriptive Statistics", "Dependency Distance",
        "POS Proportions", "Quality", "Coherence", "Information Theory"
    ]
    for group in td_groups:
        section_keys = [k for k in td_metrics if k.startswith(f"{group}::")]
        if section_keys:
            display.append([f"=== {group} (TextDescriptives) ===", ""])
            for sk in sorted(section_keys):
                label = sk.split("::", 1)[1]
                display.append([label, td_metrics[sk]])

    # --- Ollama metrics at end ---
    display.append(["=== Metrics (Ollama) ===", ""])
    for k, v in ollama_metrics.items():
        display.append([k, v])

    return display

def save_to_db(timestamp, model, prompt, llm_output, metrics, ollama_metrics, td_metrics, all_td_metrics_flat, td_extracted):
    # flatten fixed metrics
    values = [
        timestamp, model, prompt, llm_output,
        metrics["flesch_reading_ease"], metrics["flesch_kincaid_grade"], metrics["smog_index"],
        metrics["coleman_liau_index"], metrics["automated_readability_index"], metrics["dale_chall_readability_score"],
        metrics["difficult_words"], metrics["linsear_write_formula"], metrics["gunning_fog"], metrics["text_standard"],
        metrics["fernandez_huerta"], metrics["szigriszt_pazos"], metrics["gutierrez_polini"], metrics["crawford"],
        metrics["gulpease_index"], metrics["osman"], metrics["syllable_count"], metrics["lexicon_count"],
        metrics["sentence_count"], metrics["char_count"], metrics["letter_count"], metrics["polysyllabcount"],
        metrics["monosyllabcount"], metrics["reading_time"],
        ollama_metrics.get("total_duration"), ollama_metrics.get("load_duration"),
        ollama_metrics.get("prompt_eval_count"), ollama_metrics.get("prompt_eval_duration"),
        ollama_metrics.get("eval_count"), ollama_metrics.get("eval_duration"), ollama_metrics.get("tokens_per_second"),
        json.dumps(all_td_metrics_flat, ensure_ascii=False)  
    ]
    # Now add the TD metrics, in correct order!
    td_column_names = [
        'td_lix', 'td_rix', 'td_n_characters', 'td_n_sentences', 'td_n_tokens', 'td_n_unique_tokens', 'td_proportion_unique_tokens',
        'td_sentence_length_mean', 'td_sentence_length_median', 'td_sentence_length_std', 'td_syllables_per_token_mean',
        'td_syllables_per_token_median', 'td_syllables_per_token_std', 'td_token_length_mean', 'td_token_length_median',
        'td_token_length_std', 'td_dependency_distance_mean', 'td_dependency_distance_std',
        'td_prop_adjacent_dependency_relation_mean', 'td_prop_adjacent_dependency_relation_std',
        'td_pos_prop_ADJ', 'td_pos_prop_ADP', 'td_pos_prop_ADV', 'td_pos_prop_AUX', 'td_pos_prop_CCONJ', 'td_pos_prop_DET',
        'td_pos_prop_INTJ', 'td_pos_prop_NOUN', 'td_pos_prop_NUM', 'td_pos_prop_PART', 'td_pos_prop_PRON', 'td_pos_prop_PROPN',
        'td_pos_prop_PUNCT', 'td_pos_prop_SCONJ', 'td_pos_prop_SYM', 'td_pos_prop_VERB', 'td_pos_prop_X',
        'td_alpha_ratio', 'td_contains_lorem_ipsum', 'td_doc_length', 'td_duplicate_line_chr_fraction',
        'td_duplicate_ngram_chr_fraction_10', 'td_duplicate_ngram_chr_fraction_5', 'td_duplicate_ngram_chr_fraction_6',
        'td_duplicate_ngram_chr_fraction_7', 'td_duplicate_ngram_chr_fraction_8', 'td_duplicate_ngram_chr_fraction_9',
        'td_duplicate_paragraph_chr_fraction', 'td_mean_word_length', 'td_n_stop_words', 'td_oov_ratio', 'td_passed_quality_check',
        'td_proportion_bullet_points', 'td_proportion_ellipsis', 'td_symbol_to_word_ratio_hash', 'td_top_ngram_chr_fraction_2',
        'td_top_ngram_chr_fraction_3', 'td_top_ngram_chr_fraction_4', 'td_first_order_coherence',
        'td_second_order_coherence', 'td_entropy', 'td_per_word_perplexity', 'td_perplexity'
    ]
    values.extend([td_extracted.get(name) for name in td_column_names])
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        print(f"Fixed values: {len(values) - len(td_column_names)}")
        print(f"TD values: {len(td_column_names)}")
        print(f"Total values: {len(values)}")
        print(f"Total columns (SQL): {36 + len(td_column_names)}")
        c.execute(f"""
            INSERT INTO results (
                timestamp, model, prompt, llm_output, flesch_reading_ease, flesch_kincaid_grade, smog_index, coleman_liau_index,
                automated_readability_index, dale_chall_readability_score, difficult_words, linsear_write_formula, gunning_fog,
                text_standard, fernandez_huerta, szigriszt_pazos, gutierrez_polini, crawford, gulpease_index, osman,
                syllable_count, lexicon_count, sentence_count, char_count, letter_count, polysyllabcount, monosyllabcount,
                reading_time, total_duration, load_duration, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration,
                tokens_per_second, textdescriptives_json,
                {', '.join(td_column_names)}
            ) VALUES (
                {','.join(['?'] * (36 + len(td_column_names)))}
            )
        """, values)

        conn.commit()
        
    # Terminal output with headings
    print("\n[SQLITE LOGGED] Saved the following to DB:")
    labels = [
        "timestamp", "model", "prompt", "llm_output",
        "flesch_reading_ease", "flesch_kincaid_grade", "smog_index", "coleman_liau_index",
        "automated_readability_index", "dale_chall_readability_score", "difficult_words",
        "linsear_write_formula", "gunning_fog", "text_standard", "fernandez_huerta",
        "szigriszt_pazos", "gutierrez_polini", "crawford", "gulpease_index", "osman",
        "syllable_count", "lexicon_count", "sentence_count", "char_count", "letter_count",
        "polysyllabcount", "monosyllabcount", "reading_time",
        "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration",
        "eval_count", "eval_duration", "tokens_per_second", "textdescriptives_json"
    ]
    for label, val in zip(labels, values):
        val_str = (str(val)[:100] + "...") if label in ["prompt", "llm_output"] and val and len(str(val)) > 100 else val
        print(f"  {label}: {val_str}")
    print("\n[TextDescriptives Grouped Metrics]")
    for row in group_all_metrics_for_display(metrics, td_metrics, ollama_metrics):
        print(f"{row[0]:>30} : {row[1]}")
    print("-" * 60)

def infer_and_analyze(prompt, model):
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": -1}
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        llm_out = data.get("response", "")
        ollama_metrics = {
            "total_duration": data.get("total_duration"),
            "load_duration": data.get("load_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "tokens_per_second": (float(data.get("eval_count"))/float(data.get("eval_duration"))*1e9)
            if data.get("eval_count") and data.get("eval_duration") else None
        }
    except Exception as ex:
        return f"Error calling Ollama: {ex}", [], ""

    metrics, td_metrics, doc = readability_metrics(llm_out)
    td_extracted = extract_textdescriptives_metrics(doc)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # For archival, you can do:
    all_td_metrics_flat = td_extracted  # or use the JSON if you want, for now this is enough
    save_to_db(timestamp, model, prompt, llm_out, metrics, ollama_metrics, td_metrics, all_td_metrics_flat, td_extracted)


    # Group and show all metrics for Gradio (and terminal)
    display_metrics = group_all_metrics_for_display(metrics, td_metrics, ollama_metrics)
    metrics_table = "\n".join(f"{row[0]}: {row[1]}" for row in display_metrics)
    return llm_out, display_metrics, metrics_table


# ---- Gradio Interface ----
all_models = get_llm_models()

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("""
    # 🌟 LLM Readability Analyzer
    Analyze text readability and Ollama runtime metrics with local models on Raspberry Pi 4B.
    """)

    model_sel = gr.Dropdown(label="Select LLM Model", choices=all_models, value=all_models[0])
    prompt_box = gr.Textbox(label="Enter Prompt", placeholder="Type prompt here", lines=3)

    submit_btn = gr.Button("🔍 Analyze")
    llm_out = gr.Textbox(label="LLM Response", lines=5)
    metrics_out = gr.Dataframe(headers=["Metric", "Value"], datatype=["str", "str"], label="Metrics", interactive=False)
    raw_metrics = gr.Textbox(visible=False)

    state_previous_model = gr.State(all_models[0])

    def change_model(new_model, prev_model):
        if new_model != prev_model:
            unload_model(prev_model)
            preload_model(new_model)
        return new_model

    model_sel.change(change_model, inputs=[model_sel, state_previous_model], outputs=state_previous_model)

    submit_btn.click(
        fn=infer_and_analyze,
        inputs=[prompt_box, model_sel],
        outputs=[llm_out, metrics_out, raw_metrics]
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
