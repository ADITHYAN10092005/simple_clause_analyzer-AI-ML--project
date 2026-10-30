# Legal Document Clause Risk Analyzer
**KTU B.Tech AI & ML Course Mini Project**

A beginner-friendly, hybrid Artificial Intelligence system that reviews contracts (PDF, DOCX, TXT), classifies individual clauses with Machine Learning, and applies a goal-based forward-chaining rule agent to flag unfair or high-risk legal terms to protect signers.

---

## 📌 Project Overview & Architecture

Modern contracts often bury aggressive terms (such as one-sided termination windows, unlimited financial liabilities, or auto-renewal traps). This project demonstrates **Hybrid AI** by bridging two classical paradigms:
1. **Statistical Machine Learning (Inductive AI)**:
   - Supervised classification of contract clauses using TF-IDF n-grams + Logistic Regression (benchmarked against Naive Bayes and Linear SVM).
   - Cosine similarity matching against standardized benchmark clauses.
   - Unsupervised K-Means clustering ($k=3$) to reveal thematic contract groupings.
2. **Knowledge-Based System (Deductive AI)**:
   - A goal-based forward-chaining rule agent with dynamic working memory facts.
   - Multi-pass inference loop with compounding risk escalation rules (`n_flags >= 2`).

```
  [ Contract File (PDF / DOCX / TXT) ]
                   │
                   ▼
       [ Clause Splitter & Regex ]
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
 [ TF-IDF + Logistic ]   [ Semantic Facts ]
 [ Regression Model  ]   [ (Notice, Cap, etc) ]
         │                   │
         └─────────┬─────────┘
                   ▼
      [ Forward-Chaining Agent ]
     (Working Memory & Escalation)
                   │
                   ▼
  [ Risk Level | Action | Explanation ]
```

---

## 📂 Dataset: CUAD (Contract Understanding Atticus Dataset)

The Atticus Project's CUAD dataset is an open-source, SQuAD-style dataset consisting of real-world commercial agreements annotated by legal experts.

This project focuses on **6 key contractual categories**:
1. `Termination For Convenience`
2. `Renewal Term`
3. `Uncapped Liability`
4. `Cap On Liability`
5. `Non-Compete`
6. `Governing Law`

Duplicate texts are removed, and each category is capped at 400 samples (using pandas 3.x-compatible slicing) to maintain class balance.

---

## 🛠️ Algorithms Implemented

1. **NLTK Text Preprocessing**:
   - Lowercasing and punctuation removal.
   - Stop-word removal with domain-critical negations preserved (`not`, `no`, `without`, `may`, `any`).
   - WordNet lemmatization.
2. **TF-IDF Feature Vectorization**:
   - Word unigrams and bigrams (`ngram_range=(1, 2)`), `min_df=2`.
3. **Supervised ML Classification**:
   - **Logistic Regression** (Primary Classifier, ~90% accuracy).
   - **Multinomial Naive Bayes** and **Linear Support Vector Machine (LinearSVC)** as comparative baselines.
   - Stratified 80/20 train-test split, classification report, and confusion matrix.
4. **Cosine Similarity Scoring**:
   - Measures semantic alignment between each contract clause and hand-written benchmark clauses for that category.
5. **Forward-Chaining Rule Agent**:
   - Production system containing 10 IF-THEN rules represented as `(name, condition, risk, action, explanation)` tuples.
   - A forward-chaining loop evaluates rules iteratively until steady state.
   - Features dynamic chained rules: earlier risk flags increase `n_flags` in working memory, triggering compound risk escalation to `SUGGEST-REWRITE`.
6. **K-Means Clustering**:
   - Unsupervised clustering ($k=3$) on the TF-IDF feature space shown as a crosstab against true categories (demonstrating structural clustering of liability, duration, and legal jurisdiction).

---

## 💻 Local Setup & Execution

### 1. Prerequisites
Ensure Python 3.9+ is installed.

```bash
# Clone or navigate to the project directory
cd /path/to/project

# Install required dependencies
pip install -r requirements.txt
```

### 2. Download Dataset
Download the CUAD dataset zip and extract `CUADv1.json`:

```bash
# Download data.zip
curl -L -o data.zip https://github.com/TheAtticusProject/cuad/raw/main/data.zip

# Unzip CUADv1.json
unzip data.zip
```

### 3. Run Analysis

**Option A: Interactive Streamlit Web UI (Upload PDF, DOCX, PPTX, or TXT directly)**
```bash
streamlit run app.py
```

**Option B: CLI with built-in 6-clause demo contract:**
```bash
python simple_clause_analyzer.py CUADv1.json
```

**Option C: CLI with a custom contract file (PDF, DOCX, or TXT):**
```bash
python simple_clause_analyzer.py CUADv1.json sample_contract.pdf
python simple_clause_analyzer.py CUADv1.json sample_contract.docx
python simple_clause_analyzer.py CUADv1.json my_contract.txt
```

---

## 🚀 Google Colab Instructions

You can run this project in a free Google Colab notebook by creating a new notebook and executing the following two cells:

### Cell 1: Environment & Dataset Setup
```python
# 1. Install dependencies
!pip install scikit-learn nltk pandas pypdf python-docx

# 2. Download CUAD dataset and extract CUADv1.json
!wget -q -O data.zip https://github.com/TheAtticusProject/cuad/raw/main/data.zip
!unzip -q data.zip

# 3. Download simple_clause_analyzer.py (or upload your file)
# If running directly from git, or write the file to the Colab runtime
```

### Cell 2: Run the Clause Risk Analyzer
```python
# Run on built-in demo contract
!python simple_clause_analyzer.py CUADv1.json
```

---

## ⚠️ Limitations & Academic Notes

1. **Heuristic Risk Policy**:
   - The CUAD dataset provides factual category annotations but does **not** provide ground-truth risk ratings.
   - All risk assignments (`High`, `Medium`, `Low`) and actions (`ACCEPT`, `FLAG-CAUTION`, `FLAG-HIGH-RISK`, `SUGGEST-REWRITE`) represent heuristic rules crafted from the perspective of protecting the weaker signer.
2. **Commercial vs. Consumer Contracts**:
   - CUAD is curated from US SEC EDGAR filings consisting primarily of B2B corporate transactions (master service agreements, vendor contracts).
   - Residential tenancy leases, consumer service agreements, or job offer letters may feature slightly different vocabulary, although the general semantic structure remains consistent.
3. **Scanned Documents & OCR**:
   - The document parser utilizes `pypdf` for digital PDFs with embedded text layers.
   - Scanned, flattened image PDFs contain no direct text stream and require an external Optical Character Recognition (OCR) pipeline (such as Tesseract OCR) prior to clause analysis.

