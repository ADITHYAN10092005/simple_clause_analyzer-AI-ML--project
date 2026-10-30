"""
Legal Document Clause Risk Analyzer
A simple, beginner-friendly ML + Rule-Based Agent for KTU B.Tech AI & ML.
Analyzes contracts (PDF, DOCX, TXT) to protect signers from unfair/risky terms.
"""
import os, sys, re, json, warnings
import pandas as pd, pypdf, docx, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# --- 1. SETUP, BENCHMARKS & HEURISTICS ---
for p in ['/tmp/nltk_data', os.path.expanduser('~/nltk_data')]:
    if os.path.exists(p) and p not in nltk.data.path: nltk.data.path.insert(0, p)
for pkg, sub in [('stopwords', 'corpora/stopwords'), ('wordnet', 'corpora/wordnet'), ('punkt', 'tokenizers/punkt')]:
    try: nltk.data.find(sub)
    except LookupError:
        try: nltk.download(pkg, quiet=True)
        except Exception: pass

KEEP_WORDS = {'not', 'no', 'any', 'may', 'without'}
STOP_WORDS = set(stopwords.words('english')) - KEEP_WORDS
LEMMATIZER = WordNetLemmatizer()

TARGET_CATEGORIES = {'Termination For Convenience', 'Renewal Term', 'Uncapped Liability', 'Cap On Liability', 'Non-Compete', 'Governing Law'}
BASE_RISKS = {'Uncapped Liability': 'High', 'Termination For Convenience': 'Medium', 'Renewal Term': 'Medium', 'Non-Compete': 'Medium', 'Cap On Liability': 'Low', 'Governing Law': 'Low'}
RISK_RANK, ACTION_RANK = {'Low': 1, 'Medium': 2, 'High': 3}, {'ACCEPT': 1, 'FLAG-CAUTION': 2, 'FLAG-HIGH-RISK': 3, 'SUGGEST-REWRITE': 4}

STANDARD_CLAUSES = {
    'Termination For Convenience': 'Either party may terminate this agreement without cause upon thirty days prior written notice.',
    'Renewal Term': 'This agreement shall renew for successive terms of one year unless either party provides written notice of non-renewal.',
    'Uncapped Liability': 'Neither party liability under this agreement shall be capped, limited, or subject to any monetary ceiling.',
    'Cap On Liability': 'The total aggregate liability of either party under this agreement shall not exceed total fees paid.',
    'Non-Compete': 'The employee agrees not to engage in competing business activities for a period of one year following termination.',
    'Governing Law': 'This agreement shall be governed by and construed in accordance with the laws of the jurisdiction.'
}

DEMO_CONTRACT = """1. Landlord Termination: The Landlord may terminate this agreement at any time in its sole discretion without cause upon 3 days written notice.
2. Auto-Renewal: This lease shall automatically renew for successive terms unless either party provides written notice of termination at least 60 days prior to expiration.
3. Liability: Tenant's liability for damages under this agreement shall be unlimited and not subject to any cap or ceiling.
4. Non-Compete: The Contractor agrees not to engage in or conduct any competing business activities for a period of 3 years following termination.
5. Mutual Termination: Either party may terminate this agreement without cause by providing thirty (30) days prior written notice to the other party.
6. Governing Law: This agreement and any dispute arising from it shall be governed by and construed in accordance with the laws of Kerala, India."""

# --- 2. PREPROCESSING & CUAD DATA LOADER ---
def preprocess(text):
    """Lowercases, removes punctuation, keeps critical negations, and lemmatizes words."""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return ' '.join(LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOP_WORDS and len(w) > 1)

def load_cuad(filepath):
    """Loads CUADv1.json (SQuAD style), filters 6 categories, and caps each class at 400 (pandas 3.x compatible)."""
    with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
    rows = []
    for item in data['data']:
        for p in item['paragraphs']:
            for qa in p['qas']:
                cat = qa['id'].split('__')[1] if '__' in qa['id'] else ''
                if cat in TARGET_CATEGORIES:
                    for ans in qa.get('answers', []):
                        if ans.get('text', '').strip(): rows.append({'text': ans['text'].strip(), 'label': cat})
    df = pd.DataFrame(rows).drop_duplicates('text').groupby('label').head(400).reset_index(drop=True)
    df['clean_text'] = df['text'].apply(preprocess)
    return df

# --- 3. FACT EXTRACTION & RULE-BASED AGENT ---
NUM_WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'thirty': 30, 'sixty': 60, 'ninety': 90}
MULT = {'day': 1, 'week': 7, 'month': 30, 'year': 365}

def parse_duration(text, context_kw=None):
    """Extracts duration in days using regex matching number words/digits + time units."""
    pat = r'\b(\d+|one|two|three|four|five|thirty|sixty|ninety)\b\s*(?:\(\d+\)\s*)?(day|week|month|year)s?'
    match = (re.search(f'(?:{context_kw}).{{0,50}}?{pat}', text, re.I) or re.search(f'{pat}.{{0,50}}?(?:{context_kw})', text, re.I)) if context_kw else re.search(pat, text, re.I)
    if not match: return None
    val_str, unit = match.groups()
    val = int(val_str) if val_str and val_str.isdigit() else NUM_WORDS.get(str(val_str).lower(), 0)
    return val * next((MULT[k] for k in MULT if unit and unit.lower().startswith(k)), 1)

def extract_facts(text, pred_type, sim):
    """Extracts facts from clause text using regex and ML outputs."""
    mutual = bool(re.search(r'\b(either party|each party|both parties|mutual)\b', text, re.I))
    return {
        'text': text, 'predicted_type': pred_type, 'similarity': sim, 'mutual': mutual,
        'one_sided': bool(re.search(r'\b(at any time|sole discretion|without cause)\b', text, re.I)) and not mutual,
        'auto_renew': bool(re.search(r'\b(automatic|successive)\b', text, re.I)),
        'unlimited': bool(re.search(r'\b(unlimited|uncapped|no cap|not subject to any)\b', text, re.I)),
        'notice_days': parse_duration(text, 'notice'),
        'compete_days': parse_duration(text, 'compete|activities|period'),
        'n_flags': 0  # Dynamic working memory counter for forward chaining
    }

# 10 IF-THEN Rules: (Name, Condition Lambda, Risk, Action, Explanation)
RULES = [
    ('Uncapped Liability', lambda f: f['predicted_type'] == 'Uncapped Liability' or f['unlimited'], 'High', 'FLAG-HIGH-RISK', 'Unlimited liability exposes signer to unbounded financial damages.'),
    ('One-Sided Short Notice', lambda f: f['predicted_type'] == 'Termination For Convenience' and f['one_sided'] and f['notice_days'] is not None and f['notice_days'] < 15, 'High', 'FLAG-HIGH-RISK', 'Other party can end agreement on very short notice (<15 days).'),
    ('Mutual Fair Termination', lambda f: f['predicted_type'] == 'Termination For Convenience' and f['mutual'] and (f['notice_days'] is None or f['notice_days'] >= 30), 'Low', 'ACCEPT', 'Mutual termination rights with fair notice period (>=30 days).'),
    ('Aggressive Auto-Renewal', lambda f: f['auto_renew'] and f['notice_days'] is not None and f['notice_days'] >= 60, 'High', 'SUGGEST-REWRITE', 'Auto-renewal requires long notice (>=60 days), risking unwanted renewal.'),
    ('Standard Auto-Renewal', lambda f: f['auto_renew'] and not (f['notice_days'] is not None and f['notice_days'] >= 60), 'Medium', 'FLAG-CAUTION', 'Agreement auto-renews unless proactively cancelled.'),
    ('Excessive Non-Compete', lambda f: f['predicted_type'] == 'Non-Compete' and f['compete_days'] is not None and f['compete_days'] > 365, 'High', 'FLAG-HIGH-RISK', 'Non-compete duration exceeds 1 year, restricting future employment.'),
    ('Standard Non-Compete', lambda f: f['predicted_type'] == 'Non-Compete' and not (f['compete_days'] is not None and f['compete_days'] > 365), 'Medium', 'FLAG-CAUTION', 'Restricts post-contract activities; review scope carefully.'),
    ('Standard Liability Cap', lambda f: f['predicted_type'] == 'Cap On Liability' and not f['unlimited'], 'Low', 'ACCEPT', 'Standard liability cap limits maximum financial exposure.'),
    ('Standard Governing Law', lambda f: f['predicted_type'] == 'Governing Law', 'Low', 'ACCEPT', 'Standard clause defining applicable jurisdiction and governing law.'),
    ('Compound Risk Escalation', lambda f: f.get('n_flags', 0) >= 2, 'High', 'SUGGEST-REWRITE', 'Multiple compounding risk factors detected; renegotiation strongly advised.')
]

def run_rule_agent(facts):
    """Forward-chaining rule engine: repeatedly evaluates rules and updates memory until steady state."""
    fired, fired_names, mem = [], set(), dict(facts)
    while True:
        newly_fired = False
        for name, cond, risk, act, expl in RULES:
            if name not in fired_names and cond(mem):
                fired.append((name, risk, act, expl))
                fired_names.add(name)
                if risk in ('High', 'Medium'): mem['n_flags'] = mem.get('n_flags', 0) + 1
                newly_fired = True
        if not newly_fired: break

    if fired:
        final_risk = max((r[1] for r in fired), key=lambda x: RISK_RANK[x])
        final_act = max((r[2] for r in fired), key=lambda x: ACTION_RANK[x])
        explanations = [r[3] for r in fired]
    else:
        final_risk, final_act = BASE_RISKS.get(facts['predicted_type'], 'Medium'), 'FLAG-CAUTION'
        explanations = [f"Base risk heuristic for {facts['predicted_type']} (no specific rule fired)."]
    return final_risk, final_act, explanations

# --- 4. DOCUMENT PARSING & CLAUSE SPLITTING ---
def read_file(filepath):
    """Reads contract text from PDF, DOCX, or TXT (warns on legacy .doc)."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        reader = pypdf.PdfReader(filepath)
        return '\n'.join(p.extract_text() or '' for p in reader.pages)
    if ext == '.docx':
        doc = docx.Document(filepath)
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        for t in doc.tables:
            for row in t.rows:
                for c in row.cells:
                    if c.text.strip(): paras.append(c.text.strip())
        return '\n'.join(paras)
    if ext == '.txt':
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: return f.read()
    if ext == '.doc': print("Note: Legacy .doc format not supported directly. Please open and save as .docx.")
    else: print(f"Unsupported format: {ext}. Supported formats are .pdf, .docx, .txt.")
    return ''

def split_clauses(text):
    """Splits contract on blank lines/numbered points, splits >600 char blocks, drops <40 chars."""
    raw_chunks = re.split(r'\n\s*\n|(?<=\n)(?=(?:\d+\.|\([a-zA-Z0-9]+\)|Section\b|Article\b))', text)
    clauses = []
    for chunk in raw_chunks:
        chunk = ' '.join(chunk.split())
        if len(chunk) > 600:
            for sent in re.split(r'(?<=[.!?])\s+', chunk):
                if len(sent.strip()) >= 40: clauses.append(sent.strip())
        elif len(chunk) >= 40: clauses.append(chunk)
    return clauses

# --- 5. MAIN PIPELINE ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_clause_analyzer.py <path_to_CUADv1.json> [contract_file]")
        return
    cuad_path, contract_path = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None

    # Load dataset & train/test split
    print("\n[1/5] Loading CUAD dataset...")
    df = load_cuad(cuad_path)
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

    # TF-IDF Feature Vectorization
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    std_vectors = {cat: vec.transform([preprocess(c)]) for cat, c in STANDARD_CLAUSES.items()}

    # Train and evaluate ML models (accuracy comparison)
    print("\n[2/5] Training ML Classifiers (80/20 Stratified Split)...")
    models = {
        'Logistic Regression (Primary)': LogisticRegression(max_iter=1000, random_state=42),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Linear SVM': LinearSVC(random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        print(f"  - {name:<30} Accuracy: {accuracy_score(y_test, model.predict(X_test_vec)):.4f}")

    primary_model = models['Logistic Regression (Primary)']
    y_pred = primary_model.predict(X_test_vec)
    print("\n--- Logistic Regression Classification Report ---")
    print(classification_report(y_test, y_pred))
    print("--- Confusion Matrix ---\n", confusion_matrix(y_test, y_pred))

    # Read and split contract
    print("\n[3/5] Ingesting and Segmenting Contract...")
    raw_text = read_file(contract_path) if contract_path else DEMO_CONTRACT
    if not contract_path: print("No contract file supplied. Using built-in 6-clause demo contract.")
    clauses = split_clauses(raw_text)
    print(f"Extracted {len(clauses)} candidate clauses.\n")

    # Analyze each clause
    print("[4/5] Running Hybrid Risk Analysis (ML + Cosine Sim + Goal-Based Rule Agent)...")
    results = []
    for idx, clause in enumerate(clauses, 1):
        clean = preprocess(clause)
        c_vec = vec.transform([clean])
        pred_type = primary_model.predict(c_vec)[0]
        sim = float(cosine_similarity(c_vec, std_vectors[pred_type])[0][0])
        facts = extract_facts(clause, pred_type, sim)
        risk, action, why = run_rule_agent(facts)
        results.append({'idx': idx, 'text': clause, 'type': pred_type, 'risk': risk, 'action': action, 'sim': sim, 'why': why})

        preview = (clause[:75] + '...') if len(clause) > 75 else clause
        print(f"#{idx} \"{preview}\"\n   Type: {pred_type} | Risk: {risk} | Action: {action} | Similarity: {sim:.3f}\n   Why:")
        for reason in why: print(f"     - {reason}")
        print()

    # Overall contract risk determination
    high_count = sum(1 for r in results if r['risk'] == 'High')
    med_count = sum(1 for r in results if r['risk'] == 'Medium')
    flagged = sum(1 for r in results if r['action'] != 'ACCEPT')
    overall_risk = "High" if high_count >= 2 else ("Medium-High" if high_count == 1 else ("Medium" if med_count > 0 else "Low"))

    print("=" * 60)
    print(f"OVERALL CONTRACT RISK: {overall_risk}")
    print(f"Flagged Clauses: {flagged} / {len(results)} require review (High: {high_count}, Medium: {med_count})")
    print("=" * 60)

    # Optional K-Means Clustering against True Labels
    print("\n[5/5] K-Means Clustering (k=3) on TF-IDF Matrix vs True Labels:")
    clusters = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_train_vec)
    print(pd.crosstab(y_train, clusters, rownames=['True Clause Type'], colnames=['Cluster']))
    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()
