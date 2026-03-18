import os, gc, re, math, warnings
from pathlib import Path
from contextlib import nullcontext
from typing import List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import sacrebleu

warnings.filterwarnings("ignore")

# ── Config — edit these as needed ─────────────────────────────
OUTPUT_DIR      = "/kaggle/working/dino_ema_output"
DINO_MODEL_PATH = os.path.join(OUTPUT_DIR, "final")
OUT_CSV         = "/kaggle/working/submission.csv"
BATCH_SIZE      = 2
NUM_BEAMS       = 8
NUM_BEAM_CANDS  = 4
NUM_SAMPLE_CANDS= 2
MAX_NEW_TOKENS  = 384
MBR_POOL_CAP    = 32
LENGTH_PENALTY  = 1.3
REP_PENALTY     = 1.2
MBR_TOP_P       = 0.92
MBR_TEMPERATURE = 0.75

# Auto-detect test.csv
TEST_CSV = ""
for root, _, files in os.walk("/kaggle/input"):
    if "test.csv" in files:
        TEST_CSV = os.path.join(root, "test.csv")
        break
print(f"Test CSV : {TEST_CSV}")
print(f"Model    : {DINO_MODEL_PATH}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
print(f"Device   : {DEVICE} | BF16: {USE_BF16}")

# ──────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────

_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})

def _ascii_to_diacritics(s: str) -> str:
    s = s.replace("sz","š").replace("SZ","Š")
    s = s.replace("s,","ṣ").replace("S,","Ṣ")
    s = s.replace("t,","ṭ").replace("T,","Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s

_ALLOWED_FRACS = [
    (1/6,"0.16666"),(1/4,"0.25"),(1/3,"0.33333"),
    (1/2,"0.5"),(2/3,"0.66666"),(3/4,"0.75"),(5/6,"0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")

def _canon_decimal(x: float) -> str:
    ip = int(math.floor(x + 1e-12))
    frac = x - ip
    best = min(_ALLOWED_FRACS, key=lambda t: abs(frac - t[0]))
    if abs(frac - best[0]) <= _FRAC_TOL:
        dec = best[1]
        if ip == 0:
            return dec
        return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"
    return f"{x:.5f}".rstrip("0").rstrip(".")

_GAP_UNIFIED_RE = re.compile(
    r"<\s*big[\s_\-]*gap\s*>|<\s*gap\s*>|\bbig[\s_\-]*gap\b"
    r"|\bx(?:\s+x)+\b|\.{3,}|…+|\[\.+\]|\[\s*x\s*\]|\(\s*x\s*\)"
    r"|(?<!\w)x{2,}(?!\w)|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)", re.I
)
_CHAR_TRANS = str.maketrans({
    "ḫ":"h","Ḫ":"H","ʾ":"",
    "₀":"0","₁":"1","₂":"2","₃":"3","₄":"4",
    "₅":"5","₆":"6","₇":"7","₈":"8","₉":"9",
    "—":"-","–":"-",
})
_UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
_UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"
_DET_UPPER_RE = re.compile(r"\(([" + _UNICODE_UPPER + r"0-9]{1,6})\)")
_DET_LOWER_RE = re.compile(r"\(([" + _UNICODE_LOWER + r"]{1,4})\)")
_KUBABBAR_RE = re.compile(r"KÙ\.B\.")
_WS_RE = re.compile(r"\s+")
_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {"0.8333":"⅚","0.6666":"⅔","0.3333":"⅓","0.1666":"⅙","0.625":"⅝","0.75":"¾","0.25":"¼","0.5":"½"}

def _frac_repl(m: re.Match) -> str:
    return _EXACT_FRAC_MAP[m.group(0)]

class OptimizedPreprocessor:
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        ser = pd.Series(texts).fillna("").astype(str)
        ser = ser.apply(_ascii_to_diacritics)
        ser = ser.str.replace(_DET_UPPER_RE, r"\1", regex=True)
        ser = ser.str.replace(_DET_LOWER_RE, r"{\1}", regex=True)
        ser = ser.str.replace(_GAP_UNIFIED_RE, "<gap>", regex=True)
        ser = ser.str.translate(_CHAR_TRANS)
        ser = ser.str.replace("ₓ", "", regex=False)
        ser = ser.str.replace(_KUBABBAR_RE, "KÙ.BABBAR", regex=True)
        ser = ser.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
        ser = ser.str.replace(_FLOAT_RE, lambda m: _canon_decimal(float(m.group(1))), regex=True)
        ser = ser.str.replace(_WS_RE, " ", regex=True).str.strip()
        return ser.tolist()


# ──────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────

_SOFT_GRAM_RE   = re.compile(r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)(?:\.\s*(?:plur|plural|sing|singular))?\.?\s*[^)]*\)", re.I)
_BARE_GRAM_RE   = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
_UNCERTAIN_RE   = re.compile(r"\(\?\)")
_CURLY_QUOT_RE  = re.compile("[\u201c\u201d\u2018\u2019]")
_MONTH_RE       = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
_ROMAN2INT      = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10,"XI":11,"XII":12}
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE= re.compile(r"([.,])\1+")
_PUNCT_SPC_RE   = re.compile(r"\s+([.,:])")
_FORBIDDEN_TRANS= str.maketrans("","","()——<>⌈⌋⌊[]+ʾ;")
_COMMODITY_RE   = re.compile(r"-(gold|tax|textiles)\b")
_COMMODITY_REPL = {"gold":"pašallum gold","tax":"šadduātum tax","textiles":"kutānum textiles"}
_SHEKEL_REPLS   = [
    (re.compile(r"5\s+11\s*/\s*12\s+shekels?", re.I), "6 shekels less 15 grains"),
    (re.compile(r"5\s*/\s*12\s+shekels?", re.I),      "⅔ shekel 15 grains"),
    (re.compile(r"7\s*/\s*12\s+shekels?", re.I),      "½ shekel 15 grains"),
    (re.compile(r"1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?", re.I), "15 grains"),
]
_SLASH_ALT_RE   = re.compile(r"(?<!\d)\s*/\s*(?!\d)\S+")
_STRAY_MARKS_RE = re.compile(r"<<[^>]*>>|<(?!gap\b)[^>]*>")
_MULTI_GAP_RE   = re.compile(r"(?:<gap>\s*){2,}")
_PN_RE          = re.compile(r"\bPN\b")

def _month_repl(m: re.Match) -> str:
    return f"Month {_ROMAN2INT.get(m.group(1).upper(), m.group(1))}"

def _commodity_repl(m: re.Match) -> str:
    return _COMMODITY_REPL[m.group(1)]

class VectorizedPostprocessor:
    def postprocess_batch(self, translations: List[str]) -> List[str]:
        s = pd.Series(translations).fillna("").astype(str)
        s = s.str.replace(_GAP_UNIFIED_RE, "<gap>", regex=True)
        s = s.str.replace(_PN_RE, "<gap>", regex=True)
        s = s.str.replace(_COMMODITY_RE, _commodity_repl, regex=True)
        for pat, repl in _SHEKEL_REPLS:
            s = s.str.replace(pat, repl, regex=True)
        s = s.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
        s = s.str.replace(_FLOAT_RE, lambda m: _canon_decimal(float(m.group(1))), regex=True)
        s = s.str.replace(_SOFT_GRAM_RE, " ", regex=True)
        s = s.str.replace(_BARE_GRAM_RE, " ", regex=True)
        s = s.str.replace(_UNCERTAIN_RE, "", regex=True)
        s = s.str.replace(_STRAY_MARKS_RE, "", regex=True)
        s = s.str.replace(_SLASH_ALT_RE, "", regex=True)
        s = s.str.replace(_CURLY_QUOT_RE, "", regex=True)
        s = s.str.replace(_MONTH_RE, _month_repl, regex=True)
        s = s.str.replace(_MULTI_GAP_RE, "<gap>", regex=True)
        s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
        s = s.str.translate(_FORBIDDEN_TRANS)
        s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)
        s = s.str.replace(_REPEAT_WORD_RE, r"\1", regex=True)
        for n in range(4, 1, -1):
            pat = r"\b((?:\w+\s+){" + str(n-1) + r"}\w+)(?:\s+\1\b)+"
            s = s.str.replace(pat, r"\1", regex=True)
        s = s.str.replace(_PUNCT_SPC_RE, r"\1", regex=True)
        s = s.str.replace(_REPEAT_PUNCT_RE, r"\1", regex=True)
        s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
        return s.tolist()


# ──────────────────────────────────────────────────────────────
# Dataset + BucketBatchSampler
# ──────────────────────────────────────────────────────────────

class AkkadianDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: OptimizedPreprocessor):
        self.sample_ids = df["id"].tolist()
        proc = preprocessor.preprocess_batch(df["transliteration"].tolist())
        self.input_texts = ["translate Akkadian to English: " + t for t in proc]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return self.sample_ids[idx], self.input_texts[idx]


class BucketBatchSampler(Sampler):
    def __init__(self, dataset: AkkadianDataset, batch_size: int, num_buckets: int):
        lengths = [len(t.split()) for _, t in dataset]
        sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
        bsize = max(1, len(sorted_idx) // max(1, num_buckets))
        self.buckets = [
            sorted_idx[i * bsize: None if i == num_buckets - 1 else (i + 1) * bsize]
            for i in range(num_buckets)
        ]
        self.batch_size = batch_size

    def __iter__(self):
        for bucket in self.buckets:
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(b) / self.batch_size) for b in self.buckets)


# ──────────────────────────────────────────────────────────────
# MBR Selector
# ──────────────────────────────────────────────────────────────

def _load_lexicon(lexicon_path: str) -> Dict[str, str]:
    if not lexicon_path or not os.path.exists(lexicon_path):
        return {}
    df = pd.read_csv(lexicon_path, encoding="utf-8")
    target_types = ["PN", "GN", "DN", "RN"]
    entity_df = df[df["type"].isin(target_types)].copy()
    lexicon = {}
    for _, row in entity_df.iterrows():
        form = str(row["form"]).strip()
        norm = str(row["norm"]).strip()
        if form == "nan" or norm == "nan":
            continue
        clean = re.sub(r"[\[\]\(\)\?\!]", "", form).lower()
        if clean:
            lexicon[clean] = norm
    print(f"Loaded {len(lexicon)} proper nouns into lexicon.")
    return lexicon


class MBRSelector:
    def __init__(self, pool_cap: int = 32, lexicon: Dict[str, str] = None):
        self._metric = sacrebleu.metrics.CHRF(word_order=2)
        self.pool_cap = pool_cap
        self.lexicon = lexicon or {}
        self.w_chrf = 0.8 if self.lexicon else 1.0
        self.w_fidelity = 0.2 if self.lexicon else 0.0

    def _chrfpp(self, a: str, b: str) -> float:
        a, b = (a or "").strip(), (b or "").strip()
        if not a or not b:
            return 0.0
        return float(self._metric.sentence_score(a, [b]).score)

    def _fidelity(self, source: str, candidate: str) -> float:
        if not self.lexicon or not source or not candidate:
            return 100.0
        tokens = re.sub(r"[^\w\-\s]", "", source.lower()).split()
        expected = [self.lexicon[t].lower() for t in tokens if t in self.lexicon]
        if not expected:
            return 100.0
        cand_lower = candidate.lower()
        return (sum(1 for e in expected if e in cand_lower) / len(expected)) * 100.0

    @staticmethod
    def _dedup(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            x = str(x).strip()
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def pick(self, source: str, candidates: List[str]) -> str:
        cands = self._dedup(candidates)[:self.pool_cap]
        n = len(cands)
        if n == 0:
            return ""
        if n == 1:
            return cands[0]

        best_i, best_s = 0, -1e9
        for i in range(n):
            consensus = sum(self._chrfpp(cands[i], cands[j]) for j in range(n) if j != i) / max(1, n - 1)
            fidelity = self._fidelity(source, cands[i])
            score = self.w_chrf * consensus + self.w_fidelity * fidelity
            if score > best_s:
                best_s, best_i = score, i
        return cands[best_i]



# ── Dataset ───────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: OptimizedPreprocessor):
        self.ids = df["id"].tolist()
        self.texts = ["translate Akkadian to English: " + t
                      for t in preprocessor.preprocess_batch(df["transliteration"].tolist())]
    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.ids[i], self.texts[i]


# ── Load model ────────────────────────────────────────────────
print(f"\nLoading model from {DINO_MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(DINO_MODEL_PATH, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(DINO_MODEL_PATH, local_files_only=True).to(DEVICE).eval()
print(f"Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

def collate(batch):
    ids   = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    enc = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
    return ids, enc

# ── Inference ─────────────────────────────────────────────────
test_df = pd.read_csv(TEST_CSV, encoding="utf-8")
preprocessor = OptimizedPreprocessor()
postprocessor = VectorizedPostprocessor()
mbr_selector = MBRSelector(pool_cap=MBR_POOL_CAP)
dataset = TestDataset(test_df, preprocessor)
dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=collate, num_workers=2, pin_memory=(DEVICE.type=="cuda"))

ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if USE_BF16 else nullcontext()
pools_by_id: Dict[str, List[str]] = {}

with torch.inference_mode():
    for batch_ids, enc in tqdm(dl, desc="Generating"):
        input_ids = enc.input_ids.to(DEVICE)
        attn = enc.attention_mask.to(DEVICE)
        try:
            with ctx:
                beam_out = model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    do_sample=False, num_beams=NUM_BEAMS,
                    num_return_sequences=NUM_BEAM_CANDS,
                    max_new_tokens=MAX_NEW_TOKENS,
                    length_penalty=LENGTH_PENALTY,
                    repetition_penalty=REP_PENALTY,
                    early_stopping=True, use_cache=True,
                )
                beam_texts = tokenizer.batch_decode(beam_out, skip_special_tokens=True)
                samp_out = model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    do_sample=True, num_beams=1,
                    top_p=MBR_TOP_P, temperature=MBR_TEMPERATURE,
                    num_return_sequences=NUM_SAMPLE_CANDS,
                    max_new_tokens=MAX_NEW_TOKENS,
                    repetition_penalty=REP_PENALTY,
                    use_cache=True,
                )
                samp_texts = tokenizer.batch_decode(samp_out, skip_special_tokens=True)
            for i, sid in enumerate(batch_ids):
                pool = list(beam_texts[i*NUM_BEAM_CANDS:(i+1)*NUM_BEAM_CANDS])
                pool += list(samp_texts[i*NUM_SAMPLE_CANDS:(i+1)*NUM_SAMPLE_CANDS])
                pools_by_id[str(sid)] = pool
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM — skipping batch")
                torch.cuda.empty_cache()
                for sid in batch_ids:
                    pools_by_id.setdefault(str(sid), [])
            else:
                raise

# ── MBR + postprocess + save ──────────────────────────────────
results = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="MBR"):
    sid = str(row["id"])
    pool = pools_by_id.get(sid, [])
    pp = postprocessor.postprocess_batch(pool) if pool else []
    chosen = mbr_selector.pick(str(row["transliteration"]), pp)
    if not chosen.strip():
        chosen = "The tablet is too damaged to translate."
    results.append((sid, chosen))

result_df = pd.DataFrame(results, columns=["id", "translation"])
result_df.to_csv(OUT_CSV, index=False)
print(f"\nDone. {len(result_df)} rows saved to {OUT_CSV}")
print(result_df.head())
