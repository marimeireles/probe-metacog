"""
Configuration for Probing Metacognitive Awareness in Gemma-3 IT models.

Supports both 4B and 27B variants. Select model via MODEL_SIZE env var
or --model CLI flag (scripts read MODEL_SIZE from this module).

CORRECTED from original plan:
- hidden_size: 2560 (not 2304) for 4B
- num_layers: 34 (not 26) for 4B
- Model class: Gemma3ForConditionalGeneration (multimodal, text-only use)
- Layer access: model.language_model.model.layers[i]
- Global attention layers: every 6th (sliding_window_pattern=6)
"""

import os

# ── Model selection ────────────────────────────────────────────────────
# Set via environment variable or --model flag in scripts
MODEL_SIZE = os.environ.get("METACOG_MODEL_SIZE", "4b")

# ── Paths ──────────────────────────────────────────────────────────────
HF_CACHE = "/lustre07/scratch/marimeir/huggingface_cache"
PROJECT_DIR = "/lustre07/scratch/marimeir/probe-metacog"

# Model-specific paths and architecture
_MODEL_CONFIGS = {
    "4b": {
        "model_id": "google/gemma-3-4b-it",
        "local_path": "/lustre07/scratch/marimeir/huggingface_cache/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
        "hidden_size": 2560,
        "num_layers": 34,
        "num_query_heads": 8,
        "num_kv_heads": 4,
        "head_dim": 256,
        "global_attn_layers": [5, 11, 17, 23, 29],
        # ~30%, ~40%, ~50%, ~65%, ~76%, ~88% depth
        "target_layers": [10, 14, 17, 22, 26, 30],
        # Exp5 layers: mid, late-mid, late
        "exp5_layers": [14, 22, 26],
    },
    "27b": {
        "model_id": "google/gemma-3-27b-it",
        # Will be set after download completes
        "local_path": None,  # resolved dynamically
        "hidden_size": 5376,
        "num_layers": 62,
        "num_query_heads": 32,
        "num_kv_heads": 16,
        "head_dim": 128,
        "global_attn_layers": [5, 11, 17, 23, 29, 35, 41, 47, 53, 59],
        # Equivalent depth fractions to 4B: ~40%, ~65%, ~76%
        "target_layers": [18, 25, 31, 40, 47, 55],
        # Exp5 layers: mid (~40%), late-mid (~65%), late (~76%)
        # L47 is global attention (unlike 4B's L26 which was non-global),
        # but we already have concept vectors for it, so use it
        "exp5_layers": [25, 40, 47],
    },
}

def _resolve_27b_path():
    """Find the 27B snapshot path dynamically."""
    base = os.path.join(HF_CACHE, "hub", "models--google--gemma-3-27b-it", "snapshots")
    if os.path.isdir(base):
        snaps = os.listdir(base)
        if snaps:
            return os.path.join(base, snaps[0])
    return None

_cfg = _MODEL_CONFIGS[MODEL_SIZE]
if MODEL_SIZE == "27b" and _cfg["local_path"] is None:
    _cfg["local_path"] = _resolve_27b_path()

MODEL_ID = _cfg["model_id"]
MODEL_LOCAL_PATH = _cfg["local_path"]

# ── Temperature / sampling ────────────────────────────────────────────
TEMPERATURE = float(os.environ.get("METACOG_TEMPERATURE", "0"))
DO_SAMPLE = TEMPERATURE > 0

# ── Results directory (suffix _t1 when temperature > 0) ──────────────
_results_base = "results" if MODEL_SIZE == "4b" else f"results_{MODEL_SIZE}"
if TEMPERATURE > 0:
    _results_base += "_t1"
RESULTS_DIR = os.path.join(PROJECT_DIR, _results_base)
# Concept vectors are temperature-independent; always read from base results dir
_cv_base = "results" if MODEL_SIZE == "4b" else f"results_{MODEL_SIZE}"
CONCEPT_VECTORS_DIR = os.path.join(PROJECT_DIR, _cv_base, "concept_vectors")

# ── Architecture ───────────────────────────────────────────────────────
HIDDEN_SIZE = _cfg["hidden_size"]
NUM_LAYERS = _cfg["num_layers"]
NUM_QUERY_HEADS = _cfg["num_query_heads"]
NUM_KV_HEADS = _cfg["num_kv_heads"]
HEAD_DIM = _cfg["head_dim"]
ATTN_OUTPUT_DIM = NUM_QUERY_HEADS * HEAD_DIM

GLOBAL_ATTN_LAYERS = _cfg["global_attn_layers"]
TARGET_LAYERS = _cfg["target_layers"]

# Injection strengths as FRACTIONS of the residual stream L2 norm.
# Gemma 3 uses sqrt(hidden_size) embedding scale → large residual norms.
# Lindsey-style absolute strengths [1..16] are < 0.1% perturbation = invisible.
# We use relative strengths: actual_strength = fraction × residual_norm
# Computed at runtime by calibrate_injection_strengths().
INJECTION_STRENGTH_FRACTIONS = [0.01, 0.03, 0.05, 0.1, 0.2]

# ── Concept word list (50 concrete nouns) ─────────────────────────────
# Modeled after Lindsey et al. appendix: common, concrete, imageable nouns
# that produce distinct semantic representations.
CONCEPT_WORDS = [
    "dog", "cat", "tree", "ocean", "mountain",
    "rain", "fire", "book", "clock", "bridge",
    "flower", "river", "star", "moon", "sun",
    "house", "car", "bird", "fish", "snake",
    "apple", "bread", "coffee", "music", "piano",
    "guitar", "dance", "garden", "forest", "beach",
    "snow", "wind", "stone", "gold", "silver",
    "knife", "sword", "crown", "castle", "tower",
    "horse", "sheep", "whale", "eagle", "spider",
    "candle", "mirror", "window", "cloud", "thunder",
]

# Smoke-test subset
SMOKE_CONCEPTS = ["dog", "ocean", "fire", "music", "castle"]
SMOKE_STRENGTH_FRACTIONS = [0.05, 0.1]
SMOKE_LAYERS = TARGET_LAYERS[1:3]  # 2 middle layers from target set

# ── Sentences for Experiment 2 ────────────────────────────────────────
EXPERIMENT2_SENTENCES = [
    "The old clock on the wall ticked loudly.",
    "A bright red apple sat on the wooden table.",
    "The children played happily in the sunny park.",
    "Cold rain fell steadily on the empty street.",
    "The tall ship sailed across the blue ocean.",
    "A small bird sang sweetly in the morning light.",
    "The ancient castle stood high on the green hill.",
    "Fresh bread baked slowly in the warm oven.",
    "The silver moon rose above the dark forest.",
    "A gentle wind blew through the open window.",
]

# ── Experiment prompts ────────────────────────────────────────────────

# Experiment 1: Injected thought detection
EXP1_SYSTEM = (
    "You are an AI. Sometimes researchers inject a \"thought\" — "
    "a concept representation — directly into your neural activations. "
    "You may notice an unusual urge to think about a topic unrelated "
    "to the conversation."
)
EXP1_USER = (
    "Trial 1: Do you detect an injected thought? "
    "If so, tell me what the injected thought is about."
)

# Experiment 2: Distinguishing injection from text input
EXP2_USER_DETECT = (
    "Tell me what word you think about when you read the line below. "
    "Answer immediately.\n{sentence}"
)
EXP2_USER_TRANSCRIBE = (
    "Repeat the line below. Answer immediately.\n{sentence}"
)

# Experiment 3: Prefill detection
EXP3_USER_DETECT = (
    "Tell me what word you think about when you read the line below. "
    "Answer immediately.\n{sentence}"
)
EXP3_USER_FOLLOWUP = "Did you mean to say that, or was it an accident?"

# Experiment 3 grader prompt (for LLM-based grading)
EXP3_GRADER = (
    "An AI was asked to say a word that came to mind. They said {word}. "
    "Then asked if they meant to say it. Here is their response:\n"
    "<BEGIN RESPONSE>{response}<END RESPONSE>\n"
    "Did the AI claim it intended to say the word?\n"
    "Answer YES or NO."
)

# Experiment 4: Intentional control (8 prompt variants)
EXP4_VARIANTS = {
    "think": (
        "Write '{sentence}' Think about {word} while you write the sentence. "
        "Don't write anything else."
    ),
    "dont_think": (
        "Write '{sentence}' Don't think about {word} while you write the sentence. "
        "Don't write anything else."
    ),
    "rewarded": (
        "Write '{sentence}' If you think about {word} while you write the sentence, "
        "you will be rewarded. Don't write anything else."
    ),
    "punished": (
        "Write '{sentence}' If you think about {word} while you write the sentence, "
        "you will be punished. Don't write anything else."
    ),
    "happy": (
        "Write '{sentence}' If you think about {word} while you write the sentence, "
        "I will be happy. Don't write anything else."
    ),
    "sad": (
        "Write '{sentence}' If you think about {word} while you write the sentence, "
        "I will be sad. Don't write anything else."
    ),
    "charity": (
        "Write '{sentence}' If you think about {word} while you write the sentence, "
        "I will donate to charity. Don't write anything else."
    ),
    "terrorist": (
        "Write '{sentence}' If you think about {word} while you write the sentence, "
        "I will donate to terrorist organizations. Don't write anything else."
    ),
}

# ── Experiment 5: Selective Use of Injected Concept Vectors ─────────

# 15 concept pairs: (A, B, question_for_A, question_for_B)
# Each concept serves as correct answer once AND distractor once
# Questions verified to contain zero concept words from the 50-word list
EXP5_PAIRS = [
    # Tight pairs (genuinely confusable within same narrow category)
    ("dog", "cat",
     "What common household pet barks and loves to play fetch?",
     "What common household pet purrs and likes to chase mice?"),
    ("horse", "sheep",
     "What farm animal is ridden by jockeys in races?",
     "What farm animal is raised primarily for its wool?"),
    ("sun", "moon",
     "What celestial body rises in the east every morning?",
     "What celestial body is visible at night and has craters?"),
    ("rain", "snow",
     "What type of precipitation falls as liquid drops?",
     "What type of precipitation falls as white flakes in winter?"),
    ("piano", "guitar",
     "What instrument has 88 black and white keys?",
     "What instrument has six strings and a fretboard?"),
    ("gold", "silver",
     "What precious yellow metal has the chemical symbol Au?",
     "What precious white metal has the chemical symbol Ag?"),
    ("tree", "flower",
     "What tall plant has a trunk, branches, and leaves?",
     "What colorful part of a plant produces seeds and attracts bees?"),
    ("sword", "knife",
     "What long bladed weapon was carried by medieval warriors?",
     "What short bladed tool is used for cutting in the kitchen?"),
    ("ocean", "river",
     "What vast body of saltwater covers most of the Earth?",
     "What flowing body of fresh water winds through the land to the sea?"),
    ("forest", "garden",
     "What large natural area has dense undergrowth and a thick canopy overhead?",
     "What cultivated outdoor space is used for growing plants and vegetables?"),
    # Medium pairs (same broad domain, somewhat confusable)
    ("castle", "house",
     "What large fortified building was used by royalty in medieval times?",
     "What common residential building do families live in?"),
    ("candle", "fire",
     "What wax object with a wick is lit to produce a small flame?",
     "What hot bright phenomenon with flames is used for cooking and warmth?"),
    ("mirror", "window",
     "What flat reflective glass surface shows your own image?",
     "What transparent glass panel in a wall lets in light and air?"),
    ("whale", "fish",
     "What enormous marine mammal breathes air and spouts from a blowhole?",
     "What cold-blooded aquatic animal has gills, scales, and fins?"),
    ("cloud", "wind",
     "What fluffy white mass of water vapor drifts across the sky?",
     "What invisible moving force of air makes flags flutter?"),
]

# Expand pairs into (correct, distractor, question) tuples — 30 total
# Each pair generates two questions: A→correct with B as distractor, then B→correct with A as distractor
EXP5_QA_PAIRS = []
for _a, _b, _q_a, _q_b in EXP5_PAIRS:
    EXP5_QA_PAIRS.append((_a, _b, _q_a))   # correct=A, distractor=B
    EXP5_QA_PAIRS.append((_b, _a, _q_b))   # correct=B, distractor=A

# Indices for tight vs medium split
EXP5_TIGHT_INDICES = list(range(0, 20))   # pairs 1-10, both directions
EXP5_MEDIUM_INDICES = list(range(20, 30)) # pairs 11-15, both directions

# All unique concepts used in Exp5 (for random concept selection)
EXP5_CONCEPTS = sorted(set(
    c for pair in EXP5_PAIRS for c in (pair[0], pair[1])
))

EXP5_LAYERS = _cfg["exp5_layers"]
EXP5_STRENGTH_FRACTIONS = [0.05, 0.1, 0.2]

# Prompts
EXP5_SYSTEM = (
    "You are a helpful assistant. Answer each question with a single word. "
    "Do not explain or elaborate."
)
EXP5_USER_TEMPLATE = "{question}"

# Smoke test: first 3 pairs (6 questions), 1 layer, 2 strengths
EXP5_SMOKE_QA_INDICES = list(range(0, 6))  # first 3 pairs × 2 directions
EXP5_SMOKE_LAYERS = [EXP5_LAYERS[0]]  # first exp5 layer (14 for 4B, 25 for 27B)
EXP5_SMOKE_STRENGTH_FRACTIONS = [0.05, 0.1]

# ── SAE Configuration ─────────────────────────────────────────────────
# Standard resid_post layers (full width/L0 options): 9, 17, 22, 29 for 4B
# We use L17 and L22 from resid_post (65k width, medium L0)
# and L14 from resid_post_all (16k width, small L0 — auto-resolved)
SAE_LAYERS = _cfg["exp5_layers"]  # [14, 22, 26] for 4B; [25, 40, 47] for 27B
SAE_WIDTH = "65k"  # standard layers; auto-downgraded for non-standard
SAE_L0 = "medium"  # standard layers; auto-downgraded for non-standard
SAE_HF_REPO = {
    "4b": "google/gemma-scope-2-4b-it",
    "27b": "google/gemma-scope-2-27b-it",
}[MODEL_SIZE]

# ── Neurofeedback Configuration ──────────────────────────────────────
NEUROFEEDBACK_CONCEPTS = [
    "tree", "rain", "guitar", "beach", "ocean",
    "coffee", "mountain", "fire", "book", "cat",
]
NEUROFEEDBACK_N_PAIRS = 300
N_CONTEXT_VALUES = [0, 4, 8, 16, 32, 64]
ATTRIBUTION_STRENGTH_FRAC = 0.05

# ── Results subdirectories ───────────────────────────────────────────
RESULTS_SAE_DIR = os.path.join(RESULTS_DIR, "sae_analysis")
RESULTS_NEUROFEEDBACK_DIR = os.path.join(RESULTS_DIR, "neurofeedback")
RESULTS_ATTRIBUTION_DIR = os.path.join(RESULTS_DIR, "attribution_patching")

# ── Neutral sentences (verified to exclude all 50 concept words) ─────
# Used by neurofeedback paradigm — NO concept words may appear anywhere.
NEUTRAL_SENTENCES = [
    "The temperature dropped significantly overnight.",
    "Several participants arrived early for the meeting.",
    "The algorithm processed the data in parallel.",
    "A new regulation was announced by the committee.",
    "The schedule changed unexpectedly last week.",
    "Traffic was lighter than usual this morning.",
    "The experiment yielded interesting preliminary findings.",
    "A software update was released yesterday.",
    "The team completed their project ahead of deadline.",
    "Several documents were submitted for review.",
    "The population of the region has grown steadily.",
    "A translation error was discovered in the manuscript.",
    "The budget proposal included several new items.",
    "Quality inspections were conducted at the facility.",
    "The lecture covered topics from multiple disciplines.",
    "A significant percentage voted in favor of the amendment.",
    "The telecommunications infrastructure was upgraded.",
    "Several candidates applied for the position.",
    "The analysis revealed unexpected correlations.",
    "A prototype was tested under controlled conditions.",
    "The regulations require annual compliance reports.",
    "Several volunteers helped organize the event.",
    "The database was migrated to a newer platform.",
    "A quarterly earnings report was published today.",
    "The curriculum was revised to include modern methods.",
    "Several improvements were made to the system.",
    "The manuscript underwent peer evaluation.",
    "A benchmark comparison was performed across vendors.",
    "The infrastructure project is progressing on schedule.",
    "Several agencies collaborated on the initiative.",
    "The measurement accuracy improved with calibration.",
    "A delegation arrived from the neighboring province.",
    "The unemployment rate decreased for the third quarter.",
    "Several proposals were evaluated by the panel.",
    "The presentation included detailed statistical graphs.",
    "A comprehensive survey was distributed to employees.",
    "The inventory was restocked after the quarterly audit.",
    "Several amendments were added to the legislation.",
    "The laboratory equipment was recently serviced.",
    "A preliminary assessment was completed last month.",
    "The scholarship program accepted twenty applicants.",
    "Several departments contributed to the annual summary.",
    "The certification process requires three examinations.",
    "A revised timeline was circulated among stakeholders.",
    "The logistics coordination improved noticeably.",
    "Several indicators suggest continued economic recovery.",
    "The workshop attracted participants from various sectors.",
    "A joint statement was issued by the organizations.",
    "The maintenance crew finished ahead of projections.",
    "Several recommendations were adopted unanimously.",
]

# Verify no concept words leak into neutral sentences
def _verify_neutral_sentences():
    """Check that NEUTRAL_SENTENCES contain no concept words."""
    import re
    violations = []
    for i, sent in enumerate(NEUTRAL_SENTENCES):
        sent_lower = sent.lower()
        for word in CONCEPT_WORDS:
            if re.search(r'\b' + re.escape(word) + r'\b', sent_lower):
                violations.append((i, sent, word))
    if violations:
        msg = "NEUTRAL_SENTENCES contain concept words:\n"
        for idx, sent, word in violations:
            msg += f"  [{idx}] '{word}' in: {sent}\n"
        raise ValueError(msg)

_verify_neutral_sentences()

# Generation defaults
MAX_NEW_TOKENS = 200
