from pathlib import Path


GREEDY_CONFIG = (
    Path(__file__).resolve().parent / "config" / "greedy.yaml"
)

CORPORA_CONFIG = (
    Path(__file__).resolve().parent / "config" / "corpora.yaml"
)

ROCSMT = "rocsmt"
FOOTWEETS = "footweets"
MMTC = "mmtc"
PFSMB = "pfsmb"
PMUMT = "pmumt"

CORPORA = [ROCSMT, FOOTWEETS, MMTC, PFSMB, PMUMT]
CORPUS_LABELS = {
    ROCSMT: "RoCS-MT",
    FOOTWEETS: "FooTweets",
    MMTC: "MMTC",
    PFSMB: "PFSMB",
    PMUMT: "PMUMT",
}

NLLB = "facebook/nllb-200-3.3B"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
GEMMA = "google/gemma-2-9b-it"
TOWER = "Unbabel/TowerInstruct-7B-v0.2"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
GRANITE = "ibm-granite/granite-4.1-8b"
GPT = "gpt-4o-mini"
    
MODEL_LABELS = {
    NLLB: "NLLB-3B",
    LLAMA: "LLaMA-3.1-8B",
    GEMMA: "Gemma-2-9B",
    TOWER: "Tower-0.2-7B",
    QWEN: "Qwen-2.5-7B",
    MISTRAL: "Mistral-0.3-7B",
    GRANITE: "Granite-4.1-8B",
}

BLEU = "bleu"
CHRF = "chrf2"
COMET = "comet"
COMETKIWI = "cometkiwi"
XCOMET = "xcomet"

METRIC_LABELS = {
    BLEU: "BLEU",
    CHRF: "ChrF++",
    COMET: "COMET",
    COMETKIWI: "COMET-Kiwi",
    XCOMET: "xCOMET-XL",
}

COMET_MODELS = {
    COMET: "Unbabel/wmt22-comet-da",
    COMETKIWI: "Unbabel/wmt22-cometkiwi-da",
    XCOMET: "Unbabel/XCOMET-XL",
}

MINOR = "minor"
MAJOR = "major"
CRITICAL = "critical"

DEFAULT = "default"
STANDARD = "standard"
GENERAL = "general"

GUIDELINE_LABELS = {
    DEFAULT: "None",
    STANDARD: "Standard",
    GENERAL: "+General",
    ROCSMT: "+RoCS-MT",
    FOOTWEETS: "+FooTweets",
    MMTC: "+MMTC",
    PFSMB: "+PFSMB",
}

VS_NLLB = "vs_nllb"
VS_DEFAULT = "vs_default"

HIGHER_SCORE_MARKER = r"\higherscore" # r"\colorbox{mygreen}{$\uparrow$}"
LOWER_SCORE_MARKER = r"\lowerscore" # r"\colorbox{myred}{$\downarrow$}"