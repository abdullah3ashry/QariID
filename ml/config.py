"""Shared settings for the Phase 1 proof of concept."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"          # downloaded per-ayah MP3s: raw/<subfolder>/<SSSAAA>.mp3
CACHE = DATA / "wav16k"     # decoded 16 kHz mono surahs: wav16k/<subfolder>/<SSS>.flac
MODELS = DATA / "models"    # pretrained model files
RESULTS = ROOT / "results"
ENROLLMENT_FILE = DATA / "reciters.npz"

SAMPLE_RATE = 16000
BASE_URL = "https://everyayah.com/data"

# Ayah counts for surahs 1..114 (from everyayah.com/data/recitations.js).
AYAH_COUNT = [
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128,
    111, 110, 98, 135, 112, 78, 118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73,
    54, 45, 83, 182, 88, 75, 85, 54, 53, 89, 59, 37, 35, 38, 29, 18, 45, 60,
    49, 62, 55, 78, 96, 29, 22, 24, 13, 14, 11, 11, 18, 12, 12, 30, 52, 52,
    44, 28, 28, 20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25, 22, 17, 19,
    26, 30, 20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3,
    6, 3, 5, 4, 5, 6,
]

# Surahs used to build each reciter's voice vector, and different surahs used
# for testing, so the model is never tested on words it was enrolled with.
ENROLL_SURAHS = [36, 55, 67]   # Ya-Sin, Ar-Rahman, Al-Mulk
TEST_SURAHS = [18, 50]         # Al-Kahf, Qaf
CROSS_SURAHS = [50]            # Qaf only (mujawwad recordings are long)

# The 20 reciters the prototype should recognise: id -> (display name, everyayah folder)
KNOWN = {
    "abdul_basit": ("Abdul Basit Abdus-Samad", "Abdul_Basit_Murattal_192kbps"),
    "basfar": ("Abdullah Basfar", "Abdullah_Basfar_192kbps"),
    "sudais": ("Abdurrahman As-Sudais", "Abdurrahmaan_As-Sudais_192kbps"),
    "shaatree": ("Abu Bakr Ash-Shaatree", "Abu_Bakr_Ash-Shaatree_128kbps"),
    "ajamy": ("Ahmed Al-Ajamy", "ahmed_ibn_ali_al_ajamy_128kbps"),
    "alafasy": ("Mishary Alafasy", "Alafasy_128kbps"),
    "rifai": ("Hani Ar-Rifai", "Hani_Rifai_192kbps"),
    "husary": ("Mahmoud Khalil Al-Husary", "Husary_128kbps"),
    "hudhaify": ("Ali Al-Hudhaify", "Hudhaify_128kbps"),
    "muaiqly": ("Maher Al-Muaiqly", "MaherAlMuaiqly128kbps"),
    "minshawy": ("Mohamed Siddiq Al-Minshawy", "Minshawy_Murattal_128kbps"),
    "tablawi": ("Mohammad At-Tablawi", "Mohammad_al_Tablaway_128kbps"),
    "ayyoub": ("Muhammad Ayyoub", "Muhammad_Ayyoub_128kbps"),
    "jibreel": ("Muhammad Jibreel", "Muhammad_Jibreel_128kbps"),
    "shuraym": ("Saud Ash-Shuraym", "Saood_ash-Shuraym_128kbps"),
    "bukhatir": ("Salah Bukhatir", "Salaah_AbdulRahman_Bukhatir_128kbps"),
    "juhany": ("Abdullah Awad Al-Juhany", "Abdullaah_3awwaad_Al-Juhaynee_128kbps"),
    "dussary": ("Yasser Ad-Dussary", "Yasser_Ad-Dussary_128kbps"),
    "qatami": ("Nasser Al-Qatami", "Nasser_Alqatami_128kbps"),
    "ghamdi": ("Saad Al-Ghamdi", "Ghamadi_40kbps"),
}

# Reciters deliberately NOT enrolled. The system should answer "unknown" for them.
UNKNOWN = {
    "qasim": ("Muhsin Al-Qasim", "Muhsin_Al_Qasim_192kbps"),
    "budair": ("Salah Al-Budair", "Salah_Al_Budair_128kbps"),
    "matroud": ("Abdullah Matroud", "Abdullah_Matroud_128kbps"),
    "qahtani": ("Khalid Al-Qahtani", "Khaalid_Abdullaah_al-Qahtaanee_192kbps"),
    "abbad": ("Fares Abbad", "Fares_Abbad_64kbps"),
}

# A different recording (mujawwad style) of an enrolled reciter. Tests whether
# the model recognises the voice or merely the recording/studio.
CROSS = {
    "abdul_basit": "Abdul_Basit_Mujawwad_128kbps",
    "husary": "Husary_128kbps_Mujawwad",
    "minshawy": "Minshawy_Mujawwad_192kbps",
}

# ---------------------------------------------------------------------------
# Phase 2: fine-tuning dataset
# ---------------------------------------------------------------------------
TRAIN = DATA / "train"            # train/<rec_id>/<SSS>.flac  (16 kHz, silence removed)
EVAL = DATA / "eval"              # eval/<rec_id>/<SSS>.flac   (never used for training)
CATALOG_FILE = TRAIN / "catalog.json"
MANIFEST_FILE = TRAIN / "manifest.csv"

MP3QURAN_API = "https://www.mp3quran.net/api/v3/reciters?language=eng"
QURANICAUDIO_URL = "https://download.quranicaudio.com/quran"

# Surahs that no reciter is ever trained on, so every test stays on unseen words.
EVAL_SURAHS = sorted(set(ENROLL_SURAHS + TEST_SURAHS))
# Long surahs are skipped for training: one file would use most of a reciter's
# budget and decoding a two-hour file needs several GB of memory.
LONG_SURAHS = list(range(2, 10))

# Minutes of speech (after silence removal) to collect per training recording.
BUDGET_MIN = {"murattal": 60, "mujawwad": 120, "other": 60}
MAX_RECORDINGS_PER_RECITER = 3
MIN_SURAHS_PER_RECORDING = 20

# mp3quran.net reciter id for each Phase 1 reciter, so the same person gets one
# label across sources and the Phase 1 "unknown" reciters stay out of training.
MP3QURAN_ID = {
    "abdul_basit": 51, "basfar": 60, "sudais": 54, "shaatree": 4, "ajamy": 5,
    "alafasy": 123, "rifai": 89, "husary": 118, "hudhaify": 74, "muaiqly": 102,
    "minshawy": 112, "tablawi": 106, "ayyoub": 109, "jibreel": 111,
    "shuraym": 31, "bukhatir": 46, "juhany": 62, "dussary": 92, "qatami": 86,
    "ghamdi": 30,
    # Phase 1 unknowns
    "qasim": 67, "budair": 43, "matroud": 59, "qahtani": 21, "abbad": 81,
}

# Extra reciters fully held out (beyond the Phase 1 five), picked at random
# from single-recording reciters when the catalog is built.
EXTRA_UNKNOWN = 25
SPLIT_SEED = 1234

# Recordings never trained on, used to test generalisation:
# style = mujawwad of reciters whose murattal is trained on;
# recording = a different recording (riwayah) of a trained reciter.
HELDOUT_STYLE = {288: "Mustafa Ismail mujawwad", 122: "Mahmoud Ali Al-Banna mujawwad"}
HELDOUT_RECORDING = {
    120: "Husary, Warsh",
    75: "Al-Hudhaify, Qalun",
    178: "Ibrahim Al-Dosari, Warsh",
    199: "Ahmed Al-Tarabulsi, Qalun",
    52: "Abdul Basit, Warsh",
}

# Extra mujawwad source (quranicaudio.com) for reciters missing from mp3quran.
QURANICAUDIO_EXTRA = {
    "qa68": ("Abdur-Razaq bin Abtan Al-Dulaimi", "abdulrazaq_bin_abtan_al_dulaimi/", "mujawwad"),
}
