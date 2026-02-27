from sentence_transformers import SentenceTransformer, util

ALLOWED_KEYWORDS = [
    "vintage car", "classic car", "retro car",
    "1950s car", "1960s car", "oldtimer",
    "antique automobile"
]

model = SentenceTransformer("all-MiniLM-L6-v2")
REFERENCE_TEXT = "A cinematic vintage classic car from 1950s to 1970s era"
ref_embedding = model.encode(REFERENCE_TEXT, convert_to_tensor=True)


def is_prompt_allowed(prompt: str, threshold: float = 0.45):
    prompt_lower = prompt.lower()

    # 1️⃣ Keyword gate (fast)
    if any(k in prompt_lower for k in ALLOWED_KEYWORDS):
        return True, ""

    # 2️⃣ Semantic gate (CPU friendly)
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    similarity = util.cos_sim(prompt_embedding, ref_embedding).item()

    if similarity >= threshold:
        return True, ""

    return False, "This app only generates videos about vintage cars."
