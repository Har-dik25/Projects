def refine_prompt(prompt: str) -> str:
    refined = (
        f"Cinematic shot of a 1950s–1970s vintage classic car. "
        f"{prompt}. Film grain, warm lighting, shallow depth of field, "
        f"slow camera movement, ultra realistic, vintage photography style."
    )
    return refined
