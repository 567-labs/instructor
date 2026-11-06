import os
import instructor

models = [
    (os.getenv("GOOGLE_GENAI_MODEL") or "gemini-3.8-flash").removeprefix("google/")
]
modes = [instructor.Mode.GENAI_STRUCTURED_OUTPUTS]
