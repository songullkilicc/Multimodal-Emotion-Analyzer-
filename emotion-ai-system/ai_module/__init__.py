def __init__(self):
    self.client = genai.Client(api_key=GEMINI_API_KEY)
    self.system_prompt = """..."""  # aynı kalacak
    print("[AIInterpreter] Başlatıldı.")