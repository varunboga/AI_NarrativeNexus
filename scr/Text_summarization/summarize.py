import google.generativeai as genai

API_KEY = "AIzaSyC8SMeDr6eJo0tnGrfATDAx6AtyboFJ8xQ"
genai.configure(api_key=API_KEY)
MODEL = "gemini-2.5-flash"

def get_summary(text: str, max_tokens: int = 200) -> str:
    prompt = f"Provide a clear, concise summary (5-6 sentences) of the following text:\n\n{text}"
    try:
        response = genai.GenerativeModel(MODEL).generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Error generating summary: {e}]"
