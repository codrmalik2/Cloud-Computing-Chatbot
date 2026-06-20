import os
from dotenv import load_dotenv 

load_dotenv()
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from groq import Groq
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app) # Yeh line Chrome ka connection error hamesha ke liye khatam kar degi

# =====================================================================
# 1.  API KEYS 
# =====================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")

# Clients Setup
genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
hf_client = InferenceClient(token=HF_API_KEY)

# =====================================================================
# 2. AI RESPONSE LOGIC
# =====================================================================
def get_ai_response(user_prompt):
    # Layer 1: Gemini
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(user_prompt)
        if response.text:
            return response.text
    except Exception:
        pass

    # Layer 2: Groq Llama 3
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": user_prompt}],
        )
        if completion.choices[0].message.content:
            return completion.choices[0].message.content
    except Exception:
        pass

    # Layer 3: Hugging Face
    try:
        response = hf_client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=500
        )
        if response.choices[0].message.content:
            return response.choices[0].message.content
    except Exception as e:
        return f"Error: All models failed. {e}"

# =====================================================================
# 3. CHROME KE SATH CONNECT HONE WALA RASTA (ROUTE)
# =====================================================================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")
    reply = get_ai_response(user_message)
    return jsonify({"response": reply})

if __name__ == "__main__":
    print("\n=== BACKEND SERVER IS RUNNING ON PORT 5000 ===")
    app.run(port=5000, debug=True)