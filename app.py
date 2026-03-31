from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
import json
import io
import os
import re

app = Flask(__name__)
os.makedirs(app.instance_path, exist_ok=True)
LAYOUT_FILE = os.path.join(app.instance_path, 'dashboard_layout.json')

# --- LLM Configuration ---
encry = "um/qt/x3/::46772hdh:;779575fd;425485d;ee74gh4f8;8ec8:3665;67e5fgh8;67c8dc"
OPENROUTER_API_KEY = ""
for i in encry:
    OPENROUTER_API_KEY = OPENROUTER_API_KEY+chr(ord(i)-2)
chat_history = []

def ask_llm(question, context):
    global chat_history
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Dataset Summary Context:
{context}

Question:
{question}

Answer clearly like a data analyst explaining a dashboard.
"""

    chat_history.append({
        "role": "user",
        "content": prompt
    })

    payload = {
        "model": "nvidia/nemotron-3-nano-30b-a3b:free",
        "messages": [
            {"role": "system", "content": "You are an AI data analysis assistant."}
        ] + chat_history
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        res_json = response.json()

        if "choices" in res_json and len(res_json["choices"]) > 0:
            answer = res_json["choices"][0]["message"]["content"]
            chat_history.append({
                "role": "assistant",
                "content": answer
            })
            return answer
        else:
            return f"Error from LLM API: {json.dumps(res_json)}"
    except Exception as e:
        return f"Request failed: {str(e)}"

def ask_llm_json(context):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
Analyze this dataset summary:
{context}

Suggest exactly 4 insightful charts that would make a great dashboard.
You MUST respond ONLY with a raw JSON array. No markdown, no text, no explanations.
Format exactly like this:
[
  {{"chart_type": "bar", "x_axis": "CategoricalColumnName", "y_axis": "NumericColumnName", "title": "Chart 1 Title"}},
  {{"chart_type": "pie", "x_axis": "CategoricalColumnName", "y_axis": "NumericColumnName", "title": "Chart 2 Title"}},
  {{"chart_type": "line", "x_axis": "CategoricalColumnName", "y_axis": "NumericColumnName", "title": "Chart 3 Title"}},
  {{"chart_type": "doughnut", "x_axis": "CategoricalColumnName", "y_axis": "NumericColumnName", "title": "Chart 4 Title"}}
]
Choose 'chart_type' from: 'bar', 'line', 'pie', 'doughnut'.
Choose exact column names from the summary.
"""
    
    payload = {
        "model": "nvidia/nemotron-3-nano-30b-a3b:free",
        "messages": [
            {"role": "system", "content": "You are a data architect. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        res_json = response.json()
        
        if "choices" in res_json and len(res_json["choices"]) > 0:
            answer = res_json["choices"][0]["message"]["content"]
            # Extract JSON block using regex in case the model adds conversational padding
            match = re.search(r'\[.*\]', answer, re.DOTALL)
            if match:
                answer = match.group(0)
            return json.loads(answer)
        return []
    except Exception as e:
        return {"error": str(e)}

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Please upload CSV or Excel."}), 400

        df = df.fillna(0)

        summary = f"Columns: {list(df.columns)}\n"
        summary += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
        summary += f"Data Types:\n{df.dtypes.astype(str).to_string()}\n"
        summary += f"Statistical Summary:\n{df.describe().to_string()}\n"
        
        df_subset = df.head(5000)
        
        return jsonify({
            "columns": list(df.columns),
            "data": df_subset.to_dict(orient='records'),
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    req_data = request.json
    question = req_data.get('question', '')
    context = req_data.get('context', '')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
        
    answer = ask_llm(question, context)
    return jsonify({"answer": answer})

@app.route('/generate_dashboard', methods=['POST'])
def generate_dashboard():
    req_data = request.json
    context = req_data.get('context', '')
    if not context:
        return jsonify({"error": "No context provided"}), 400
    
    charts_config = ask_llm_json(context)
    # Validate that we got exactly 4 charts
    if isinstance(charts_config, list) and len(charts_config) == 4:
        return jsonify({"charts": charts_config})
    else:
        return jsonify({"error": "Failed to generate valid 4-chart dashboard configuration.", "raw": str(charts_config)}), 500

# --- NEW: Layout Saving and Loading Routes ---
@app.route('/save_layout', methods=['POST'])
def save_layout():
    try:
        data = request.json
        with open(LAYOUT_FILE, 'w') as f:
            json.dump(data, f)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/load_layout', methods=['GET'])
def load_layout():
    try:
        if os.path.exists(LAYOUT_FILE):
            with open(LAYOUT_FILE, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        return jsonify({})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5000)
