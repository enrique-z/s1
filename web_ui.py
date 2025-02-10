from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

app = Flask(__name__)

# Initialize model and tokenizer globally
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    "simplescaling/s1-32B",
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "75GB"}
)
tokenizer = AutoTokenizer.from_pretrained("simplescaling/s1-32B")
print("Model loaded successfully!")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    message = data.get('message', '')
    system_prompt = data.get('system_prompt', 'You are a helpful and harmless assistant. You should think step-by-step.')
    max_tokens = data.get('max_tokens', 2048)
    temperature = data.get('temperature', 0.7)

    # Format the message
    formatted_message = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    
    # Generate
    start_time = time.time()
    inputs = tokenizer(formatted_message, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_time = time.time() - start_time
    
    # Extract just the assistant's response
    try:
        response = response.split("assistant\n")[-1].strip()
    except:
        response = response.strip()
    
    return jsonify({
        'response': response,
        'generation_time': f"{generation_time:.2f} seconds"
    })

@app.route('/', methods=['GET'])
def home():
    return '''
    <html>
        <body>
            <h2>S1-32B Simple Interface</h2>
            <form id="form">
                <textarea id="message" rows="4" cols="50" placeholder="Enter your message"></textarea><br>
                <button type="submit">Generate</button>
            </form>
            <div id="response" style="white-space: pre-wrap; margin-top: 20px;"></div>
            
            <script>
                document.getElementById('form').onsubmit = async (e) => {
                    e.preventDefault();
                    const message = document.getElementById('message').value;
                    document.getElementById('response').textContent = 'Generating...';
                    
                    try {
                        const response = await fetch('/generate', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({message})
                        });
                        const data = await response.json();
                        document.getElementById('response').textContent = 
                            `Response (${data.generation_time}):\n\n${data.response}`;
                    } catch (error) {
                        document.getElementById('response').textContent = 
                            'Error: ' + error.message;
                    }
                };
            </script>
        </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7860, debug=False) 