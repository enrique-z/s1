import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from datetime import datetime
import os
import sys
import tempfile

# Create responses directory if it doesn't exist
os.makedirs('responses', exist_ok=True)

def print_gpu_info():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU Memory: {memory:.2f} GB")

print("=== System Information ===")
print_gpu_info()

print("\n=== Loading Model ===")
start_time = time.time()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "simplescaling/s1-32B",
    trust_remote_code=True
)

# Load model with minimal settings
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "simplescaling/s1-32B",
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "75GB"},
    low_cpu_mem_usage=True  # Faster loading
)

load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds")

def save_response(question: str, response: str, generation_time: float) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"responses/response-{timestamp}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Question and Response - {timestamp}\n\n")
        f.write(f"## Question\n{question}\n\n")
        f.write(f"## Response (Generated in {generation_time:.2f} seconds)\n{response}\n")
    
    print(f"\nResponse saved to: {filename}")
    return filename

def get_multiline_input():
    """Get multi-line input from user. End with 'END' on a new line."""
    print("Enter your question (type 'END' on a new line when finished):")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        except EOFError:
            break
    return '\n'.join(lines)

def get_paste_input():
    """Get pasted input from user with clear instructions"""
    print("\nPaste Mode Instructions:")
    print("1. First, enter your instruction (e.g., 'analyze following text', 'summarize this article')")
    print("2. Then paste your text")
    print("3. Type 'DONE' on a new line when finished")
    
    print("\nEnter your instruction:")
    instruction = input().strip()
    
    print("\nNow paste your text:")
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'DONE':
                break
            lines.append(line)
        except EOFError:
            break
    
    text = '\n'.join(lines)
    return f"{instruction}:\n\n{text}"

# Initialize conversation history
conversation_history = []

def get_response(question: str) -> str:
    global conversation_history
    
    # Add user's question to history
    conversation_history.append({"role": "user", "content": question})
    
    # Format all messages including history
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You should respond only in english language, never in chinese. You should think step-by-step and analyze text thoroughly when asked. For analysis tasks, break down your response into clear sections."}
    ] + conversation_history
    
    # Generate response
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    print("\nGenerating response...")
    start_time = time.time()
    
    # Use fixed high token limit to ensure complete responses
    outputs = model.generate(
        **inputs,
        max_new_tokens=32768,  # Maximum possible tokens
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,  # Ensure proper ending
        stopping_criteria=None,  # Don't stop early
        min_new_tokens=10  # Ensure some minimum response
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_time = time.time() - start_time
    
    try:
        response = response.split("assistant\n")[-1].strip()
    except:
        response = response.strip()
    
    # Add assistant's response to history
    conversation_history.append({"role": "assistant", "content": response})
    
    # Limit history to last 10 exchanges to prevent context overflow
    if len(conversation_history) > 20:  # 10 exchanges = 20 messages
        conversation_history = conversation_history[-20:]
    
    # Save the response to a file
    save_response(question, response, generation_time)
    
    print(f"Generated in {generation_time:.2f} seconds")
    return response

print("\n=== S1-32B Interactive Chat ===")
print("Commands:")
print("- Type 'exit' to quit")
print("- Type 'clear' to clear conversation history")
print("- Type 'long' to enter multi-line question mode")
print("- Type 'paste' to open editor for pasting long text")
print("- Type 'load' to load question from file")
print("----------------------------------------")

while True:
    try:
        command = input("\nYou: ").strip()
        
        if command.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
            
        if command.lower() == 'clear':
            conversation_history = []
            print("Conversation history cleared!")
            continue
            
        if command.lower() == 'long':
            question = get_multiline_input()
        elif command.lower() == 'paste':
            question = get_paste_input()
        elif command.lower() == 'load':
            filename = input("Enter path to question file: ").strip()
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    question = f.read()
                print(f"Loaded question from {filename}")
            except Exception as e:
                print(f"Error loading file: {str(e)}")
                continue
        else:
            question = command
            
        if not question:
            continue
            
        # Add analysis instruction if it's a long text
        if len(question.split()) > 50:  # If text is longer than 50 words
            question = "Please analyze the following text thoroughly:\n\n" + question
            
        response = get_response(question)
        print("\nAssistant:", response)
        print("----------------------------------------")
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {str(e)}")
        continue

# Print final memory usage
if torch.cuda.is_available():
    print(f"\nFinal GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Final GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")