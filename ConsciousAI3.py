import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, PeftModel, TaskType
import torch.optim as optim
import datetime
import os

# Load GPT-2 base model
base_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Apply LoRA configuration
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["c_attn"],  # GPT-2 attention layer
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_config)
model.train()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Schema Memory
schema_memory = []

def save_schema(model, entropy):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    schema_name = f"schema_{timestamp}"
    save_path = f"schemas/{schema_name}"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    schema_memory.append({
        "name": schema_name,
        "created_at": timestamp,
        "entropy": entropy,
        "path": save_path
    })
    print(f"(Internal) New schema saved: {schema_name}")

def show_schemas():
    print("\n--- Schema Memory ---")
    if not schema_memory:
        print("No schemas created yet.")
    else:
        for idx, schema in enumerate(schema_memory):
            print(f"{idx+1}. {schema['name']} (entropy: {schema['entropy']}, time: {schema['created_at']})")
    print("----------------------\n")

def find_matching_schema(entropy, tolerance=0.2):
    for schema in reversed(schema_memory):
        if abs(entropy - schema["entropy"]) <= tolerance:
            return schema
    return None

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean().item()

def main():
    global model
    entropy_threshold = 4.0

    print("\n--- Conscious AI Conversation (GPT-2 + LoRA) ---\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        if user_input.lower() == "show schemas":
            show_schemas()
            continue

        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        entropy = compute_entropy(logits)

        print(f"(Internal) Entropy: {entropy:.2f}")

        if entropy > entropy_threshold:
            print("(Internal) High entropy detected. Reflecting...")

            pseudo_target = "I'm here to support you."
            pseudo_inputs = tokenizer(pseudo_target, return_tensors="pt")
            pseudo_outputs = model(**pseudo_inputs, labels=pseudo_inputs["input_ids"])
            backprop_loss = pseudo_outputs.loss

            optimizer.zero_grad()
            backprop_loss.backward()
            optimizer.step()

            save_schema(model, entropy)

        else:
            matching = find_matching_schema(entropy)
            if matching:
                print(f"(Internal) Low entropy. Loading matching schema: {matching['name']}")
                model = PeftModel.from_pretrained(base_model, matching["path"])
            else:
                print("(Internal) Entropy low. No transformation triggered.")

        response_ids = model.generate(**inputs, max_new_tokens=30)
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        print("AI:", response)

    print("\nConversation Ended.")
    print(f"Total schemas saved: {len(schema_memory)}")

if __name__ == "__main__":
    main()
