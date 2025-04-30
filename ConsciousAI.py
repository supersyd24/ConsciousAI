# conscious_ai_conversation.py

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import pipeline
import datetime

# Load sentiment model
sentiment_pipeline = pipeline("sentiment-analysis")

# 1. Define the ConsciousNet
class ConsciousNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ConsciousNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 2. Fuzzy Logic System
def fuzzy_score(input_features):
    weights = torch.tensor([0.4, 0.3, 0.3])
    score = torch.dot(weights, input_features)
    return score.item()

# 3. Reflection Module
def reflect(internal_state):
    reflection_questions = [
        internal_state["emotion_recognition"],
        internal_state["memory_match"],
        internal_state["context_completeness"],
        internal_state["predictive_confidence"],
        internal_state["purpose_alignment"]
    ]
    reflection_score = sum(reflection_questions) / len(reflection_questions)
    return reflection_score

# 4. Autonomous Backpropagation
def autonomous_backprop(model, optimizer, input_tensor, target_tensor):
    model.train()  # 1. Tell the model to get ready for training
    output = model(input_tensor)  # 2. Forward pass: get the model’s prediction
    loss_fn = nn.MSELoss()  # 3. Define the loss function: Mean Squared Error
    loss = loss_fn(output, target_tensor)  # 4. Calculate the loss (the 'pain')
    optimizer.zero_grad()  # 5. Clear previous gradients
    loss.backward()  # 6. Backward pass: calculate gradients (how much each weight contributed to the error)
    optimizer.step()  # 7. Update weights: change the internal model to reduce future error


# 5. Schema Memory
schema_memory = []

def save_schema(model_state_dict, trigger_conditions):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    schema_memory.append({
        "model": model_state_dict,
        "created_at": timestamp,
        "trigger_conditions": trigger_conditions
    })
    print(f"(Internal) New schema saved at {timestamp} with trigger: {trigger_conditions}")

def find_closest_schema(fuzzy_score, tolerance=0.05):
    for schema in reversed(schema_memory):  # prioritize recent schemas
        past_score = schema["trigger_conditions"].get("fuzzy_score")
        if abs(fuzzy_score - past_score) <= tolerance:
            return schema
    return None

# 6. Schema Viewer
def show_schemas():
    print("\n--- Schema Memory ---")
    if not schema_memory:
        print("No schemas created yet.")
    else:
        for idx, schema in enumerate(schema_memory):
            print(f"Schema {idx+1}: Created at {schema['created_at']}, Trigger Conditions: {schema['trigger_conditions']}")
    print("----------------------\n")

# 7. Emotion Analyzer
def analyze_emotion(user_message):
    result = sentiment_pipeline(user_message)[0]
    label = result['label']
    score = result['score']

    print(f"(Internal) Sentiment detected: {label} with confidence {score:.2f}")

    if label == "NEGATIVE":
        return torch.tensor([0.9, 0.9, 0.8])
    elif label == "POSITIVE":
        return torch.tensor([0.2, 0.1, 0.1])
    else:
        return torch.tensor([0.5, 0.5, 0.5])

# 8. Main Conversation Loop
def main():
    input_size = 3
    hidden_size = 5
    output_size = 1
    model = ConsciousNet(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    fuzzy_threshold = 0.75
    reflection_threshold = 0.4

    print("\n--- Conscious AI Conversation Start ---\n")

    while True:
        user_message = input("User: ")
        if user_message.lower() in ["quit", "exit"]:
            print("\nConversation Ended.")
            break
        if user_message.lower() == "show schemas":
            show_schemas()
            continue

        # Analyze user's message
        input_features = analyze_emotion(user_message)
        target = torch.tensor([1.0])  # Placeholder target

        # Step 1: Fuzzy Logic
        mu = fuzzy_score(input_features)
        print(f"(Internal) Fuzzy Score: {mu:.2f}")

        if mu > fuzzy_threshold:
            print("(Internal) High ambiguity detected. Reflecting...")

            internal_state = {
                "emotion_recognition": 0.6,
                "memory_match": 0.3,
                "context_completeness": 0.4,
                "predictive_confidence": 0.2,
                "purpose_alignment": 0.5
            }
            reflection_result = reflect(internal_state)
            print(f"(Internal) Reflection Score: {reflection_result:.2f}")

            if reflection_result < reflection_threshold:
                print("(Internal) Low confidence. Searching for prior experience...")

                matching_schema = find_closest_schema(mu)
                if matching_schema:
                    model.load_state_dict(matching_schema["model"])
                    print(f"(Internal) Reused schema from {matching_schema['created_at']} with fuzzy_score ≈ {matching_schema['trigger_conditions']['fuzzy_score']:.2f}")
                else:
                    print("(Internal) No prior match found. Initiating conscious transformation...")
                    autonomous_backprop(model, optimizer, input_features, target)
                    save_schema(model.state_dict(), trigger_conditions={"fuzzy_score": mu})
                    print("(Internal) New schema created and saved.")
            else:
                print("(Internal) Reflection sufficient. Minor adjustment or no update needed.")
        else:
            print("(Internal) No need for major transformation.")

        print(f"AI: Thank you for sharing that. I'm adapting my understanding.")

    print("\n--- Conscious AI Conversation End ---\n")
    print(f"Total schemas saved during conversation: {len(schema_memory)}")

if __name__ == "__main__":
    main()
