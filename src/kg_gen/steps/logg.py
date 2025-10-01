import os
import datetime

def log_dspy_messages(history_entry: dict, save_dir: str = "logs"):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "dspy_messages.log")

    timestamp = datetime.datetime.now().isoformat()
    print(type(history_entry))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n--- DSPy call @ {timestamp} ---\n")
        f.write(f"SYSTEM:\n{history_entry['system']}\n\n")
        f.write(f"USER:\n{history_entry['user']}\n\n")
        f.write(f"ASSISTANT:\n{history_entry.get('outputs')}\n")
        f.write("-" * 40 + "\n")