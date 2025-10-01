import os


def _abbreviate_text(text: str, max_len: int = 100) -> str:
    """Shorten long text for logging, keeping start and end."""
    if len(text) <= max_len:
        return text
    half = 50
    return (
        text[:half]
        + " ... ABBRIDGED-FOR-LOG ... "
        + text[-half:]
    )

def log_dspy_messages(history_entry: dict, save_dir: str = "logs"):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "dspy_messages.log")

    # timestamp = datetime.datetime.now().isoformat()
    # ['prompt', 'messages', 'kwargs', 'response', 'outputs', 'usage', 'cost', 'timestamp', 'uuid', 'model', 'response_model', 'model_type']
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n--- DSPy call @ {history_entry["timestamp"]} ---\n")
        for message in history_entry.get("messages", []):
            role = message.get("role", "unknown").upper()
            content = message.get("content", "")
            if role == "USER":
                content = _abbreviate_text(content, max_len=100)
            f.write(f"{role}:\n{content}\n\n")

        f.write(f"Output:\n{history_entry.get('outputs', '')}\n")
               
            
            
        f.write("-" * 20 + "\n")