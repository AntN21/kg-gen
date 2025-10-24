from typing import List
import dspy

from .logg import log_dspy_messages

class TextEntities(dspy.Signature):
    """
    You are given a transcript of a tutorial youtube video.
    Your task is to read the transcript and decompose it into a list of clear, actionable steps.
    Extract key actions from the source text. Extracted actions are association of a verb and an object
    Write each action as the shortest verb-object phrase possible, e.g. "eat apple", not "I eat an apple".
    This is for an extraction task, please be THOROUGH and accurate to the reference text."""

    #Consider actions that are explicitly mentioned as well as those that are implied.
    source_text: str = dspy.InputField()
    actions: list[str] = dspy.OutputField(desc="THOROUGH list of key actions")


class ConversationEntities(dspy.Signature):
    """Extract key entities from the conversation Extracted entities are subjects or objects.
    Consider both explicit entities and participants in the conversation.
    This is for an extraction task, please be THOROUGH and accurate."""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


def get_entities(input_data: str, is_conversation: bool = False, save_dir = "logs") -> List[str]:
    extract = (
        dspy.Predict(ConversationEntities)
        if is_conversation
        else dspy.Predict(TextEntities)
    )
    result = extract(source_text=input_data)
    # Grab the most recent history entry
    history_entry = extract.history[-1]
    log_dspy_messages(history_entry, save_dir=save_dir)
    return result.actions
