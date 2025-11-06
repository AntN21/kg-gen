from typing import List
import dspy
from pydantic import BaseModel
from .logg import log_dspy_messages
from dspy.utils.exceptions import AdapterParseError

def extraction_sig(
    Relation: BaseModel, is_conversation: bool, context: str = ""
) -> dspy.Signature:
    if not is_conversation:

        class ExtractTextRelations(dspy.Signature):
            __doc__ = f"""Extract action1-time-action2 triples from tutorial youtube video transcripts.
            1. Action1 and action2 are verb-object phrases.
            2. Action1 is related to action2 by a temporal relation.
            3. Time is a temporal relation between action1 and action2 (e.g. before, after, during, while, then, next, later, etc.)
      Actions must be from actions list. Actions provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text. 
      Respect the structure. Do not forget spaces in the output format. {context}"""
    #         __doc__ = f"""Extract subject-predicate-object triples from the source text. 
    #   Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
    #   This is for an extraction task, please be thorough, accurate, and faithful to the reference text. {context}"""

            source_text: str = dspy.InputField()
            actions: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.OutputField(
                desc="List of action1-time-action2 tuples. Be thorough."
            )

        return ExtractTextRelations
    else:

        class ExtractConversationRelations(dspy.Signature):
            __doc__ = f"""Extract subject-predicate-object triples from the conversation, including:
      1. Relations between concepts discussed
      2. Relations between speakers and concepts (e.g. user asks about X)
      3. Relations between speakers (e.g. assistant responds to user)
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text. {context}"""

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.OutputField(
                desc="List of subject-predicate-object tuples where subject and object are exact matches to items in entities list. Be thorough"
            )

        return ExtractConversationRelations


def fallback_extraction_sig(
    entities, is_conversation, context: str = ""
) -> dspy.Signature:
    """This fallback extraction does not strictly type the action strings."""

    entities_str = "\n- ".join(entities)

    class Relation(BaseModel):
        # TODO: should use literal's here instead.
        __doc__ = f"""Knowledge graph action1-time-action2 tuple. Actions must be one of: {entities_str}"""

        action1: str = dspy.InputField(desc="Action1", examples=["Add flour"])
        time: str = dspy.InputField(desc="Time Relation", examples=["Then"])
        action2: str = dspy.InputField(desc="Action2", examples=["Mix ingredients"])

    return Relation, extraction_sig(Relation, is_conversation, context)


def get_relations(
    input_data: str,
    entities: list[str],
    is_conversation: bool = False,
    context: str = "",
    save_dir = "logs",
) -> List[str]:
    class Relation(BaseModel):
        """Knowledge graph action1-time-action2 tuple."""

        action1: str = dspy.InputField(desc="Action1", examples=["Add flour"])
        time: str = dspy.InputField(desc="Time Relation", examples=["Then"])
        action2: str = dspy.InputField(desc="Action2", examples=["Mix ingredients"])

    ExtractRelations = extraction_sig(Relation, is_conversation, context)

    try:
        extract = dspy.Predict(ExtractRelations)
        result = extract(source_text=input_data, entities=entities)
        # Grab the most recent history entry
        history_entry = extract.history[-1]
        log_dspy_messages(history_entry, save_dir=save_dir)
        return [(r.action1, r.time, r.action2) for r in result.relations]
        #return [(r.subject, r.predicate, r.object) for r in result.relations]

    except Exception as _:
        Relation, ExtractRelations = fallback_extraction_sig(
            entities, is_conversation, context
        )
        extract = dspy.Predict(ExtractRelations)
        result = extract(source_text=input_data, entities=entities)
        # Grab the most recent history entry
        history_entry = extract.history[-1]
        log_dspy_messages(history_entry, save_dir=save_dir)

        class FixedRelations(dspy.Signature):
            """Fix the relations so that every actions of the relations are exact matches to an action previously extracted. Keep the predicate the same. The meaning of every relation should stay faithful to the reference text. If you cannot maintain the meaning of the original relation relative to the source text, then do not return it."""

            source_text: str = dspy.InputField()
            actions: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.InputField()
            fixed_relations: list[Relation] = dspy.OutputField()

        fix = dspy.ChainOfThought(FixedRelations)
        try:
            fix_res = fix(
                source_text=input_data, actions=entities, relations=result.relations
            )

            good_relations = []
            for rel in fix_res.fixed_relations:
                if rel.action1 in entities and rel.action2 in entities:
                    good_relations.append(rel)
            return [(r.action1, r.time, r.action2) for r in good_relations]
        
        except AdapterParseError as e:
            print(f"AdapterParseError during relation fixing: {e}")
            return []
        # return [(r.subject, r.predicate, r.object) for r in good_relations]
