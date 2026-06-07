"""Conversation trimming and response grounding checks."""

import re

from course_utils import display_course_name


def recent_conversation(messages, keep=6, char_limit=900):
    recent = messages[-keep:]
    trimmed = []
    for message in recent:
        content = " ".join(str(message.get("content", "")).split())
        if len(content) > char_limit:
            content = content[:char_limit].rsplit(" ", 1)[0] + "..."
        trimmed.append({"role": message.get("role", "user"), "content": content})
    return trimmed


def validate_grounding(reply, selected_courses, filtered_courses):
    known_codes = set(filtered_courses.keys())
    evidence_codes = set(selected_courses.keys())
    mentioned_codes = set(re.findall(r"\b[A-Z]{2,4}\d{4}(?:-\d{2}){0,2}\b", reply or ""))
    unknown_codes = sorted(mentioned_codes - known_codes)
    weak_codes = sorted((mentioned_codes & known_codes) - evidence_codes)

    notes = []
    if unknown_codes:
        notes.append(
            "Grounding check: I do not have database evidence for "
            + ", ".join(unknown_codes)
            + ". Please ignore those course references unless you add them to the data."
        )
    if weak_codes:
        weak_labels = [
            f"{display_course_name(filtered_courses[code].get('metadata', {}).get('name', code))} ({code})"
            for code in weak_codes
        ]
        notes.append(
            "Grounding note: "
            + ", ".join(weak_labels)
            + " appeared in the filtered catalog but was not part of this turn's detailed evidence. Treat those mentions as candidate-level only."
        )
    if notes:
        return reply.rstrip() + "\n\n" + "\n".join(notes)
    return reply
