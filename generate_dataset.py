"""
Generates a larger, balanced synthetic dataset for the emotion classifier.

This replaces the original 5-row dataset.csv with ~480 labeled short texts
across 6 emotion classes, built from templates + phrase banks so there's
lexical variety (not just 5 fixed sentences repeated).

NOTE: This is still a *synthetic, illustrative* dataset for a demo classifier.
It is not a clinical or validated mental-health dataset, and the resulting
model should not be treated as a diagnostic tool.
"""

import csv
import random

random.seed(42)

# Each emotion has: subjects/openers, feelings, contexts -> combined into sentences
TEMPLATES = {
    "depression": {
        "openers": ["I feel", "Lately I've felt", "Honestly, I feel", "I've been feeling", "These days I feel"],
        "feelings": [
            "hopeless and empty", "numb most of the time", "like nothing matters anymore",
            "exhausted no matter how much I sleep", "like I have no motivation to do anything",
            "sad for no clear reason", "like a burden to everyone around me",
            "like I can't enjoy things I used to love", "worthless", "like I'm just going through the motions",
            "like getting out of bed takes everything I have", "disconnected from everyone",
        ],
        "contexts": [
            "", " every single day", " and it's been going on for weeks", " even when good things happen",
            " and I don't know why", " no matter what I try",
        ],
    },
    "stress": {
        "openers": ["I feel", "I'm", "I've been", "I am", "Right now I feel"],
        "feelings": [
            "so stressed about my exams", "overwhelmed with deadlines at work",
            "under a lot of pressure lately", "stretched way too thin",
            "stressed about money", "swamped with responsibilities",
            "anxious about everything I have to finish", "running on no sleep because of stress",
            "tense because of all the deadlines", "overloaded with tasks I can't keep up with",
        ],
        "contexts": [
            "", " and I can't relax", " and it's affecting my sleep", " this week",
            " and I don't see it slowing down", " with no time to breathe",
        ],
    },
    "anxiety": {
        "openers": ["I feel", "I'm", "I've been", "My mind is", "I keep feeling"],
        "feelings": [
            "really anxious about the future", "nervous and can't sit still",
            "worried something bad is going to happen", "on edge all the time",
            "racing with worried thoughts", "panicky before I have to speak in public",
            "restless and jittery", "scared even though nothing is technically wrong",
            "anxious about things I can't control", "overthinking every small decision",
        ],
        "contexts": [
            "", " and my heart races", " and I can't calm down", " for no obvious reason",
            " before big events", " and it's hard to focus on anything else",
        ],
    },
    "anger": {
        "openers": ["I feel", "I'm", "I've been", "Right now I'm", "I get"],
        "feelings": [
            "so angry and frustrated", "furious about how I was treated", "irritated by everything today",
            "annoyed at my coworkers", "mad about the unfair situation", "fed up with being ignored",
            "angry that things never seem to go my way", "frustrated that no one listens to me",
            "pissed off about the whole thing", "resentful about how it turned out",
        ],
        "contexts": [
            "", " and I want to yell", " and I can't shake it off", " every time it happens",
            " and it's hard to stay calm", " and I just need to vent",
        ],
    },
    "happy": {
        "openers": ["I feel", "I'm", "I've been", "Today I feel", "Right now I'm"],
        "feelings": [
            "so happy and grateful", "really excited about the future", "proud of what I accomplished",
            "content with how things are going", "thrilled about the good news",
            "in a great mood today", "feeling optimistic about everything",
            "cheerful and full of energy", "delighted with how the day went",
            "hopeful and looking forward to what's next",
        ],
        "contexts": [
            "", " and it feels amazing", " for the first time in a while", " and I want to celebrate",
            " and everything feels right", " because of the people around me",
        ],
    },
    "neutral": {
        "openers": ["I feel", "I'm", "Today was", "Things have been", "I'd say I'm"],
        "feelings": [
            "calm and relaxed", "pretty okay, nothing special", "fine, just an average day",
            "steady, not too high or low", "content with a normal routine", "quiet and uneventful",
            "just going about my day as usual", "settled and at ease", "neither happy nor sad, just neutral",
            "peaceful this afternoon",
        ],
        "contexts": [
            "", " and that's fine by me", " with nothing much happening", " like most days",
            " and I don't mind it", " overall",
        ],
    },
}

SAMPLES_PER_CLASS = 80


def build_sentence(opener: str, feeling: str, context: str) -> str:
    text = f"{opener} {feeling}{context}."
    return text[0].upper() + text[1:]


def generate():
    rows = []
    for emotion, parts in TEMPLATES.items():
        combos = [
            (o, f, c)
            for o in parts["openers"]
            for f in parts["feelings"]
            for c in parts["contexts"]
        ]
        random.shuffle(combos)
        seen = set()
        count = 0
        for o, f, c in combos:
            sentence = build_sentence(o, f, c)
            if sentence in seen:
                continue
            seen.add(sentence)
            rows.append((sentence, emotion))
            count += 1
            if count >= SAMPLES_PER_CLASS:
                break
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = generate()
    with open("dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "emotion"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows across {len(TEMPLATES)} classes to dataset.csv")
