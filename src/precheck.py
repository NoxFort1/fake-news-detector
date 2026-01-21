"""Text validation and precheck functions."""

import re


OFFENSIVE_WORDS = [
    "fuck", "fucking", "fucked", "fucker", "fucks", "fuckin",
    "shit", "shitty", "bullshit", "shitting",
    "bitch", "bitches", "bitchy",
    "ass", "asshole", "asses", "arsehole",
    "bastard", "bastards",
    "damn", "damned", "damnit",
    "crap", "crappy",
    "hell",
    "piss", "pissed", "pissing",
    "cunt", "cunts",
    "dick", "dicks", "dickhead",
    "cock", "cocks",
    "motherfucker", "motherfucking", "mf",
    "slut", "sluts", "slutty",
    "whore", "whores",
    "fag", "faggot", "fags",
    "nigger", "nigga", "niggas",
    "retard", "retarded", "retards",
    "idiot", "idiots", "idiotic",
    "moron", "morons", "moronic",
    "stupid", "dumb", "dumbass",
    "loser", "losers",
    "trash", "garbage",
    "scum", "scumbag",
]


def count_offensive_words(text: str) -> int:
    """Count the number of offensive words in the text."""
    words = re.findall(r"\b\w+\b", text.lower())
    return sum(1 for w in words if w in OFFENSIVE_WORDS)


def local_text_precheck(text: str) -> tuple[bool, str]:
    """
    Validate text before analysis.
    
    Returns:
        tuple: (is_suitable, reason)
    """
    cleaned = text.strip()
    
    if len(cleaned) < 10:
        return False, "Text is too short for reliable analysis."

    words = re.findall(r"\b\w+\b", cleaned.lower())
    total_words = len(words)
    unique_words = len(set(words))
    
    if unique_words < 3:
        return False, "Text has too few unique words."
    
    offensive_count = count_offensive_words(cleaned)
    
    if total_words > 0:
        offensive_ratio = offensive_count / total_words
        
        if offensive_ratio > 0.3:
            return False, "Text contains too high a ratio of offensive words."
        
        if total_words < 10 and offensive_count > 0:
            return False, "Short text with offensive content is not suitable for analysis."
        
        if total_words < 20 and offensive_count > 1:
            return False, "Text contains too many offensive words for its length."
        
        if offensive_count > 2:
            return False, "Text contains too many offensive words."
    
    if total_words > 5:
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        max_repetition = max(word_counts.values())
        if max_repetition / total_words > 0.5:
            return False, "Text appears to be spam (too many repeated words)."

    non_letter_ratio = len(re.findall(r"[^a-zA-Z\s]", cleaned)) / max(len(cleaned), 1)
    if non_letter_ratio > 0.4:
        return False, "Text contains too many non-letter characters."

    return True, "Text passes basic quality checks."
