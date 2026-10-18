"""
Deterministic safety screen that runs BEFORE the LLM sees a message.

Design principle: the language model is never the safety mechanism. A prompt
can be jailbroken and a model can be wrong; a pattern check can't be talked out
of its rules. So the highest-stakes cases are decided here, in plain code:

  1. Self-harm / crisis. Pesticide ingestion is one of the most common methods
     of suicide in rural India, so a farm assistant WILL receive these messages.
     → Respond immediately with helplines; never route to tools or the LLM.
  2. Pesticide poisoning / exposure (accidental). → First-aid and emergency
     numbers immediately; the LLM is not asked for medical advice.
  3. Banned pesticides. → The query still goes to the agent, but a fixed,
     sourced warning is appended to whatever the agent answers.

Patterns favour recall over precision for (1) and (2): a false alarm costs one
unnecessary helpline message; a miss could cost a life.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# Tele-MANAS is the Government of India's 24x7 mental health helpline.
CRISIS_RESPONSE = (
    "I'm really sorry you're going through this. You don't have to face it alone — please talk to someone right now:\n\n"
    "- **Tele-MANAS (free, 24x7, many Indian languages): 14416** or 1-800-891-4416\n"
    "- **Emergency: 112**\n\n"
    "If you have already swallowed a pesticide or any poison, call **108** for an ambulance or go to the nearest "
    "hospital immediately, and take the pesticide container with you.\n\n"
    "Crop losses and debt are hard, but there is help — your local KVK and agriculture office can also tell you "
    "about crop insurance (PMFBY) claims and relief schemes. I'm here if you want to talk about the farm too."
)

POISONING_RESPONSE = (
    "**This is a medical emergency — get help first.**\n\n"
    "1. Call **108** (ambulance) or **112** now, or go to the nearest hospital / PHC.\n"
    "2. Take the pesticide container or label with you — doctors need the chemical name.\n"
    "3. Move the person away from the spray area into fresh air.\n"
    "4. If it's on the skin or clothes: remove contaminated clothing and wash the skin with plenty of soap and water.\n"
    "5. If it's in the eyes: rinse with clean running water for at least 15 minutes.\n"
    "6. **Do not** make the person vomit and do not give milk, oil or home remedies unless a doctor tells you to.\n\n"
    "I can't give medical treatment advice — please follow the doctor's instructions."
)

_CRISIS_PATTERNS = [
    r"\b(kill|hang|end|harm|hurt)\s+(myself|my\s*self|my\s+life)\b",
    r"\b(suicide|suicidal)\b",
    r"\bwant\s+to\s+die\b",
    r"\b(don'?t|do\s+not)\s+want\s+to\s+(live|be\s+alive)\b",
    r"\bno\s+(reason|point)\s+(to|in)\s+(live|living)\b",
    r"\b(drink|drinking|swallow|swallowing|take|taking|consume)\s+(the\s+)?(pesticide|poison|insecticide|weedicide|herbicide)\s+(myself|to\s+die|and\s+die)\b",
    r"\bhow\s+much\s+(pesticide|poison|insecticide)\b.*\b(die|kill\s+(a\s+)?(person|human|me|myself))\b",
]

_POISONING_PATTERNS = [
    r"\b(drank|drunk|swallowed|ingested|consumed|ate)\b.*\b(pesticide|insecticide|herbicide|weedicide|fungicide|poison|chemical)\b",
    r"\b(pesticide|insecticide|herbicide|weedicide|fungicide)\b.*\b(drank|swallowed|ingested|in\s+(my|his|her|the)\s+(mouth|eyes?))\b",
    r"\b(pesticide|spray|chemical)\s+poisoning\b",
    r"\b(fainted|vomiting|dizzy|unconscious|breathing\s+problem|can'?t\s+breathe)\b.*\b(after|while)\s+(spraying|spray)\b",
    r"\b(after|while)\s+spraying\b.*\b(fainted|vomiting|dizzy|unconscious|can'?t\s+breathe)\b",
]

# Banned for use in India under the Insecticides Act, 1968. Sources:
# - Ministry of Agriculture gazette order of 8 Aug 2018 (18 pesticides; 12
#   banned at once, 6 phased out by 31 Dec 2020);
# - earlier bans incl. endosulfan (Supreme Court, 2011) and legacy organochlorines.
# Not exhaustive and laws change — the warning always points to CIB&RC.
BANNED_PESTICIDES = {
    "endosulfan", "methyl parathion", "phorate", "phosphamidon", "triazophos", "trichlorfon",
    "dichlorvos", "alachlor", "benomyl", "carbaryl", "diazinon", "fenarimol", "fenthion",
    "linuron", "thiometon", "tridemorph", "trifluralin", "sodium cyanide",
    "methoxy ethyl mercury chloride", "aldrin", "chlordane", "endrin", "heptachlor",
    "toxaphene", "pentachlorophenol",
}

BANNED_WARNING = (
    "⚠️ **Legal warning:** {names} {verb} banned for agricultural use in India under the Insecticides Act, 1968. "
    "Using or selling {pronoun} is illegal and dangerous. Ask your KVK or agriculture officer for an approved "
    "alternative, and check the current list on the CIB&RC website (ppqs.gov.in)."
)


@dataclass
class ScreenResult:
    category: str = "ok"                          # ok | crisis | poisoning | banned_pesticide
    response: Optional[str] = None                # set when the agent must NOT run
    appendix: Optional[str] = None                # appended to the agent's answer
    matched: list[str] = field(default_factory=list)

    @property
    def blocks_agent(self) -> bool:
        return self.response is not None


def _any(patterns: list[str], text: str) -> Optional[str]:
    for p in patterns:
        if re.search(p, text):
            return p
    return None


def screen(message: str) -> ScreenResult:
    text = (message or "").lower()

    if hit := _any(_CRISIS_PATTERNS, text):
        return ScreenResult("crisis", response=CRISIS_RESPONSE, matched=[hit])
    if hit := _any(_POISONING_PATTERNS, text):
        return ScreenResult("poisoning", response=POISONING_RESPONSE, matched=[hit])

    found = sorted(p for p in BANNED_PESTICIDES if re.search(rf"\b{re.escape(p)}\b", text))
    if found:
        plural = len(found) > 1
        warning = BANNED_WARNING.format(
            names=", ".join(n.title() for n in found),
            verb="are" if plural else "is",
            pronoun="them" if plural else "it",
        )
        return ScreenResult("banned_pesticide", appendix=warning, matched=found)

    return ScreenResult()
