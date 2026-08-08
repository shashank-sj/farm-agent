"""
Structured output schema for Farm Agent responses.
Every chat answer is reshaped into this schema before being shown to the user,
so results look the same (topic, summary, options, precautions, sources)
whether the answer came from RAG, vision, web search, or the yield tool.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    name: str = Field(description="Name of the product, method, or option being recommended")
    how_to_use: str = Field(description="How/when to apply or use it")
    where_to_buy: Optional[str] = Field(
        default=None, description="Where a farmer can buy it, e.g. local Krishi Kendra, eNAM, mandi"
    )
    pros: List[str] = Field(default_factory=list, description="Benefits of this option")
    cons: List[str] = Field(
        default_factory=list, description="Risks, downsides, or effects of overuse/misuse"
    )


class FarmResponse(BaseModel):
    topic: Optional[str] = Field(
        default=None,
        description=(
            "Short title summarising what the answer is about, e.g. 'Wheat Fertilizer Recommendation'. "
            "Leave null for greetings, small talk, or when the answer is itself a clarifying question."
        ),
    )
    crop: Optional[str] = Field(default=None, description="Crop the answer relates to, if any")
    summary: str = Field(description="A concise 1-3 sentence direct answer to the farmer's question")
    recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description="Concrete options/products/steps being recommended. Leave empty if the question isn't about a product/method choice.",
    )
    precautions: List[str] = Field(
        default_factory=list, description="Warnings, cautions, or things to avoid"
    )
    sources: List[str] = Field(
        default_factory=list, description="Where this information came from, e.g. knowledge base, web search, KVK"
    )


def render_farm_response(r: FarmResponse) -> str:
    """Render a FarmResponse as a Markdown card for the Gradio chatbot.

    Greetings/clarifying questions (no topic) render as plain text — the card
    header and chrome are reserved for answers that actually deliver information.
    """
    if not r.topic:
        return r.summary.strip()

    lines = [f"### 🌱 {r.topic}"]
    if r.crop:
        lines.append(f"**Crop:** {r.crop}")
    lines.append("")
    lines.append(r.summary)

    if r.recommendations:
        lines.append("\n**Options**\n")
        for i, item in enumerate(r.recommendations, 1):
            lines.append(f"**{i}. {item.name}**")
            lines.append(f"- 🛠️ How to use: {item.how_to_use}")
            if item.where_to_buy:
                lines.append(f"- 🛒 Where to buy: {item.where_to_buy}")
            if item.pros:
                lines.append(f"- ✅ Pros: {', '.join(item.pros)}")
            if item.cons:
                lines.append(f"- ⚠️ Cons / overuse risk: {', '.join(item.cons)}")
            lines.append("")

    if r.precautions:
        lines.append("**Precautions**")
        for p in r.precautions:
            lines.append(f"- ⚠️ {p}")
        lines.append("")

    if r.sources:
        lines.append(f"*Source: {', '.join(r.sources)}*")

    return "\n".join(lines).strip()
