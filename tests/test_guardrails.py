import pytest

from src.agent.guardrails import CRISIS_RESPONSE, POISONING_RESPONSE, screen


@pytest.mark.parametrize("msg", [
    "I want to die, the crop failed again",
    "thinking of suicide because of the loan",
    "I will drink pesticide to die",
    "I don't want to live anymore",
])
def test_crisis_messages_get_helplines_without_the_agent(msg):
    result = screen(msg)
    assert result.category == "crisis" and result.blocks_agent
    assert result.response == CRISIS_RESPONSE and "14416" in result.response


@pytest.mark.parametrize("msg", [
    "my son swallowed some insecticide what do I do",
    "worker fainted after spraying chlorpyrifos",
    "pesticide got in my eyes",
])
def test_poisoning_messages_get_first_aid_without_the_agent(msg):
    result = screen(msg)
    assert result.category == "poisoning" and result.blocks_agent
    assert result.response == POISONING_RESPONSE


def test_banned_pesticide_gets_a_warning_but_still_reaches_the_agent():
    result = screen("Can I use endosulfan and phorate on cotton?")
    assert result.category == "banned_pesticide" and not result.blocks_agent
    assert "Endosulfan" in result.appendix and "Phorate" in result.appendix and "are banned" in result.appendix


@pytest.mark.parametrize("msg", [
    "best pesticide for stem borer in paddy",
    "how to kill aphids myself without chemicals",
    "my plants are dying, leaves turning yellow",
    "what is the PM-KISAN installment amount",
])
def test_ordinary_farming_questions_pass_through(msg):
    result = screen(msg)
    assert result.category == "ok" and not result.blocks_agent and result.appendix is None
