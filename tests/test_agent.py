import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import FINAL_ANSWER_NUDGE, FarmAgent, history_to_messages
from src.agent.schemas import FarmResponse, render_farm_response


class FakeStructurer:
    def invoke(self, messages):
        return FarmResponse(topic="Wheat yield", crop="Wheat", summary="structured answer")


class FakeLLM:
    """Returns scripted responses in order; records what it was called with."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return FakeStructurer()

    def invoke(self, messages):
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeFallback:
    def answer(self, question, context=""):
        return "fallback answer"


def make_agent(tmp_path, responses, **kwargs):
    return FarmAgent(llm=FakeLLM(responses), fallback_llm=kwargs.pop("fallback_llm", None),
                     rag_index_path=str(tmp_path / "no-index"), yield_model_dir=str(tmp_path / "no-model"),
                     **kwargs)


def yield_call():
    return AIMessage(content="", tool_calls=[{
        "name": "yield_prediction", "id": "call_1",
        "args": {"crop_type": "wheat", "farm_area_acres": 2, "season": "Rabi"},
    }])


def test_small_talk_needs_no_tool_and_no_structuring(tmp_path):
    result = make_agent(tmp_path, [AIMessage(content="Hello! Ask me about your crops.")]).run("hi")
    assert result.answer == "Hello! Ask me about your crops."
    assert result.tools_used == [] and result.steps == 1


def test_tool_loop_runs_the_tool_and_structures_the_answer(tmp_path):
    agent = make_agent(tmp_path, [yield_call(), AIMessage(content="About 2 t/acre.")])
    result = agent.run("expected wheat yield on 2 acres in rabi?")
    assert result.tools_used == ["yield_prediction"]
    assert result.answer == render_farm_response(FakeStructurer().invoke([]))
    tool_output = agent.llm.calls[1][-1]
    assert tool_output.type == "tool" and "rule-based" in tool_output.content


def test_last_step_forces_an_answer_without_tools(tmp_path):
    agent = make_agent(tmp_path, [yield_call(), AIMessage(content="final")], max_steps=2)
    agent.run("yield?")
    assert agent.llm.calls[-1][-1].content == FINAL_ANSWER_NUDGE


def test_crisis_message_never_reaches_the_llm(tmp_path):
    agent = make_agent(tmp_path, [])
    result = agent.run("I want to die")
    assert result.guardrail == "crisis" and "14416" in result.answer
    assert agent.llm.calls == []


def test_banned_pesticide_warning_is_appended(tmp_path):
    result = make_agent(tmp_path, [AIMessage(content="Please use an approved product.")]).run("is endosulfan ok?")
    assert result.answer.startswith("Please use an approved product.")
    assert "Endosulfan is banned" in result.answer


def test_llm_failure_uses_the_fallback_model(tmp_path):
    agent = make_agent(tmp_path, [RuntimeError("429 rate limit")], fallback_llm=FakeFallback())
    result = agent.run("how to control aphids?")
    assert result.fallback_used and result.answer.endswith("fallback answer")


def test_llm_failure_without_fallback_raises(tmp_path):
    with pytest.raises(RuntimeError):
        make_agent(tmp_path, [RuntimeError("down")]).run("how to control aphids?")


def test_history_window_keeps_only_recent_turns():
    history = []
    for i in range(10):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    messages = history_to_messages(history, max_turns=3)
    assert [m.content for m in messages] == ["q7", "a7", "q8", "a8", "q9", "a9"]
    assert isinstance(messages[0], HumanMessage)
    assert history_to_messages([("hi", "hello")], max_turns=3)[1].content == "hello"
