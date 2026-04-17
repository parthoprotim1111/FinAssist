from dialogue.state_machine import DialogueEngine
from llm.mock_backend import MockLLMBackend


def test_extraction_prompt_renders():
    eng = DialogueEngine(MockLLMBackend())
    p = eng.render_extraction_prompt("hello", "Respond in clear, professional English.")
    assert "define_task" in p or "phase" in p.lower()
    assert "hello" in p
