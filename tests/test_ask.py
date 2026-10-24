from types import SimpleNamespace
from unittest.mock import Mock

import app.ask as ask_module


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        text="answer",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=7,
        ),
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(return_value=response))
    )
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", Mock(return_value=client))

    assert ask_module.ask("question") == "answer"
    assert capsys.readouterr().out == "[トークン数] 入力: 12, 出力: 7\n"


def test_ask_reports_when_token_usage_is_unavailable(monkeypatch, capsys):
    response = SimpleNamespace(text="answer", usage_metadata=None)
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(return_value=response))
    )
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", Mock(return_value=client))

    assert ask_module.ask("question") == "answer"
    assert capsys.readouterr().out == "[トークン数] 使用量を取得できませんでした\n"
