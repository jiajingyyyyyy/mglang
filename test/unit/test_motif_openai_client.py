import unittest
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "python" / "sglang" / "motif_openai_client.py"
)
_SPEC = importlib.util.spec_from_file_location("motif_openai_client_under_test", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load module under test from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

MotifSGLangConfig = _MODULE.MotifSGLangConfig
MotifSGLangOpenAIClient = _MODULE.MotifSGLangOpenAIClient
default_base_url = _MODULE.default_base_url


class TestMotifSGLangConfig(unittest.TestCase):
    def test_base_url_defaults_to_local_openai_endpoint(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(default_base_url(), "http://127.0.0.1:30003/v1")

    def test_base_url_normalizes_missing_v1_suffix(self):
        config = MotifSGLangConfig(base_url="http://localhost:30003", model="qwen3-14b")

        self.assertEqual(config.base_url, "http://localhost:30003/v1")

    def test_litellm_kwargs_use_openai_compatible_route(self):
        config = MotifSGLangConfig(
            base_url="http://localhost:30003/v1",
            api_key="EMPTY",
            model="qwen3-14b",
            max_tokens=256,
        )

        kwargs = config.litellm_kwargs([{"role": "user", "content": "hi"}])

        self.assertEqual(kwargs["model"], "openai/qwen3-14b")
        self.assertEqual(kwargs["api_base"], "http://localhost:30003/v1")
        self.assertEqual(kwargs["api_key"], "EMPTY")
        self.assertEqual(kwargs["max_tokens"], 256)
        self.assertEqual(kwargs["num_retries"], 0)
        self.assertEqual(
            kwargs["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )


class TestMotifSGLangOpenAIClient(unittest.TestCase):
    def test_complete_text_returns_first_message_content(self):
        client = MotifSGLangOpenAIClient(
            base_url="http://localhost:30003/v1",
            api_key="EMPTY",
            model="qwen3-14b",
        )
        client._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="hello from local sglang")
                            )
                        ]
                    )
                )
            )
        )

        text = client.complete_text([{"role": "user", "content": "hello"}])

        self.assertEqual(text, "hello from local sglang")

    def test_chat_passes_qwen_thinking_option_to_openai_sdk(self):
        seen = {}

        def fake_create(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(choices=[])

        client = MotifSGLangOpenAIClient(
            base_url="http://localhost:30003/v1",
            api_key="EMPTY",
            model="qwen3-14b",
        )
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )

        client.chat(
            [{"role": "user", "content": "hello"}],
            extra_body={"metadata": {"caller": "motif_agent"}},
        )

        self.assertEqual(seen["model"], "qwen3-14b")
        self.assertFalse(seen["stream"])
        self.assertEqual(
            seen["extra_body"],
            {
                "chat_template_kwargs": {"enable_thinking": False},
                "metadata": {"caller": "motif_agent"},
            },
        )


if __name__ == "__main__":
    unittest.main()
