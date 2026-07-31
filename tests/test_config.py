"""Tests for environment driven configuration."""
import pytest

from analyst.config import DEFAULT_MODEL, LLMConfig, _int_env, find_api_key, load_config
from analyst.errors import ConfigError

TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_API_KEY")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from an environment with no inference settings."""
    for name in (*TOKEN_VARS, "HF_MODEL", "HF_PROVIDER", "HF_MAX_TOKENS", "HF_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)


class TestFindApiKey:
    def test_returns_none_when_unset(self):
        assert find_api_key() is None

    @pytest.mark.parametrize("var", TOKEN_VARS)
    def test_accepts_each_supported_variable(self, monkeypatch, var):
        monkeypatch.setenv(var, "secret")
        assert find_api_key() == "secret"

    def test_blank_value_is_ignored(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "   ")
        assert find_api_key() is None

    def test_value_is_stripped(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "  secret\n")
        assert find_api_key() == "secret"


class TestLoadConfig:
    def test_defaults(self):
        config = load_config()
        assert config.model == DEFAULT_MODEL
        assert config.provider == "nebius"
        assert config.api_key is None

    def test_overrides_from_env(self, monkeypatch):
        monkeypatch.setenv("HF_MODEL", "other/model")
        monkeypatch.setenv("HF_PROVIDER", "together")
        monkeypatch.setenv("HF_MAX_TOKENS", "128")
        config = load_config()
        assert (config.model, config.provider, config.max_tokens) == (
            "other/model",
            "together",
            128,
        )

    def test_invalid_integer_is_reported_clearly(self, monkeypatch):
        monkeypatch.setenv("HF_MAX_TOKENS", "lots")
        with pytest.raises(ConfigError, match="HF_MAX_TOKENS"):
            load_config()


class TestRequireApiKey:
    def test_returns_the_key(self):
        assert LLMConfig(api_key="abc").require_api_key() == "abc"

    def test_missing_key_names_the_variable_to_set(self):
        with pytest.raises(ConfigError, match="HF_TOKEN"):
            LLMConfig().require_api_key()


class TestIntEnv:
    def test_default_when_unset(self):
        assert _int_env("DEFINITELY_NOT_SET_12345", 7) == 7

    def test_blank_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("SOME_INT", "  ")
        assert _int_env("SOME_INT", 7) == 7


class TestNoHardcodedSecrets:
    def test_config_module_has_no_literal_token(self):
        """Guards against a key being pasted back into the source."""
        import re
        from pathlib import Path

        import analyst

        root = Path(analyst.__file__).resolve().parent
        pattern = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})")
        for path in root.glob("*.py"):
            assert not pattern.search(path.read_text(encoding="utf-8")), path
