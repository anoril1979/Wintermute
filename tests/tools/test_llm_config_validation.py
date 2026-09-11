"""Tests for the llm.yaml schema validation (config_loader)."""

from __future__ import annotations

import unittest

from src.tools.config_loader import LLMConfigError, validate_llm_config


def _valid_config() -> dict:
    return {
        "models": {
            "router": {
                "model_name": "llama3:8b",
                "temperature": 0.0,
                "max_response_tokens": 256,
                "timeout_seconds": 15,
                "top_p": 0.95,
                "context_window": 8192,
                "keep_alive": "5m",
                "thinking": False,
                "max_retries": 2,
                "description": "routing",
            },
            "default": {"model_name": "llama3:8b"},
        },
        "defaults": {"active_router": "router", "active_default": "default"},
    }


def _assert_error(testcase, config, *fragments):
    with testcase.assertRaises(LLMConfigError) as ctx:
        validate_llm_config(config)
    message = str(ctx.exception)
    for fragment in fragments:
        testcase.assertIn(fragment, message)


class ValidLLMConfigTest(unittest.TestCase):
    def test_valid_config_passes_and_is_returned(self):
        config = _valid_config()
        self.assertIs(validate_llm_config(config), config)

    def test_minimal_role_only_needs_model_name(self):
        config = {"models": {"default": {"model_name": "llama3:8b"}}}
        self.assertEqual(validate_llm_config(config), config)

    def test_defaults_section_is_optional(self):
        config = {"models": {"default": {"model_name": "qwen3"}}}
        self.assertEqual(validate_llm_config(config), config)

    def test_temperature_two_is_accepted(self):
        config = _valid_config()
        config["models"]["default"]["temperature"] = 2.0
        self.assertEqual(validate_llm_config(config), config)

    def test_unknown_keys_are_allowed(self):
        config = _valid_config()
        config["models"]["router"]["repeat_penalty"] = 1.1
        self.assertEqual(validate_llm_config(config), config)


class ModelsSectionTest(unittest.TestCase):
    def test_non_mapping_document(self):
        _assert_error(self, ["models"], "mapping")

    def test_missing_models_key(self):
        _assert_error(self, {"defaults": {}}, "'models'")

    def test_models_wrong_type(self):
        _assert_error(self, {"models": []}, "'models'", "list")

    def test_models_empty(self):
        _assert_error(self, {"models": {}}, "'models'", "vide")

    def test_invalid_role_name(self):
        _assert_error(self, {"models": {42: {"model_name": "x"}}}, "rôle", "42")

    def test_role_entry_not_a_mapping(self):
        _assert_error(self, {"models": {"router": "llama3:8b"}}, "'models.router'", "str")

    def test_missing_model_name(self):
        _assert_error(self, {"models": {"router": {"temperature": 0.0}}}, "'model_name'")

    def test_model_name_wrong_type(self):
        _assert_error(self, {"models": {"router": {"model_name": 42}}}, "'models.router.model_name'", "int")

    def test_model_name_empty(self):
        _assert_error(self, {"models": {"router": {"model_name": "  "}}}, "'models.router.model_name'")


class RoleOptionValidationTest(unittest.TestCase):
    def test_temperature_wrong_type(self):
        config = _valid_config()
        config["models"]["router"]["temperature"] = "cold"
        _assert_error(self, config, "'models.router.temperature'")

    def test_temperature_out_of_range(self):
        config = _valid_config()
        config["models"]["router"]["temperature"] = 3.0
        _assert_error(self, config, "'models.router.temperature'", "entre 0.0 et 2.0")

    def test_top_p_out_of_range(self):
        config = _valid_config()
        config["models"]["router"]["top_p"] = 1.5
        _assert_error(self, config, "'models.router.top_p'")

    def test_max_response_tokens_not_positive(self):
        config = _valid_config()
        config["models"]["router"]["max_response_tokens"] = 0
        _assert_error(self, config, "'models.router.max_response_tokens'", "strictement positif")

    def test_max_response_tokens_wrong_type(self):
        config = _valid_config()
        config["models"]["router"]["max_response_tokens"] = "many"
        _assert_error(self, config, "'models.router.max_response_tokens'")

    def test_context_window_not_positive(self):
        config = _valid_config()
        config["models"]["router"]["context_window"] = -1
        _assert_error(self, config, "'models.router.context_window'", "strictement positif")

    def test_max_token_renamed_key_rejected_with_hint(self):
        config = _valid_config()
        config["models"]["router"]["max_token"] = 8192
        _assert_error(self, config, "'models.router.max_token'",
                      "renommé", "'context_window'")

    def test_max_tokens_renamed_key_rejected_with_hint(self):
        config = _valid_config()
        config["models"]["router"]["max_tokens"] = 256
        _assert_error(self, config, "'models.router.max_tokens'",
                      "renommé", "'max_response_tokens'")

    def test_timeout_seconds_not_positive(self):
        config = _valid_config()
        config["models"]["router"]["timeout_seconds"] = -1
        _assert_error(self, config, "'models.router.timeout_seconds'")

    def test_max_retries_negative(self):
        config = _valid_config()
        config["models"]["router"]["max_retries"] = -1
        _assert_error(self, config, "'models.router.max_retries'", "entier >= 0")

    def test_max_retries_bool_rejected(self):
        config = _valid_config()
        config["models"]["router"]["max_retries"] = True
        _assert_error(self, config, "'models.router.max_retries'")

    def test_keep_alive_wrong_type(self):
        config = _valid_config()
        config["models"]["router"]["keep_alive"] = 5
        _assert_error(self, config, "'models.router.keep_alive'")

    def test_thinking_wrong_type(self):
        config = _valid_config()
        config["models"]["router"]["thinking"] = "yes"
        _assert_error(self, config, "'models.router.thinking'", "booléen")

    def test_description_wrong_type(self):
        config = _valid_config()
        config["models"]["router"]["description"] = 123
        _assert_error(self, config, "'models.router.description'")


class DefaultsConsistencyTest(unittest.TestCase):
    def test_defaults_not_a_mapping(self):
        config = _valid_config()
        config["defaults"] = ["router"]
        _assert_error(self, config, "'defaults'", "mapping")

    def test_defaults_referencing_unknown_role(self):
        config = _valid_config()
        config["defaults"] = {"active_router": "routr"}
        _assert_error(self, config, "'defaults.active_router'", "'routr'")

    def test_defaults_value_wrong_type(self):
        config = _valid_config()
        config["defaults"] = {"active_router": 42}
        _assert_error(self, config, "'defaults.active_router'")


if __name__ == "__main__":
    unittest.main()