"""Tests for the ingestion.yaml schema validation (config_loader)."""

from __future__ import annotations

import unittest
import unittest.mock

from src.tools.config_loader import (
    DEFAULT_ORIGINS,
    DocumentsConfigError,
    IngestionConfigError,
    LoggingConfigError,
    RetrievalConfigError,
    coerce_origin,
    get_default_origin,
    get_valid_origins,
    load_retrieval_config,
    validate_documents_config,
    validate_ingestion_config,
    validate_logging_config,
    validate_retrieval_config,
)


def _valid_config() -> dict:
    return {
        "documents_root": "data/sources",
        "extensions": {".pdf": "pdf", ".md": "text"},
        "extraction_output_dir": "data/extracted",
        "mineru_json_extension": "_content_list.json",
    }


class ValidConfigTest(unittest.TestCase):
    def test_valid_config_passes_and_is_returned(self):
        config = _valid_config()
        self.assertIs(validate_ingestion_config(config), config)

    def test_optional_keys_may_be_omitted(self):
        config = {"documents_root": "data/sources", "extensions": {".pdf": "pdf"}}
        self.assertEqual(validate_ingestion_config(config), config)

    def test_several_extensions_may_share_one_folder(self):
        config = _valid_config()
        config["extensions"] = {".txt": "text", ".md": "text", ".markdown": "text"}
        self.assertEqual(validate_ingestion_config(config), config)

    def test_absolute_paths_are_accepted(self):
        config = _valid_config()
        config["documents_root"] = "C:/absolute/documents"
        config["extraction_output_dir"] = "D:/absolute/out"
        self.assertEqual(validate_ingestion_config(config), config)


class DocumentsRootValidationTest(unittest.TestCase):
    def test_missing_key(self):
        config = _valid_config()
        del config["documents_root"]
        with self.assertRaises(IngestionConfigError) as ctx:
            validate_ingestion_config(config)
        self.assertIn("'documents_root'", str(ctx.exception))

    def test_wrong_type(self):
        config = _valid_config()
        config["documents_root"] = 42
        with self.assertRaises(IngestionConfigError) as ctx:
            validate_ingestion_config(config)
        self.assertIn("'documents_root'", str(ctx.exception))
        self.assertIn("int", str(ctx.exception))

    def test_empty_value(self):
        config = _valid_config()
        config["documents_root"] = "   "
        with self.assertRaises(IngestionConfigError):
            validate_ingestion_config(config)

    def test_traversal_rejected(self):
        config = _valid_config()
        config["documents_root"] = "../outside"
        with self.assertRaises(IngestionConfigError) as ctx:
            validate_ingestion_config(config)
        self.assertIn("'..'", str(ctx.exception))


class ExtensionsValidationTest(unittest.TestCase):
    def _assert_extension_error(self, config, *fragments):
        with self.assertRaises(IngestionConfigError) as ctx:
            validate_ingestion_config(config)
        message = str(ctx.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)

    def test_missing_key(self):
        config = _valid_config()
        del config["extensions"]
        self._assert_extension_error(config, "'extensions'")

    def test_wrong_type(self):
        config = _valid_config()
        config["extensions"] = [".pdf"]
        self._assert_extension_error(config, "'extensions'", "list")

    def test_empty_mapping(self):
        config = _valid_config()
        config["extensions"] = {}
        self._assert_extension_error(config, "'extensions'", "vide")

    def test_non_string_key_is_named(self):
        config = _valid_config()
        config["extensions"] = {".pdf": "pdf", 42: "forty_two"}
        self._assert_extension_error(config, "'extensions'", "42")

    def test_key_without_leading_dot(self):
        config = _valid_config()
        config["extensions"] = {"pdf": "pdf"}
        self._assert_extension_error(config, "'pdf'")

    def test_uppercase_key(self):
        config = _valid_config()
        config["extensions"] = {".PDF": "pdf"}
        self._assert_extension_error(config, "'.PDF'")

    def test_folder_value_wrong_type(self):
        config = _valid_config()
        config["extensions"] = {".pdf": 3}
        self._assert_extension_error(config, "'.pdf'", "int")

    def test_folder_value_empty(self):
        config = _valid_config()
        config["extensions"] = {".pdf": "  "}
        self._assert_extension_error(config, "'.pdf'")

    def test_folder_value_with_path_separators(self):
        config = _valid_config()
        config["extensions"] = {".pdf": "sub/pdf"}
        self._assert_extension_error(config, "'.pdf'", "sous-dossier")

    def test_folder_value_traversal(self):
        config = _valid_config()
        config["extensions"] = {".pdf": ".."}
        self._assert_extension_error(config, "'.pdf'")


class ExtractionSettingsValidationTest(unittest.TestCase):
    def test_wrong_type_is_named(self):
        config = _valid_config()
        config["extraction_output_dir"] = 7
        with self.assertRaises(IngestionConfigError) as ctx:
            validate_ingestion_config(config)
        self.assertIn("'extraction_output_dir'", str(ctx.exception))

    def test_empty_value(self):
        config = _valid_config()
        config["mineru_json_extension"] = ""
        with self.assertRaises(IngestionConfigError):
            validate_ingestion_config(config)

    def test_output_dir_traversal_rejected(self):
        config = _valid_config()
        config["extraction_output_dir"] = "data/../elsewhere"
        with self.assertRaises(IngestionConfigError):
            validate_ingestion_config(config)

    def test_json_extension_with_path_separator(self):
        config = _valid_config()
        config["mineru_json_extension"] = "sub/out.json"
        with self.assertRaises(IngestionConfigError) as ctx:
            validate_ingestion_config(config)
        self.assertIn("'mineru_json_extension'", str(ctx.exception))

    def test_json_extension_without_expected_prefix(self):
        config = _valid_config()
        config["mineru_json_extension"] = "content_list"
        with self.assertRaises(IngestionConfigError) as ctx:
            validate_ingestion_config(config)
        self.assertIn("'.' ou '_'", str(ctx.exception))

    def test_valid_dot_and_underscore_prefixes(self):
        config = _valid_config()
        config["mineru_json_extension"] = ".content_list.json"
        self.assertEqual(validate_ingestion_config(config), config)


class IngestionLogValidationTest(unittest.TestCase):
    """The ingestion CLI's durable log path (scripts/ingest.py)."""

    def test_valid_path_passes(self):
        config = _valid_config()
        config["ingestion_log"] = "data/logs/ingestion.log"
        self.assertIs(validate_ingestion_config(config), config)

    def test_absolute_path_passes(self):
        config = _valid_config()
        config["ingestion_log"] = "D:/logs/ingestion.log"
        self.assertIs(validate_ingestion_config(config), config)

    def test_absent_key_is_ok(self):
        config = _valid_config()
        self.assertIs(validate_ingestion_config(config), config)

    def test_wrong_type_is_named(self):
        config = _valid_config()
        config["ingestion_log"] = 7
        with self.assertRaises(IngestionConfigError) as ctx:
            validate_ingestion_config(config)
        self.assertIn("'ingestion_log'", str(ctx.exception))

    def test_empty_value(self):
        config = _valid_config()
        config["ingestion_log"] = "   "
        with self.assertRaises(IngestionConfigError) as ctx:
            validate_ingestion_config(config)
        self.assertIn("'ingestion_log'", str(ctx.exception))

    def test_traversal_rejected(self):
        config = _valid_config()
        config["ingestion_log"] = "data/../elsewhere/ingestion.log"
        with self.assertRaises(IngestionConfigError) as ctx:
            validate_ingestion_config(config)
        self.assertIn("'ingestion_log'", str(ctx.exception))


class NonMappingDocumentTest(unittest.TestCase):
    def test_list_is_rejected(self):
        with self.assertRaises(IngestionConfigError) as ctx:
            validate_ingestion_config(["documents_root", "extensions"])
        self.assertIn("mapping", str(ctx.exception))


# ---------------------------------------------------------------------------
# retrieval.yaml validation
# ---------------------------------------------------------------------------

class RetrievalConfigValidationTest(unittest.TestCase):
    def _valid_config(self) -> dict:
        return {
            "default_top_k": 6,
            "max_top_k": 20,
            "min_score": 0.35,
            "embedding_role": "embedding",
            "source_collection_key": "source_chunks",
        }

    def test_valid_config_roundtrip(self):
        config = self._valid_config()
        self.assertIs(validate_retrieval_config(config), config)

    def test_non_mapping_document_is_rejected(self):
        with self.assertRaises(RetrievalConfigError):
            validate_retrieval_config(["default_top_k"])

    def test_missing_required_key_is_rejected(self):
        for key in ("default_top_k", "max_top_k", "min_score",
                    "embedding_role", "source_collection_key"):
            config = self._valid_config()
            del config[key]
            with self.assertRaises(RetrievalConfigError, msg=key):
                validate_retrieval_config(config)

    def test_non_positive_top_k_is_rejected(self):
        config = self._valid_config()
        config["default_top_k"] = 0
        with self.assertRaises(RetrievalConfigError) as ctx:
            validate_retrieval_config(config)
        self.assertIn("'default_top_k'", str(ctx.exception))

    def test_bool_top_k_is_rejected(self):
        config = self._valid_config()
        config["max_top_k"] = True
        with self.assertRaises(RetrievalConfigError):
            validate_retrieval_config(config)

    def test_default_top_k_above_max_is_rejected(self):
        config = self._valid_config()
        config["default_top_k"] = 50
        with self.assertRaises(RetrievalConfigError) as ctx:
            validate_retrieval_config(config)
        self.assertIn("max_top_k", str(ctx.exception))

    def test_min_score_out_of_range_is_rejected(self):
        config = self._valid_config()
        config["min_score"] = 1.5
        with self.assertRaises(RetrievalConfigError):
            validate_retrieval_config(config)
        config["min_score"] = "high"
        with self.assertRaises(RetrievalConfigError):
            validate_retrieval_config(config)

    def test_unknown_embedding_role_is_rejected(self):
        """Cross-file consistency: the role must exist in llm.yaml."""
        config = self._valid_config()
        config["embedding_role"] = "no_such_role"
        with self.assertRaises(RetrievalConfigError) as ctx:
            validate_retrieval_config(config)
        self.assertIn("llm.yaml", str(ctx.exception))

    def test_query_instruction_must_be_a_string(self):
        config = self._valid_config()
        config["query_instruction"] = 42
        with self.assertRaises(RetrievalConfigError) as ctx:
            validate_retrieval_config(config)
        self.assertIn("query_instruction", str(ctx.exception))

    def test_query_instruction_accepts_empty_string(self):
        """Empty = disabled: valid, the agent embeds the raw question."""
        config = self._valid_config()
        config["query_instruction"] = ""
        self.assertIs(validate_retrieval_config(config), config)

    def test_query_instruction_is_optional(self):
        config = self._valid_config()
        self.assertIs(validate_retrieval_config(config), config)

    def test_real_retrieval_yaml_loads(self):
        """The shipped retrieval.yaml must pass its own validation."""
        config = load_retrieval_config()
        self.assertGreater(config["default_top_k"], 0)
        self.assertGreaterEqual(config["max_top_k"], config["default_top_k"])
        self.assertIsInstance(config.get("query_instruction", ""), str)
        self.assertTrue(config.get("query_instruction", "").startswith("Instruct:"))


class DocumentsSectionValidationTest(unittest.TestCase):
    """setup.yaml's ``documents`` section — the user-defined origin
    vocabulary (governance labels for ingested documents)."""

    @staticmethod
    def _section() -> dict:
        return {"documents": {"origins": ["canon", "community", "rpg"]}}

    def test_valid_section_passes_and_is_returned(self):
        config = self._section()
        self.assertIs(validate_documents_config(config), config)

    def test_section_may_be_absent(self):
        config: dict = {}
        self.assertIs(validate_documents_config(config), config)

    def test_origins_key_may_be_absent(self):
        config = {"documents": {}}
        self.assertIs(validate_documents_config(config), config)

    def test_non_mapping_section_is_rejected(self):
        with self.assertRaises(DocumentsConfigError) as ctx:
            validate_documents_config({"documents": ["canon"]})
        self.assertIn("'documents'", str(ctx.exception))

    def test_empty_origins_list_is_rejected(self):
        config = {"documents": {"origins": []}}
        with self.assertRaises(DocumentsConfigError) as ctx:
            validate_documents_config(config)
        self.assertIn("'documents.origins'", str(ctx.exception))

    def test_non_list_origins_is_rejected(self):
        config = {"documents": {"origins": "canon"}}
        with self.assertRaises(DocumentsConfigError) as ctx:
            validate_documents_config(config)
        self.assertIn("'documents.origins'", str(ctx.exception))

    def test_non_string_entry_is_named(self):
        config = {"documents": {"origins": ["canon", 7]}}
        with self.assertRaises(DocumentsConfigError) as ctx:
            validate_documents_config(config)
        self.assertIn("'documents.origins[1]'", str(ctx.exception))

    def test_blank_entry_is_rejected(self):
        config = {"documents": {"origins": ["  "]}}
        with self.assertRaises(DocumentsConfigError) as ctx:
            validate_documents_config(config)
        self.assertIn("'documents.origins[0]'", str(ctx.exception))

    def test_duplicate_after_normalization_is_rejected(self):
        config = {"documents": {"origins": ["Canon", "canon"]}}
        with self.assertRaises(DocumentsConfigError) as ctx:
            validate_documents_config(config)
        self.assertIn("doublon", str(ctx.exception))


class OriginVocabularyTest(unittest.TestCase):
    """The configured vocabulary and its accessors (live setup.yaml).

    Deliberately vocabulary-agnostic: the user defines their own origin
    labels (the shipped default is canon/community/rpg, but any list is
    valid), so the tests derive their expectations from the config.
    """

    def test_real_setup_yaml_vocabulary(self):
        origins = get_valid_origins()
        self.assertTrue(origins, "the shipped setup.yaml must define origins")
        self.assertEqual(origins, tuple(o.lower() for o in origins))
        self.assertEqual(len(origins), len(set(origins)))
        # First entry = the default origin.
        self.assertEqual(get_default_origin(), origins[0])

    def test_coerce_normalizes_and_validates(self):
        origins = get_valid_origins()
        probe = origins[-1]
        self.assertEqual(coerce_origin(f"  {probe.upper()}  "), probe)
        self.assertEqual(coerce_origin(origins[0].capitalize()), origins[0])
        self.assertIsNone(coerce_origin("definitely-not-a-configured-origin"))
        self.assertIsNone(coerce_origin(7))
        self.assertIsNone(coerce_origin(None))

    def test_broken_yaml_fails_open_to_defaults(self):
        with unittest.mock.patch(
            "src.tools.config_loader._load_documents_config",
            side_effect=RuntimeError("broken yaml"),
        ):
            self.assertEqual(get_valid_origins(), DEFAULT_ORIGINS)
            self.assertEqual(get_default_origin(), DEFAULT_ORIGINS[0])

    def test_custom_user_vocabulary_is_honored(self):
        import src.tools.config_loader as cl

        config = {
            "documents": {
                "origins": ["Official", " fan-work ", "homebrew"]
            }
        }
        cl._load_documents_config.cache_clear()
        try:
            with unittest.mock.patch.object(
                cl, "_load_yaml", return_value=config
            ):
                self.assertEqual(
                    get_valid_origins(),
                    ("official", "fan-work", "homebrew"),
                )
                self.assertEqual(get_default_origin(), "official")
                self.assertEqual(coerce_origin("FAN-WORK"), "fan-work")
                self.assertIsNone(coerce_origin("rpg"))
        finally:
            cl._load_documents_config.cache_clear()


class LoggingSectionValidationTest(unittest.TestCase):
    """setup.yaml's ``logging`` section (main_log, level, format, rotation)."""

    @staticmethod
    def _section() -> dict:
        return {
            "logging": {
                "level": "DEBUG",
                "main_log": "data/logs/wintermute.log",
                "max_bytes": 5242880,
                "backup_count": 3,
            }
        }

    def test_valid_section_passes_and_is_returned(self):
        config = self._section()
        self.assertIs(validate_logging_config(config), config)

    def test_section_may_be_absent(self):
        config: dict = {}
        self.assertIs(validate_logging_config(config), config)

    def test_empty_main_log_disables_file_logging(self):
        config = self._section()
        config["logging"]["main_log"] = ""
        self.assertIs(validate_logging_config(config), config)

    def test_absolute_main_log_passes(self):
        config = self._section()
        config["logging"]["main_log"] = "D:/logs/wintermute.log"
        self.assertIs(validate_logging_config(config), config)

    def test_non_mapping_section_is_rejected(self):
        with self.assertRaises(LoggingConfigError) as ctx:
            validate_logging_config({"logging": ["main_log"]})
        self.assertIn("'logging'", str(ctx.exception))

    def test_wrong_type_main_log_is_named(self):
        config = self._section()
        config["logging"]["main_log"] = 7
        with self.assertRaises(LoggingConfigError) as ctx:
            validate_logging_config(config)
        self.assertIn("'logging.main_log'", str(ctx.exception))

    def test_traversal_main_log_is_rejected(self):
        config = self._section()
        config["logging"]["main_log"] = "data/../elsewhere/w.log"
        with self.assertRaises(LoggingConfigError) as ctx:
            validate_logging_config(config)
        self.assertIn("'logging.main_log'", str(ctx.exception))

    def test_unknown_level_is_rejected(self):
        config = self._section()
        config["logging"]["level"] = "LOUD"
        with self.assertRaises(LoggingConfigError) as ctx:
            validate_logging_config(config)
        self.assertIn("'logging.level'", str(ctx.exception))

    def test_level_is_case_insensitive(self):
        config = self._section()
        config["logging"]["level"] = "info"
        self.assertIs(validate_logging_config(config), config)

    def test_non_string_format_is_rejected(self):
        config = self._section()
        config["logging"]["format"] = 12
        with self.assertRaises(LoggingConfigError) as ctx:
            validate_logging_config(config)
        self.assertIn("'logging.format'", str(ctx.exception))

    def test_format_with_unknown_field_is_rejected(self):
        config = self._section()
        config["logging"]["format"] = "%(nope)s"
        with self.assertRaises(LoggingConfigError) as ctx:
            validate_logging_config(config)
        self.assertIn("'logging.format'", str(ctx.exception))

    def test_bool_max_bytes_is_rejected(self):
        config = self._section()
        config["logging"]["max_bytes"] = True  # a bool IS an int in Python
        with self.assertRaises(LoggingConfigError) as ctx:
            validate_logging_config(config)
        self.assertIn("'logging.max_bytes'", str(ctx.exception))

    def test_non_positive_max_bytes_is_rejected(self):
        config = self._section()
        config["logging"]["max_bytes"] = 0
        with self.assertRaises(LoggingConfigError):
            validate_logging_config(config)

    def test_negative_backup_count_is_rejected(self):
        config = self._section()
        config["logging"]["backup_count"] = -1
        with self.assertRaises(LoggingConfigError):
            validate_logging_config(config)

    def test_zero_backup_count_means_no_rotation(self):
        config = self._section()
        config["logging"]["backup_count"] = 0
        self.assertIs(validate_logging_config(config), config)

    def test_real_setup_yaml_loads(self):
        """The shipped setup.yaml must pass its own validation."""
        from src.tools.config_loader import load_logging_config

        section = load_logging_config()
        self.assertEqual(
            section.get("main_log"), "data/logs/wintermute.log"
        )


if __name__ == "__main__":
    unittest.main()
