"""Tests for the ingestion.yaml schema validation (config_loader)."""

from __future__ import annotations

import unittest

from src.tools.config_loader import (
    IngestionConfigError,
    validate_ingestion_config,
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


class NonMappingDocumentTest(unittest.TestCase):
    def test_list_is_rejected(self):
        with self.assertRaises(IngestionConfigError) as ctx:
            validate_ingestion_config(["documents_root", "extensions"])
        self.assertIn("mapping", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
