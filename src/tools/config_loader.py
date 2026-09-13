"""
config_loader.py

Charge et valide la configuration du prototype :
- Variables d'environnement (.env) pour les secrets
- config/setup.yaml pour les paramètres généraux
- config/llm.yaml pour la configuration des modèles Ollama

Ce module est le point d'entrée unique pour accéder à la configuration
depuis le reste du projet (tools/ et engine/).
"""

import os
from pathlib import Path
from functools import lru_cache
# lru_cache rend la config quasi-singleton :
# pratique pour un prototype, mais pour recharger la config à chaud sans redémarrer le service,
# il faudra retirer ce cache ou ajouter une fonction reload().

import yaml
from dotenv import load_dotenv


# ------------------------------------------------------------------
# Résolution des chemins
# ------------------------------------------------------------------

def _find_project_root() -> Path:
    """
    Remonte l'arborescence jusqu'au dossier racine du projet,
    identifié par la présence de config/setup.yaml.

    Indépendant de l'emplacement exact du module (src/tools/, tools/, ...)
    tant qu'il reste sous la racine du projet.
    """
    current = Path(__file__).resolve().parent
    for _ in range(8):  # garde-fou : on ne remonte pas indéfiniment
        if (current / "config" / "setup.yaml").exists():
            return current
        current = current.parent
    raise ConfigError(
        "Impossible de localiser la racine du projet : "
        "config/setup.yaml introuvable en remontant depuis ce module."
    )


# ------------------------------------------------------------------
# Exceptions dédiées
# ------------------------------------------------------------------

class ConfigError(Exception):
    """Erreur levée en cas de configuration manquante ou invalide."""
    pass


class IngestionConfigError(ConfigError):
    """ingestion.yaml is malformed or incomplete.

    Raised at load time by ``validate_ingestion_config``. The message is
    written to be forwarded verbatim to the user (or to the calling LLM)
    so the yaml can be fixed without a debugger.
    """


class LLMConfigError(ConfigError):
    """llm.yaml is malformed or incomplete.

    Raised at load time by ``validate_llm_config``. Messages are written to
    be forwarded verbatim to the user (or to the calling LLM) so the yaml
    can be fixed without a debugger.
    """


class VectorConfigError(ConfigError):
    """setup.yaml's vector_db section is malformed or incomplete.

    Raised at load time by ``validate_vector_config``. Messages are written
    to be fixed by hand in setup.yaml without a debugger.
    """


class RoutingConfigError(ConfigError):
    """setup.yaml's ``routing`` section is malformed or incomplete.

    Raised by validate_routing_config/load_routing_config with a message
    naming the faulty entry, so the yaml can be fixed by hand.
    """


class RetrievalConfigError(ConfigError):
    """retrieval.yaml is malformed or incomplete.

    Raised at load time by ``validate_retrieval_config``. Messages are
    written to be forwarded verbatim to the user (or to the calling LLM)
    so the yaml can be fixed without a debugger.
    """


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
SETUP_YAML_PATH = CONFIG_DIR / "setup.yaml"
LLM_YAML_PATH = CONFIG_DIR / "llm.yaml"
INGESTION_YAML_PATH = CONFIG_DIR / "ingestion.yaml"
RETRIEVAL_YAML_PATH = CONFIG_DIR / "retrieval.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


# ------------------------------------------------------------------
# Chargement des fichiers bruts
# ------------------------------------------------------------------

def _load_env() -> None:
    """Charge le fichier .env à la racine du projet dans os.environ."""
    if not ENV_PATH.exists():
        raise ConfigError(f".env introuvable à l'emplacement attendu : {ENV_PATH}")
    load_dotenv(dotenv_path=ENV_PATH)


def _load_yaml(path: Path) -> dict:
    """Charge un fichier YAML et retourne son contenu sous forme de dict."""
    if not path.exists():
        raise ConfigError(f"Fichier de configuration introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            content = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            # A syntax-broken yaml must surface as a ConfigError too, so
            # callers only ever need to catch ConfigError.
            raise ConfigError(
                f"Fichier de configuration illisible (erreur de syntaxe YAML) : {path}\n{exc}"
            ) from exc
    if content is None:
        raise ConfigError(f"Fichier de configuration vide ou invalide : {path}")
    return content


def _require_env(var_name: str) -> str:
    """Récupère une variable d'environnement obligatoire, lève une erreur claire si absente."""
    value = os.environ.get(var_name)
    if not value or value.startswith("<"):
        raise ConfigError(
            f"Variable d'environnement '{var_name}' manquante ou non renseignée. "
            f"Vérifiez votre fichier .env."
        )
    return value


# ------------------------------------------------------------------
# Chargement et fusion de la configuration setup.yaml + .env
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_setup_config() -> dict:
    """
    Charge setup.yaml, injecte les secrets depuis .env, et retourne
    un dictionnaire de configuration unifié.

    Le résultat est mis en cache (lru_cache) : la configuration n'est
    lue et validée qu'une seule fois par exécution du programme.
    """
    _load_env()
    config = _load_yaml(SETUP_YAML_PATH)

    # --- Injection des secrets email (uniquement si la section existe) ---
    if "email" in config:
        config["email"]["username"] = _require_env("EMAIL_USERNAME")
        config["email"]["password"] = _require_env("EMAIL_PASSWORD")

    # --- Injection des chemins Google Calendar (uniquement si la section existe) ---
    if "calendar" in config:
        config["calendar"]["credentials_path"] = _require_env("GOOGLE_CREDENTIALS_PATH")
        config["calendar"]["token_path"] = _require_env("GOOGLE_TOKEN_PATH")

    return config


def _is_number(value: object) -> bool:
    """True for int/float, excluding bool (a bool is an int in Python)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_llm_config(config: object) -> dict:
    """Validate the full llm.yaml schema; raise LLMConfigError on malformed entries.

    This validates STRUCTURE only (types, required keys, ranges) plus the
    self-referential ``defaults`` section. It deliberately does NOT decide
    which roles must exist: role existence is semantic knowledge that only
    the agent using a role possesses, so it is enforced at the point of use
    (strict lookup, see src/agents/llm_roles.require_llm_role) rather than
    through a hardcoded list here.

    Checks, per role under ``models``:

    * role name is a non-empty string;
    * the entry is a mapping;
    * ``model_name`` is required and a non-empty string;
    * typed optional keys: ``temperature``/``top_p`` (ranges),
      ``max_response_tokens``/``timeout_seconds``/``context_window``
      (positive numbers), ``max_retries`` (non-negative int),
      ``keep_alive``/``description`` (strings), ``thinking`` (bool).
      The former ``max_token``/``max_tokens`` keys were renamed to
      ``context_window``/``max_response_tokens`` and are rejected with a
      rename hint. Unknown keys are left to the consumers.

    And, when present: ``defaults`` must be a mapping whose values are
    strings naming roles that exist under ``models``.

    Returns the same dict on success, so callers can do
    ``config = validate_llm_config(config)``.
    """
    prefix = "llm.yaml invalide"

    if not isinstance(config, dict):
        raise LLMConfigError(
            f"{prefix}: le contenu doit être un mapping YAML "
            f"(type trouvé : {type(config).__name__})."
        )

    # -- models --------------------------------------------------------------
    if "models" not in config:
        raise LLMConfigError(
            f"{prefix}: section requise manquante 'models' "
            "(les rôles de modèles)."
        )
    models = config["models"]
    if not isinstance(models, dict):
        raise LLMConfigError(
            f"{prefix}: 'models' doit être un mapping rôle -> configuration "
            f"(type trouvé : {type(models).__name__})."
        )
    if not models:
        raise LLMConfigError(
            f"{prefix}: 'models' ne doit pas être vide "
            "(au moins un rôle, ex. 'default')."
        )

    _POSITIVE_NUMBERS = {"max_response_tokens", "timeout_seconds", "context_window"}

    # Renamed keys (see the header of config/llm.yaml): rejected loudly
    # instead of silently ignored, so a stale yaml explains itself.
    _RENAMED_KEYS = {
        "max_token": "context_window",
        "max_tokens": "max_response_tokens",
    }

    _RANGED_NUMBERS = {"temperature": (0.0, 2.0), "top_p": (0.0, 1.0)}
    _NON_NEGATIVE_INTS = {"max_retries"}
    _STRINGS = {"keep_alive", "description"}
    _BOOLEANS = {"thinking"}

    for role, entry in models.items():
        if not isinstance(role, str) or not role.strip():
            raise LLMConfigError(
                f"{prefix}: 'models' : nom de rôle invalide {role!r} "
                "(doit être une chaîne non vide)."
            )
        if not isinstance(entry, dict):
            raise LLMConfigError(
                f"{prefix}: 'models.{role}' doit être un mapping de paramètres "
                f"(type trouvé : {type(entry).__name__})."
            )
        if "model_name" not in entry:
            raise LLMConfigError(
                f"{prefix}: 'models.{role}' : clé requise manquante 'model_name' "
                '(nom du modèle Ollama, ex. "llama3:8b").'
            )
        model_name = entry["model_name"]
        if not isinstance(model_name, str) or not model_name.strip():
            raise LLMConfigError(
                f"{prefix}: 'models.{role}.model_name' doit être une chaîne non "
                f"vide (type trouvé : {type(model_name).__name__})."
            )

        for old_key, new_key in _RENAMED_KEYS.items():
            if old_key in entry:
                raise LLMConfigError(
                    f"{prefix} : 'models.{role}.{old_key}' a été renommé "
                    f"en '{new_key}' : remplacez la clé dans llm.yaml."
                )

        for key in _POSITIVE_NUMBERS:
            if key in entry and (not _is_number(entry[key]) or entry[key] <= 0):
                raise LLMConfigError(
                    f"{prefix}: 'models.{role}.{key}' doit être un nombre "
                    f"strictement positif (valeur : {entry[key]!r})."
                )
        for key, (lo, hi) in _RANGED_NUMBERS.items():
            if key in entry and (not _is_number(entry[key]) or not lo <= entry[key] <= hi):
                raise LLMConfigError(
                    f"{prefix}: 'models.{role}.{key}' doit être un nombre entre "
                    f"{lo} et {hi} (valeur : {entry[key]!r})."
                )
        for key in _NON_NEGATIVE_INTS:
            if key in entry and (
                isinstance(entry[key], bool)
                or not isinstance(entry[key], int)
                or entry[key] < 0
            ):
                raise LLMConfigError(
                    f"{prefix}: 'models.{role}.{key}' doit être un entier >= 0 "
                    f"(valeur : {entry[key]!r})."
                )
        for key in _STRINGS:
            if key in entry and not isinstance(entry[key], str):
                raise LLMConfigError(
                    f"{prefix}: 'models.{role}.{key}' doit être une chaîne "
                    f"(type trouvé : {type(entry[key]).__name__})."
                )
        for key in _BOOLEANS:
            if key in entry and not isinstance(entry[key], bool):
                raise LLMConfigError(
                    f"{prefix}: 'models.{role}.{key}' doit être un booléen "
                    f"(type trouvé : {type(entry[key]).__name__})."
                )

    # -- defaults: internal consistency only (values must name existing roles)
    if "defaults" in config:
        defaults = config["defaults"]
        if not isinstance(defaults, dict):
            raise LLMConfigError(
                f"{prefix}: 'defaults' doit être un mapping "
                f"(type trouvé : {type(defaults).__name__})."
            )
        for key, value in defaults.items():
            if not isinstance(value, str):
                raise LLMConfigError(
                    f"{prefix}: 'defaults.{key}' doit être le nom d'un rôle "
                    f"(chaîne, type trouvé : {type(value).__name__})."
                )
            if value not in models:
                raise LLMConfigError(
                    f"{prefix}: 'defaults.{key}' fait référence au rôle "
                    f"'{value}' qui n'existe pas dans 'models' "
                    f"(rôles disponibles : {', '.join(sorted(models))})."
                )

    return config


@lru_cache(maxsize=1)
def load_llm_config() -> dict:
    """
    Charge llm.yaml, valide le schéma complet et retourne la configuration
    des modèles Ollama (section ``models`` : les rôles fonctionnels, et
    section ``defaults`` : les rôles actifs).

    Lève LLMConfigError (sous-classe de ConfigError) avec un message
    explicite désignant l'entrée fautive dès que le fichier est mal formé —
    y compris pour une erreur de syntaxe YAML.
    Mise en cache pour éviter une relecture répétée du fichier.
    """
    config = _load_yaml(LLM_YAML_PATH)
    return validate_llm_config(config)


def _is_valid_relative_path(value: str) -> bool:
    """True when ``value`` is a usable, non-traversal path reference."""
    return not ("\x00" in value or value.strip() == "" or ".." in Path(value).parts)


def validate_ingestion_config(config: object) -> dict:
    """Validate the full ingestion.yaml schema; raise IngestionConfigError.

    Checks (in order):

    * the document is a mapping;
    * ``documents_root`` exists, is a string, is a usable (non-traversal)
      path reference;
    * ``extensions`` exists, is a dict, is non-empty, and every entry is
      ``".ext" -> folder-name`` (dotted lowercase extension, non-empty
      plain folder name, no path separators);
    * optional extraction settings ``extraction_output_dir``,
      ``extraction_mineru_output_dir``, ``mineru_json_extension``,
      ``summarization_output_dir``, ``extraction_job_file`` and
      ``summarization_job_file`` are strings and usable when present;
    * optional summarization settings ``summary_min_chars`` and
      ``summary_max_chars`` are strictly positive numbers when present
      (the summarizer copies content below the first one verbatim and
      targets the second one).

    Returns the same dict on success, so callers can do
    ``config = validate_ingestion_config(config)``.
    """
    prefix = "ingestion.yaml invalide"

    if not isinstance(config, dict):
        raise IngestionConfigError(
            f"{prefix}: le contenu doit être un mapping YAML "
            f"(type trouvé : {type(config).__name__})."
        )

    # -- documents_root -----------------------------------------------------
    if "documents_root" not in config:
        raise IngestionConfigError(
            f"{prefix} : clé requise manquante 'documents_root' (chemin du "
            "dossier contenant les documents à ingérer)."
        )
    documents_root = config["documents_root"]
    if not isinstance(documents_root, str):
        raise IngestionConfigError(
            f"{prefix} : 'documents_root' doit être une chaîne "
            f"(type trouvé : {type(documents_root).__name__})."
        )
    if not documents_root.strip():
        raise IngestionConfigError(
            f"{prefix} : 'documents_root' ne doit pas être vide."
        )
    if not _is_valid_relative_path(documents_root):
        raise IngestionConfigError(
            f"{prefix} : 'documents_root' ({documents_root!r}) n'est pas un "
            "chemin utilisable (vide ou contient '..')."
        )

    # -- extensions ---------------------------------------------------------
    if "extensions" not in config:
        raise IngestionConfigError(
            f"{prefix} : clé requise manquante 'extensions' (mapping "
            "extension -> sous-dossier)."
        )
    extensions = config["extensions"]
    if not isinstance(extensions, dict):
        raise IngestionConfigError(
            f"{prefix} : 'extensions' doit être un mapping "
            f"(type trouvé : {type(extensions).__name__})."
        )
    if not extensions:
        raise IngestionConfigError(
            f"{prefix} : 'extensions' ne doit pas être vide "
            "(aucune extension ingestable)."
        )

    seen_folders: dict = {}
    for ext, folder in extensions.items():
        # Keys: YAML may parse unquoted keys as non-strings (e.g. .md -> date)
        if not isinstance(ext, str):
            raise IngestionConfigError(
                f"{prefix} : 'extensions' : la clé {ext!r} doit être une "
                "chaîne ; mettez l'extension entre guillemets dans le yaml "
                "(ex. \".md\": \"text\")."
            )
        if not ext.startswith(".") or len(ext) < 2:
            raise IngestionConfigError(
                f"{prefix} : 'extensions' : la clé {ext!r} doit être une "
                "extension commençant par un point (ex. \".pdf\")."
            )
        if ext != ext.lower():
            raise IngestionConfigError(
                f"{prefix} : 'extensions' : la clé {ext!r} doit être en "
                "minuscules (ex. \".pdf\", pas \".PDF\")."
            )
        if "\x00" in ext or "/" in ext or "\\" in ext:
            raise IngestionConfigError(
                f"{prefix} : 'extensions' : clé {ext!r} invalide.",
            )
        if not isinstance(folder, str):
            raise IngestionConfigError(
                f"{prefix} : 'extensions' : la valeur de {ext!r} doit être "
                f"une chaîne (type trouvé : {type(folder).__name__})."
            )
        if not folder.strip():
            raise IngestionConfigError(
                f"{prefix} : 'extensions' : la valeur de {ext!r} ne doit "
                "pas être vide."
            )
        if ("/" in folder or "\\" in folder or ".." in folder
                or folder.strip(". ") != folder or folder in (".", "..")):
            raise IngestionConfigError(
                f"{prefix} : 'extensions' : la valeur de {ext!r} ({folder!r}) "
                "doit être un nom de sous-dossier simple, pas un chemin."
            )
        seen_folders.setdefault(folder, []).append(ext)

    # Several extensions may share one folder (text/), that is fine — but a
    # duplicated *extension* key silently overwrites in YAML, so no check here.

    # -- optional extraction settings ---------------------------------------
    # -- optional summarization settings -------------------------------------
    for key in ("summary_min_chars", "summary_max_chars"):
        if key not in config:
            continue
        value = config[key]
        if not _is_number(value) or value <= 0:
            raise IngestionConfigError(
                f"{prefix} : '{key}' doit être un nombre strictement positif "
                f"(valeur : {value!r})."
            )

    # -- optional extraction settings ---------------------------------------
    for key in ("extraction_output_dir", "extraction_mineru_output_dir",
                "summarization_output_dir",
                "mineru_json_extension", "extraction_job_file",
                "summarization_job_file"):
        if key not in config:
            continue
        value = config[key]
        if not isinstance(value, str):
            raise IngestionConfigError(
                f"{prefix} : '{key}' doit être une chaîne "
                f"(type trouvé : {type(value).__name__})."
            )
        if not value.strip():
            raise IngestionConfigError(
                f"{prefix} : '{key}' ne doit pas être vide "
                "(supprimez la clé pour utiliser la valeur par défaut)."
            )
        if key in ("extraction_output_dir", "extraction_mineru_output_dir",
                   "summarization_output_dir",
                   "extraction_job_file", "summarization_job_file") \
                and not _is_valid_relative_path(value):
            raise IngestionConfigError(
                f"{prefix} : '{key}' ({value!r}) n'est pas un "
                "chemin utilisable (vide ou contient '..')."
            )
        if key == "mineru_json_extension":
            if "/" in value or "\\" in value or "\x00" in value:
                raise IngestionConfigError(
                    f"{prefix} : 'mineru_json_extension' ({value!r}) doit être "
                    "un suffixe de nom de fichier, pas un chemin."
                )
            if not value.startswith(".") and not value.startswith("_"):
                raise IngestionConfigError(
                    f"{prefix} : 'mineru_json_extension' ({value!r}) doit "
                    "commencer par '.' ou '_' (ex. \"_content_list.json\")."
                )

    return config


@lru_cache(maxsize=1)
def load_ingestion_config() -> dict:
    """
    Charge ingestion.yaml, valide le schéma complet et retourne la
    configuration de l'ingestion : dossier racine des documents
    (documents_root), association extension -> sous-dossier (extensions),
    et paramètres d'extraction (extraction_output_dir — le magasin
    canonique du contenu extrait, extraction_mineru_output_dir — le
    bac de travail de MinerU, mineru_json_extension,
    extraction_job_file) et de résumé (summary_min_chars — copie
    verbatim sous la limite, summary_max_chars — cible de taille des
    résumés LLM).

    Lève IngestionConfigError (sous-classe de ConfigError) avec un message
    explicite désignant l'entrée fautive dès que le fichier est mal formé —
    y compris pour une erreur de syntaxe YAML.
    Mise en cache pour éviter une relecture répétée du fichier.
    """
    config = _load_yaml(INGESTION_YAML_PATH)
    return validate_ingestion_config(config)


def validate_vector_config(config: object) -> dict:
    """Validate setup.yaml's ``vector_db`` section; raise VectorConfigError.

    Checks:

    * the document is a mapping containing a ``vector_db`` mapping;
    * ``vector_db.path`` is a required, non-empty, usable (non-traversal)
      path reference — the embedded ChromaDB store's folder;
    * ``vector_db.embedding_batch_size`` (optional tuning knob) is a
      strictly positive int when present (texts per embedding HTTP call);
    * ``vector_db.collections`` is a required, non-empty mapping. The two
      collections the system knows are required (``source_chunks`` and
      ``knowledge_chunks``); any additional key is validated the same way
      (future collections). Every value must be a non-empty single name
      (no path separators) and names must be unique across entries — two
      collections sharing a name would silently be the same store.

    Returns the same dict on success, so callers can do
    ``config = validate_vector_config(config)``.
    """
    prefix = "setup.yaml invalide (section vector_db)"

    if not isinstance(config, dict):
        raise VectorConfigError(
            f"{prefix}: le contenu doit être un mapping YAML "
            f"(type trouvé : {type(config).__name__})."
        )

    if "vector_db" not in config:
        raise VectorConfigError(
            f"{prefix} : section requise manquante 'vector_db'."
        )
    vector = config["vector_db"]
    if not isinstance(vector, dict):
        raise VectorConfigError(
            f"{prefix} : 'vector_db' doit être un mapping "
            f"(type trouvé : {type(vector).__name__})."
        )

    # -- path ----------------------------------------------------------------
    if "path" not in vector:
        raise VectorConfigError(
            f"{prefix} : clé requise manquante 'vector_db.path' (dossier du "
            "magasin ChromaDB embarqué)."
        )
    path = vector["path"]
    if not isinstance(path, str):
        raise VectorConfigError(
            f"{prefix} : 'vector_db.path' doit être une chaîne "
            f"(type trouvé : {type(path).__name__})."
        )
    if not path.strip():
        raise VectorConfigError(
            f"{prefix} : 'vector_db.path' ne doit pas être vide."
        )
    if not _is_valid_relative_path(path):
        raise VectorConfigError(
            f"{prefix} : 'vector_db.path' ({path!r}) n'est pas un chemin "
            "utilisable (vide ou contient '..')."
        )

    # -- embedding batch size (optional tuning knob) ----------------------------
    if "embedding_batch_size" in vector:
        batch = vector["embedding_batch_size"]
        if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
            raise VectorConfigError(
                f"{prefix} : 'vector_db.embedding_batch_size' doit être un "
                f"entier strictement positif (valeur : {batch!r})."
            )

    if "collections" not in vector:
        raise VectorConfigError(
            f"{prefix} : clé requise manquante 'vector_db.collections' "
            "(noms des collections source_chunks et knowledge_chunks)."
        )
    collections = vector["collections"]
    if not isinstance(collections, dict):
        raise VectorConfigError(
            f"{prefix} : 'vector_db.collections' doit être un mapping "
            f"(type trouvé : {type(collections).__name__})."
        )
    if not collections:
        raise VectorConfigError(
            f"{prefix} : 'vector_db.collections' ne doit pas être vide "
            "(au moins source_chunks et knowledge_chunks)."
        )

    REQUIRED_COLLECTIONS = ("source_chunks", "knowledge_chunks")
    for required in REQUIRED_COLLECTIONS:
        if required not in collections:
            raise VectorConfigError(
                f"{prefix} : 'vector_db.collections' : collection requise "
                f"manquante '{required}'."
            )

    seen_names: dict = {}
    for key, name in collections.items():
        if not isinstance(key, str) or not key.strip():
            raise VectorConfigError(
                f"{prefix} : 'vector_db.collections' : nom de collection "
                f"invalide {key!r} (doit être une chaîne non vide)."
            )
        if not isinstance(name, str):
            raise VectorConfigError(
                f"{prefix} : 'vector_db.collections.{key}' doit être une "
                f"chaîne (type trouvé : {type(name).__name__})."
            )
        if not name.strip():
            raise VectorConfigError(
                f"{prefix} : 'vector_db.collections.{key}' ne doit pas être "
                "vide."
            )
        if ("/" in name or "\\" in name or "\x00" in name
                or name.strip(". ") != name or name in (".", "..")):
            raise VectorConfigError(
                f"{prefix} : 'vector_db.collections.{key}' ({name!r}) doit "
                "être un nom simple de collection, pas un chemin."
            )
        if name in seen_names:
            raise VectorConfigError(
                f"{prefix} : 'vector_db.collections.{key}' ({name!r}) "
                f"duplique le nom déjà utilisé par "
                f"'vector_db.collections.{seen_names[name]}' — deux "
                "collections doivent porter des noms distincts."
            )
        seen_names[name] = key

    return config


@lru_cache(maxsize=1)
def load_vector_config() -> dict:
    """Load setup.yaml and return its validated ``vector_db`` section.

    Validates the full vector_db schema at load time (path, collection
    names) and raises VectorConfigError (a ConfigError subclass) with an
    explicit message naming the faulty entry — including for a YAML syntax
    error. Cached: read and validated once per program run.
    """
    config = _load_yaml(SETUP_YAML_PATH)
    validate_vector_config(config)
    return config["vector_db"]


# ------------------------------------------------------------------
# setup.yaml — validation et chargement de la section routing
# ------------------------------------------------------------------

def validate_routing_config(config: object) -> dict:
    """Validate setup.yaml's ``routing`` section; raise RoutingConfigError.

    The section is optional (an absent ``routing`` key keeps the default:
    ``max_requests_per_prompt = 8``); when present it is validated
    strictly:

    * ``max_requests_per_prompt`` (optional) is a strictly positive int —
      the cap on how many structured requests one user prompt may yield;
      a bool is rejected explicitly (a ``bool`` is an ``int`` in Python).

    Returns the same dict on success, so callers can do
    ``config = validate_routing_config(config)``.
    """
    prefix = "setup.yaml invalide (section routing)"

    if not isinstance(config, dict):
        raise RoutingConfigError(
            f"{prefix}: le contenu doit être un mapping YAML "
            f"(type trouvé : {type(config).__name__})."
        )

    if "routing" not in config:
        return config
    routing = config["routing"]
    if not isinstance(routing, dict):
        raise RoutingConfigError(
            f"{prefix} : 'routing' doit être un mapping "
            f"(type trouvé : {type(routing).__name__})."
        )

    if "max_requests_per_prompt" in routing:
        cap = routing["max_requests_per_prompt"]
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            raise RoutingConfigError(
                f"{prefix} : 'routing.max_requests_per_prompt' doit être un "
                f"entier strictement positif (valeur : {cap!r})."
            )

    return config


@lru_cache(maxsize=1)
def load_routing_config() -> dict:
    """Load setup.yaml and return its validated ``routing`` section.

    The section is optional: an absent section yields the default
    ``{"max_requests_per_prompt": 8}`` so a minimal setup.yaml keeps
    working. Malformed present values raise RoutingConfigError (a
    ConfigError subclass) naming the faulty entry. Cached: read and
    validated once per program run.
    """
    config = _load_yaml(SETUP_YAML_PATH)
    validate_routing_config(config)
    routing = config.get("routing") or {}
    return {
        "max_requests_per_prompt": int(
            routing.get("max_requests_per_prompt", 8)
        )
    }


# ------------------------------------------------------------------
# retrieval.yaml — validation et chargement
# ------------------------------------------------------------------

def validate_retrieval_config(config: object) -> dict:
    """Validate the full retrieval.yaml schema; raise RetrievalConfigError.

    Checks (in order):

    * the document is a mapping;
    * ``default_top_k`` (required) and ``max_top_k`` (required) are
      strictly positive ints with ``default_top_k <= max_top_k``;
    * ``min_score`` (required) is a number in [0.0, 1.0] — a cosine
      similarity threshold;
    * ``embedding_role`` (required, non-empty string) names an llm.yaml
      role — cross-file consistency is checked here because a typo would
      otherwise surface as a runtime LLM failure;
    * ``source_collection_key`` (required, non-empty string) is a plain
      name (the key under setup.yaml's ``vector_db.collections``);
    * ``query_instruction`` (optional) is a string when present — the
      query-side instruction prepended to questions for instruction-aware
      embedding models; empty disables it.
    """
    prefix = "retrieval.yaml invalide"

    if not isinstance(config, dict):
        raise RetrievalConfigError(
            f"{prefix}: le contenu doit être un mapping YAML "
            f"(type trouvé : {type(config).__name__})."
        )

    # -- required positive ints ---------------------------------------------
    for key in ("default_top_k", "max_top_k"):
        if key not in config:
            raise RetrievalConfigError(
                f"{prefix} : clé requise manquante '{key}'."
            )
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RetrievalConfigError(
                f"{prefix} : '{key}' doit être un entier strictement positif "
                f"(valeur : {value!r})."
            )

    if config["default_top_k"] > config["max_top_k"]:
        raise RetrievalConfigError(
            f"{prefix} : 'default_top_k' ({config['default_top_k']}) ne doit "
            f"pas dépasser 'max_top_k' ({config['max_top_k']})."
        )

    # -- min_score -------------------------------------------------------------
    if "min_score" not in config:
        raise RetrievalConfigError(
            f"{prefix} : clé requise manquante 'min_score' (seuil de "
            "similarité cosine, entre 0.0 et 1.0)."
        )
    min_score = config["min_score"]
    if not _is_number(min_score) or not 0.0 <= min_score <= 1.0:
        raise RetrievalConfigError(
            f"{prefix} : 'min_score' doit être un nombre entre 0.0 et 1.0 "
            f"(valeur : {min_score!r})."
        )

    # -- embedding_role: must name an existing llm.yaml role -------------------
    if "embedding_role" not in config:
        raise RetrievalConfigError(
            f"{prefix} : clé requise manquante 'embedding_role' (rôle llm.yaml "
            "fournissant les embeddings des requêtes — le même modèle que "
            "celui qui a indexé le corpus)."
        )
    embedding_role = config["embedding_role"]
    if not isinstance(embedding_role, str) or not embedding_role.strip():
        raise RetrievalConfigError(
            f"{prefix} : 'embedding_role' doit être une chaîne non vide "
            f"(type trouvé : {type(embedding_role).__name__})."
        )
    try:
        llm_roles = set(load_llm_config().get("models", {}))
    except Exception as exc:  # noqa: BLE001 — reported with the real cause below
        raise RetrievalConfigError(
            f"{prefix} : 'embedding_role' ({embedding_role!r}) n'a pas pu être "
            f"vérifié contre llm.yaml : {exc}"
        ) from exc
    if embedding_role not in llm_roles:
        raise RetrievalConfigError(
            f"{prefix} : 'embedding_role' ({embedding_role!r}) ne nomme aucun "
            "rôle de llm.yaml (rôles disponibles : "
            f"{', '.join(sorted(llm_roles))}) — les requêtes doivent être "
            "embedées avec le MÊME modèle que le corpus."
        )

    # -- source_collection_key ---------------------------------------------------
    if "source_collection_key" not in config:
        raise RetrievalConfigError(
            f"{prefix} : clé requise manquante 'source_collection_key' (clé "
            "sous setup.yaml vector_db.collections)."
        )
    collection_key = config["source_collection_key"]
    if not isinstance(collection_key, str) or not collection_key.strip():
        raise RetrievalConfigError(
            f"{prefix} : 'source_collection_key' doit être une chaîne non vide "
            f"(type trouvé : {type(collection_key).__name__})."
        )
    if "/" in collection_key or "\\" in collection_key or "\x00" in collection_key:
        raise RetrievalConfigError(
            f"{prefix} : 'source_collection_key' ({collection_key!r}) doit être "
            "un nom simple de clé, pas un chemin."
        )

    # -- query_instruction (optional) ------------------------------------------
    # Query-side instruction for instruction-aware embedding models
    # (qwen3-embedding: "Instruct: ...\nQuery: ..."). Empty string or an
    # absent key disables the instruction entirely — a model without a
    # query protocol must never receive one. A present value must be a
    # plain string: an instruction is sent VERBATIM to the embedder, so a
    # non-string type would crash the retrieval agent at query time.
    if "query_instruction" in config:
        instruction = config["query_instruction"]
        if not isinstance(instruction, str):
            raise RetrievalConfigError(
                f"{prefix} : 'query_instruction' doit être une chaîne "
                f"(type trouvé : {type(instruction).__name__}) — vide pour "
                "désactiver l'instruction."
            )

    return config


@lru_cache(maxsize=1)
def load_retrieval_config() -> dict:
    """Load retrieval.yaml, validate the full schema and return it.

    Lève RetrievalConfigError (sous-classe de ConfigError) avec un message
    explicite désignant l'entrée fautive dès que le fichier est mal formé —
    y compris pour une erreur de syntaxe YAML. Cached: read and validated
    once per program run.
    """
    config = _load_yaml(RETRIEVAL_YAML_PATH)
    return validate_retrieval_config(config)


# ------------------------------------------------------------------
# Accès pratiques à un modèle spécifique
# ------------------------------------------------------------------

def get_model_config(role: str) -> dict:
    """
    Retourne la configuration d'un modèle selon son rôle fonctionnel
    ('router', 'summarizer', 'default', ...) tel que défini dans llm.yaml.

    STRICT : ne retombe JAMAIS sur le rôle 'default' quand le rôle demandé
    n'existe pas — un nom de rôle mal orthographié est une erreur, pas un
    repli silencieux. Si le modèle par défaut est voulu, demander
    explicitement le rôle 'default'. Les agents utilisent
    ``src.agents.llm_roles.require_llm_role`` pour ce contrat strict.

    Lève une ConfigError (LLMConfigError si llm.yaml est mal formé) si le
    rôle demandé n'existe pas.
    """
    llm_config = load_llm_config()
    models = llm_config.get("models", {})
    if role not in models:
        raise ConfigError(
            f"Rôle de modèle '{role}' introuvable dans llm.yaml. "
            f"Rôles disponibles : {list(models.keys())}"
        )
    return models[role]


# ------------------------------------------------------------------
# Point de vérification manuelle (exécution directe du module)
# ------------------------------------------------------------------

if __name__ == "__main__":
    try:
        setup_cfg = load_setup_config()
        llm_cfg = load_llm_config()
        print("setup.yaml chargé avec succès. Clés principales :", list(setup_cfg.keys()))
        print("llm.yaml chargé avec succès. Modèles définis :", list(llm_cfg.get("models", {}).keys()))
    except ConfigError as e:
        print(f"[ERREUR CONFIG] {e}")