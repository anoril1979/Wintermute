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


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
SETUP_YAML_PATH = CONFIG_DIR / "setup.yaml"
LLM_YAML_PATH = CONFIG_DIR / "llm.yaml"
INGESTION_YAML_PATH = CONFIG_DIR / "ingestion.yaml"
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