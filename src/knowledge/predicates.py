"""Predicate registry for the knowledge layer.

``PREDICATES`` maps each :class:`~src.knowledge.models.Predicate` to its
:class:`~src.knowledge.models.PredicateDefinition` (accepted entity types
for subject/object, simple ``value`` type, symmetric flag, inverse).
It is the single source of truth used by the validation layer
(src/validation) to check claims against registered entities.
"""

from src.knowledge.models import (
    EntityType,
    Predicate,
    PredicateDefinition,
)


def _def(
    predicate: Predicate,
    subject_types: set[EntityType],
    object_types: set[EntityType] | None = None,
    value_type: str | None = None,
    symmetric: bool = False,
    inverse: Predicate | None = None,
) -> PredicateDefinition:
    """Compact helper to build a PredicateDefinition."""
    return PredicateDefinition(
        predicate=predicate,
        subject_types=subject_types,
        object_types=object_types,
        value_type=value_type,
        symmetric=symmetric,
        inverse=inverse,
    )


CHAR = EntityType.CHARACTER
PLACE = EntityType.PLACE
OBJ = EntityType.OBJECT
EVENT = EntityType.EVENT
ORG = EntityType.ORGANIZATION

PREDICATES: dict[Predicate, PredicateDefinition] = {
    # --- Between characters -------------------------------------------
    Predicate.SPOUSE_OF: _def(Predicate.SPOUSE_OF, {CHAR}, {CHAR}, symmetric=True),
    Predicate.PARENT_OF: _def(Predicate.PARENT_OF, {CHAR}, {CHAR}, inverse=Predicate.CHILD_OF),
    Predicate.CHILD_OF: _def(Predicate.CHILD_OF, {CHAR}, {CHAR}, inverse=Predicate.PARENT_OF),
    Predicate.SIBLING_OF: _def(Predicate.SIBLING_OF, {CHAR}, {CHAR}, symmetric=True),
    Predicate.FRIEND_OF: _def(Predicate.FRIEND_OF, {CHAR}, {CHAR}, symmetric=True),
    Predicate.ENEMY_OF: _def(Predicate.ENEMY_OF, {CHAR}, {CHAR}, symmetric=True),
    Predicate.ALLY_OF: _def(Predicate.ALLY_OF, {CHAR}, {CHAR}, symmetric=True),
    Predicate.KNOWS: _def(Predicate.KNOWS, {CHAR}, {CHAR}, symmetric=True),
    Predicate.WORKS_WITH: _def(Predicate.WORKS_WITH, {CHAR}, {CHAR}, symmetric=True),
    # One-directional for now: no inverse predicate in the vocabulary yet.
    Predicate.COMMANDS: _def(Predicate.COMMANDS, {CHAR}, {CHAR}),
    Predicate.FOLLOWS: _def(Predicate.FOLLOWS, {CHAR}, {CHAR}),
    Predicate.BETRAYED: _def(Predicate.BETRAYED, {CHAR}, {CHAR}),
    Predicate.TRUSTS: _def(Predicate.TRUSTS, {CHAR}, {CHAR}),
    Predicate.LOVES: _def(Predicate.LOVES, {CHAR}, {CHAR}),
    # --- Character / Place --------------------------------------------
    Predicate.LIVES_IN: _def(Predicate.LIVES_IN, {CHAR}, {PLACE}, inverse=Predicate.HAS_INHABITANT),
    Predicate.HAS_INHABITANT: _def(Predicate.HAS_INHABITANT, {PLACE}, {CHAR}, inverse=Predicate.LIVES_IN),
    Predicate.BORN_IN: _def(Predicate.BORN_IN, {CHAR}, {PLACE}),
    Predicate.DIED_IN: _def(Predicate.DIED_IN, {CHAR}, {PLACE}),
    Predicate.VISITED: _def(Predicate.VISITED, {CHAR}, {PLACE}),
    Predicate.TRAVELLED_TO: _def(Predicate.TRAVELLED_TO, {CHAR}, {PLACE}),
    # --- Character / Object --------------------------------------------
    Predicate.OWNS: _def(Predicate.OWNS, {CHAR}, {OBJ}, inverse=Predicate.IS_OWNED),
    Predicate.IS_OWNED: _def(Predicate.IS_OWNED, {OBJ}, {CHAR}, inverse=Predicate.OWNS),
    Predicate.USES: _def(Predicate.USES, {CHAR}, {OBJ}, inverse=Predicate.IS_USED),
    Predicate.IS_USED: _def(Predicate.IS_USED, {OBJ}, {CHAR}, inverse=Predicate.USES),
    Predicate.CARRIES: _def(Predicate.CARRIES, {CHAR}, {OBJ}, inverse=Predicate.IS_CARRIED),
    Predicate.IS_CARRIED: _def(Predicate.IS_CARRIED, {OBJ}, {CHAR}, inverse=Predicate.CARRIES),
    Predicate.CREATED: _def(Predicate.CREATED, {CHAR}, {OBJ}, inverse=Predicate.IS_CREATED),
    Predicate.IS_CREATED: _def(Predicate.IS_CREATED, {OBJ}, {CHAR}, inverse=Predicate.CREATED),
    Predicate.DESTROYED: _def(Predicate.DESTROYED, {CHAR}, {OBJ}, inverse=Predicate.IS_DESTROYED),
    Predicate.IS_DESTROYED: _def(Predicate.IS_DESTROYED, {OBJ}, {CHAR}, inverse=Predicate.DESTROYED),
    # --- Character / Event ---------------------------------------------
    Predicate.PARTICIPATED_IN: _def(Predicate.PARTICIPATED_IN, {CHAR}, {EVENT}),
    Predicate.WITNESSED: _def(Predicate.WITNESSED, {CHAR}, {EVENT}),
    Predicate.CAUSED: _def(Predicate.CAUSED, {CHAR}, {EVENT}),
    Predicate.PREVENTED: _def(Predicate.PREVENTED, {CHAR}, {EVENT}),
    Predicate.SURVIVED: _def(Predicate.SURVIVED, {CHAR}, {EVENT}),
    # --- Object / Place -------------------------------------------------
    Predicate.LOCATED_IN: _def(Predicate.LOCATED_IN, {OBJ}, {PLACE}),
    Predicate.CREATED_AT: _def(Predicate.CREATED_AT, {OBJ}, {PLACE}),
    Predicate.DESTROYED_AT: _def(Predicate.DESTROYED_AT, {OBJ}, {PLACE}),
    # --- Object / Event -------------------------------------------------
    Predicate.LOST_DURING: _def(Predicate.LOST_DURING, {OBJ}, {EVENT}),
    Predicate.CREATED_DURING: _def(Predicate.CREATED_DURING, {OBJ}, {EVENT}),
    Predicate.DESTROYED_DURING: _def(Predicate.DESTROYED_DURING, {OBJ}, {EVENT}),
    # --- Value-only (no entity object) ----------------------------------
    Predicate.OCCUPATION: _def(Predicate.OCCUPATION, {CHAR}, value_type="profession"),
}


def get_definition(predicate: Predicate) -> PredicateDefinition:
    """Return the definition for ``predicate`` or raise a clear error."""
    definition = PREDICATES.get(predicate)
    if definition is None:
        raise ValueError(f"Unknown predicate: {predicate}")
    return definition
