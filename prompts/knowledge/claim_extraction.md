# Knowledge extraction — claims

Role: `summarizer` (config/llm.yaml) unless a dedicated role is added later.

Extract claims from the given content. A claim is a piece of information the
text asserts about the world of the document: a fact or a relation between
two entities, or a property of one entity ("This Guy lives in that City",
"The Man is a soldier").

## Output format

Respond with a single JSON object, no prose, no markdown fences:

```json
{
  "claims": [
    {
      "subject": {"kind": "character", "name": "Edmond Dantès"},
      "predicate": "lives_in",
      "object": {"kind": "place", "name": "Paris"},
      "value": null,
      "confidence": 0.9
    },
    {
      "subject": {"kind": "character", "name": "The Man"},
      "predicate": "occupation",
      "object": null,
      "value": "soldier",
      "confidence": 0.7
    }
  ]
}
```

## Rules

- `predicate` must be one of the predicates listed by the orchestrator in
  the run message (see `src/knowledge/predicates.py` for the registry).
  Never invent predicates.
- Use `object` when the claim links two entities (both must be named in the
  text); use `value` when the claim's object is not an entity worth
  registering ("soldier"). Never fill both.
- `subject`/`object.kind` is one of: `character`, `place`, `object`, `event`,
  `organization`.
- Reference entities by name as they appear in the text; the orchestrator
  resolves them to ids (`char:...`, `place:001`, ...) — do not guess ids.
- `confidence` is your certainty the text actually asserts this claim, as a
  float between 0.0 and 1.0. Downgrade for implicit, ambiguous or
  ironical statements; a character's lie is still a claim *the character
  asserts* — extract it, the status layer will classify it.
- One claim per atomic fact; split compound sentences.
- Extract only what the text asserts, never what you infer from outside
  knowledge. Inferences belong to a later, explicitly-flagged pass.

## Failure mode

If the content is unreadable or you cannot comply, respond with:

```json
{"claims": [], "error": "reason"}
```

The orchestrator treats an `error` field as a retryable LLM-response failure.
