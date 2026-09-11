# Knowledge extraction — characters

Role: `summarizer` (config/llm.yaml) unless a dedicated role is added later.

Extract the characters mentioned in the given content. One pass per content
unit (page / section / chapter); the orchestrator runs this prompt several
times, once per knowledge model family.

## Output format

Respond with a single JSON object, no prose, no markdown fences:

```json
{
  "characters": [
    {
      "full_name": "Edmond Dantès",
      "short_name": "Dantès",
      "aliases": ["Edmond", "le Comte de Monte-Cristo", "Lord Wilmore"]
    }
  ]
}
```

## Rules

- `full_name`: the name most often seen in the sources for this character.
- `short_name`: the shortest name by which the character is usually called
  (used to build the stable id `char:<short_name>`; slugified downstream).
  When only a full name exists, repeat it.
- `aliases`: every other way the character is called in the text — short
  names, hypocoristics, pseudonyms, titles ("Monseigneur"), nicknames.
  Empty list when none.
- One entry per distinct character per pass; duplicates across passes are
  expected and will be merged later (check-n-merge step) — do not try to
  resolve identities yourself.
- Never invent characters that are not named in the text. Anonymous roles
  ("the jailer") are extracted only if they recur or matter to the plot.
- Preserve accents and original spelling ("Dantès", not "Dantes").
- Do not include the narrator or the reader.

## Failure mode

If the content is unreadable or you cannot comply, respond with:

```json
{"characters": [], "error": "reason"}
```

The orchestrator treats an `error` field as a retryable LLM-response failure.
