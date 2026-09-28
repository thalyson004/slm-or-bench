# Datasets

The benchmark input consists of three JSON files:

- `ComplexOR.json` — 18 executable optimization instances;
- `LogiOR.json` — 92 logistics optimization instances;
- `IndustryOR.json` — 82 executable optimization instances.

`ComplexOR.json` and `IndustryOR.json` include the question, expected objective, formulation, code, and verification records consumed by the runner. `LogiOR.json` stores its expected objective in each record.

The runner records SHA-256 hashes and selected example identifiers in each run manifest before inference starts. The files in this directory are release inputs; changing one creates a new benchmark data version.

The two worked few-shot demonstrations are embedded in the versioned prompts and are not evaluation instances. They do not change any of the three denominators.
