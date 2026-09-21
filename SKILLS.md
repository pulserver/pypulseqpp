# pypulseqpp — agent skills

`AGENTS.md` states the project's rules and invariants. The skills listed here
are the procedures for the recurring tasks those rules govern: what to run,
in what order, and what has to hold before the task is finished.

Each skill is a directory under `.claude/skills/`, holding a `SKILL.md` with
the procedure. Agent runtimes that read `.claude/skills/` discover them
automatically; this page is the index for those that do not.

| Skill | Use it when |
|---|---|
| [`build-and-test`](.claude/skills/build-and-test/SKILL.md) | Building the native extension from a checkout, and running the formatter, linter and test suite before reporting a change complete. Covers which skips are expected and the benchmark a bindings change has to report. |
| [`write-documentation`](.claude/skills/write-documentation/SKILL.md) | Writing, moving or auditing documentation: choosing the documentation type, the binding conventions, the gallery and generator mechanics, and the build that validates the result. |
| [`add-a-sequence`](.claude/skills/add-a-sequence/SKILL.md) | Adding a complete sequence to `examples/sequence/`, or changing one: the `SequenceApp` contract, the catalogue row, the gallery page, the family page and the tests that tie them together. |

A skill states the procedure and points at the authority for the rules it
applies; it does not restate them. Where a skill and `AGENTS.md` disagree,
`AGENTS.md` governs, and the skill is wrong and should be corrected.
