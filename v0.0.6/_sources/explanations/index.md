# Explanations

Conceptual background for the interfaces documented in {doc}`../api/index` and
applied in the {doc}`examples <../examples/index>`. These pages state the
vocabulary, the models and the conventions the rest of the documentation
assumes.

| Explanation | What it covers |
| --- | --- |
| {doc}`pulseq/index` | What a `.seq` file records, how it stores it, and the rasters an event time is addressed on. |
| {doc}`design/index` | The two abstractions the package places above the file format, what each is responsible for, and how sampling support and ordering feed them. |
| {doc}`safety/index` | What each check computes, the model it computes it from, and the criterion it applies. |

```{toctree}
:hidden:

pulseq/index
design/index
safety/index
```
