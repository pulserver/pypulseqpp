# The sequence corpus

These files are the reference corpus from
[pypulseq-matlab-like](https://github.com/m-a-x-i-m-z/pypulseq-matlab-like),
`tests/expected_output`, which is MIT licensed and is the transcription of
MATLAB Pulseq this package treats as the authority for the file format.

They are checked in rather than generated because that is what makes them
evidence: each was written by a toolbox other than this one, several by
MATLAB itself, and between them they span every revision from 1.2.0 to 1.5.1
and every section the format has. A reader tested only against files its own
writer produced is tested against its own assumptions.

`simple_mprage1XX.seq` is one sequence written at seven revisions, which is
what makes it the test of reading an older file: what 1.5.0 says is what the
others have to be read as.
