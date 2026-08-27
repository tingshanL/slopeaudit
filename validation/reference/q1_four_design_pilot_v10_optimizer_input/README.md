# Private v10 pilot provenance stub

The historical v10 pilot inputs used by the v11 optimizer diagnostic are
**restricted-derived provenance** because they encode the exact AABC2-derived
irregular schedule through seeds and diagnostic summaries. The payload files
are therefore not distributed in the public repository.

Authorized internal reproduction may point `SLOPEAUDIT_Q1_PILOT_INPUT` to a
private byte-preserved copy of the completed-run directory and
`SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE` to the exact private schedule. These files
are not required for package tests or ordinary examples.

The private record includes payload hashes. Historical reproduction requires
the byte-preserved inputs rather than an approximate schedule.
