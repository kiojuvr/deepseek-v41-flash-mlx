# Working agreement

- Continue authorized implementation and short correctness checks autonomously.
- Before starting a validation expected to take several minutes or longer, prepare a reproducible script for the user to run. Include the exact command, expected scope/resource use, log/result paths, failure handling and a resume mechanism when practical. Do not launch it and spend the turn monitoring it. This records the user's explicit preference.
- Do not call a long validation passed until its result has been read and checked. Continue independent implementation while the user runs it.
- Checkpoint data remains read-only. Python is permitted for offline oracle/fixture generation and validation orchestration, never as a production runtime dependency.
- Maintain the correctness, memory, and full-path promotion contracts in `docs/`. An isolated kernel/storage probe is not end-to-end runtime qualification.
