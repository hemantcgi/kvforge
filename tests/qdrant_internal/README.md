# Qdrant Internal Tests

These test suites are carried over from the upstream [Qdrant](https://github.com/qdrant/qdrant)
repository. They test Qdrant's internal cluster consensus, TLS configuration, snapshot
compatibility, and REST API compliance — **not** SmartQdrant's RAG pipeline.

## Contents

| Directory | What it tests |
|-----------|---------------|
| `consensus_tests/` | Raft consensus protocol between Qdrant cluster nodes |
| `e2e_tests/` | End-to-end: TLS, snapshots, data compatibility across versions |
| `openapi/` | Qdrant REST API endpoint contract tests |

## Running these tests

These tests require a running Qdrant cluster and are not part of the SmartQdrant CI suite.
Run SmartQdrant-specific tests with:

    pytest tests/test_*.py -v
