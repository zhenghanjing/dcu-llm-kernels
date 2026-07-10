#!/bin/sh
# run_rocprof_probe.sh -- one-shot real-hardware probe of rocprof's actual
# CLI surface on this DTK install, same rationale as run_rocm_probe.sh: we
# don't know if rocprof is even installed here, or what flags/output format
# this specific version supports, and the portal gives no interactive shell
# to poke around -- so print everything to stdout for a human to read
# instead of guessing flags for a real profiling run and finding out later.
echo "=== which rocprof ==="
which rocprof 2>&1
echo
echo "=== rocprof --version ==="
rocprof --version 2>&1
echo
echo "=== rocprof (no args) ==="
rocprof 2>&1
echo
echo "=== rocprof --help (first 120 lines) ==="
rocprof --help 2>&1 | head -120
echo
echo "=== hipcc / DTK version context ==="
hipcc --version 2>&1
echo
echo "=== which rocm-smi (for cross-reference) ==="
which rocm-smi 2>&1
