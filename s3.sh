#!/usr/bin/bash
poetry run python -m torch_exec.fuzz_loop --input-dir ./test_input --result-dir ./test_output
