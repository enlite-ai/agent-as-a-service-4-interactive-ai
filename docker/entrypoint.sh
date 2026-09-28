#!/bin/bash
python -m flask run --host=0.0.0.0 --port="${PORT:-5010}" --reload
