#!/bin/sh
# The container's CMD. The harness execs into this container for the whole evaluation, so keep the
# server alive: if it ever dies (driver hiccup, OOM) it comes straight back instead of leaving every
# later question without an answer.
cd /app
while true; do
    python3 /app/server.py
    echo "server exited with $?, restarting in 2s" >&2
    sleep 2
done
