#!/bin/bash
set -eo pipefail

###############################################################################
# Validate required environment variables
###############################################################################
if [ -z "$SUPERLINK_URL" ]; then
  echo "ERROR: SUPERLINK_URL is not set."
  echo "       Pass -e SUPERLINK_URL=<superlink-ip>:9092"
  exit 1
fi

if [ -z "$DATA_PATH" ]; then
  echo "ERROR: DATA_PATH is not set."
  echo "       Pass -e DATA_PATH=/path/to/data"
  exit 1
fi

echo "Starting SuperNode..."
echo "  SUPERLINK_URL: $SUPERLINK_URL"
echo "  DATA_PATH:     $DATA_PATH"

###############################################################################
# Start SuperNode
###############################################################################
exec flower-supernode \
  --root-certificates /app/certs/ca.crt \
  --superlink "$SUPERLINK_URL" \
  --node-config "data-path=\"$DATA_PATH\""
