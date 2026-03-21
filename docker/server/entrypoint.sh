#!/bin/bash
set -eo pipefail

###############################################################################
# Validate TLS certificates are mounted
###############################################################################
if [ ! -f "/certs/server.crt" ] || [ ! -f "/certs/server.key" ] || [ ! -f "/certs/ca.crt" ]; then
  echo "ERROR: TLS certificates not found in /certs/"
  echo "       Mount certs directory: -v /path/to/certs:/certs:ro"
  echo "       Required files: server.crt, server.key, ca.crt"
  exit 1
fi

echo "Starting SuperLink with TLS..."

###############################################################################
# Start SuperLink
###############################################################################
exec flower-superlink \
  --ssl-certfile  /certs/server.crt \
  --ssl-keyfile   /certs/server.key \
  --ssl-ca-certfile /certs/ca.crt    # ← was --ca-certificate
