#!/bin/sh
set -e

echo "Waiting for Vault..."

until vault status >/dev/null 2>&1
do
  sleep 1
done

echo "Vault is ready"

echo "Writing database configuration..."

vault kv put secret/database \
  host="mssql" \
  port="1433" \
  database="banknote" \
  username="sa" \
  password="$MSSQL_SA_PASSWORD"

echo "Creating API policy..."

vault policy write banknote-api - <<EOF
path "secret/data/database" {
  capabilities = ["read"]
}
EOF

echo "Creating API token..."

API_TOKEN=$(vault token create \
  -policy="banknote-api" \
  -ttl="24h" \
  -renewable=true \
  -field=token)

echo "$API_TOKEN" > /vault-auth/api-token

echo "Vault initialization completed"
