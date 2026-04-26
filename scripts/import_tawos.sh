#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SQL_FILE="${1:-"$ROOT_DIR/data/raw/TAWOS.sql"}"
DB_NAME="${TAWOS_DB_NAME:-tawos_cleaned}"
MYSQL_USER="${MYSQL_USER:-}"
MYSQL_BIN="${MYSQL_BIN:-mysql}"

if [[ -x /usr/local/mysql/bin/mysql && "$MYSQL_BIN" == "mysql" ]]; then
  MYSQL_BIN="/usr/local/mysql/bin/mysql"
fi

if [[ ! -f "$SQL_FILE" ]]; then
  echo "SQL dump not found: $SQL_FILE" >&2
  exit 1
fi

if [[ -z "$MYSQL_USER" ]]; then
  read -r -p "MySQL username only, not password [root]: " MYSQL_USER_INPUT
  MYSQL_USER="${MYSQL_USER_INPUT:-root}"
fi

read -r -s -p "MySQL password for ${MYSQL_USER}: " MYSQL_PASSWORD
echo

CLIENT_CNF="$(mktemp)"
trap 'rm -f "$CLIENT_CNF"' EXIT
chmod 600 "$CLIENT_CNF"
cat > "$CLIENT_CNF" <<EOF
[client]
user=${MYSQL_USER}
password=${MYSQL_PASSWORD}
default-character-set=utf8mb4
EOF

echo "Creating database ${DB_NAME} if needed..."
"$MYSQL_BIN" --defaults-extra-file="$CLIENT_CNF" \
  -e "CREATE DATABASE IF NOT EXISTS \`${DB_NAME}\` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;"

echo "Importing ${SQL_FILE} into ${DB_NAME}. This can take several minutes..."
"$MYSQL_BIN" --defaults-extra-file="$CLIENT_CNF" "$DB_NAME" < "$SQL_FILE"

echo "Verifying import..."
"$MYSQL_BIN" --defaults-extra-file="$CLIENT_CNF" "$DB_NAME" \
  -e "SELECT COUNT(*) AS issue_count FROM Issue; SELECT COUNT(*) AS issue_link_count FROM Issue_Link; SHOW TABLES;"

echo "TAWOS import complete."
