#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-"$ROOT_DIR/data/processed"}"
DB_NAME="${TAWOS_DB_NAME:-tawos_cleaned}"
MYSQL_USER="${MYSQL_USER:-}"
MYSQL_BIN="${MYSQL_BIN:-mysql}"

if [[ -x /usr/local/mysql/bin/mysql && "$MYSQL_BIN" == "mysql" ]]; then
  MYSQL_BIN="/usr/local/mysql/bin/mysql"
fi

mkdir -p "$OUT_DIR"

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
database=${DB_NAME}
default-character-set=utf8mb4
EOF

echo "Exporting task records..."
"$MYSQL_BIN" --defaults-extra-file="$CLIENT_CNF" --batch -e "
SELECT
  i.ID AS task_id,
  i.Jira_ID AS jira_id,
  i.Issue_Key AS issue_key,
  i.URL AS issue_url,
  p.Project_Key AS project_key,
  p.Name AS project_name,
  r.Name AS repository_name,
  i.Title AS title,
  i.Description_Text AS description_text,
  i.Type AS task_type,
  i.Priority AS priority,
  i.Status AS status,
  i.Resolution AS resolution,
  i.Creation_Date AS creation_date,
  i.Estimation_Date AS estimation_date,
  i.Resolution_Date AS resolution_date,
  i.Last_Updated AS last_updated,
  i.Story_Point AS story_point,
  CASE
    WHEN i.Story_Point IS NULL THEN NULL
    WHEN i.Story_Point <= 2 THEN 'easy'
    WHEN i.Story_Point <= 5 THEN 'medium'
    ELSE 'hard'
  END AS difficulty_label,
  i.Timespent AS timespent,
  i.In_Progress_Minutes AS in_progress_minutes,
  i.Total_Effort_Minutes AS total_effort_minutes,
  i.Resolution_Time_Minutes AS resolution_time_minutes,
  i.Title_Changed_After_Estimation AS title_changed_after_estimation,
  i.Description_Changed_After_Estimation AS description_changed_after_estimation,
  i.Story_Point_Changed_After_Estimation AS story_point_changed_after_estimation,
  comp.components AS components
FROM Issue i
LEFT JOIN Project p ON p.ID = i.Project_ID
LEFT JOIN Repository r ON r.ID = p.Repository_ID
LEFT JOIN (
  SELECT
    ic.Issue_ID,
    GROUP_CONCAT(DISTINCT c.Name ORDER BY c.Name SEPARATOR '|') AS components
  FROM Issue_Component ic
  JOIN Component c ON c.ID = ic.Component_ID
  GROUP BY ic.Issue_ID
) comp ON comp.Issue_ID = i.ID
WHERE i.Title IS NOT NULL OR i.Description_Text IS NOT NULL;
" > "$OUT_DIR/tasks.tsv"

echo "Exporting positive task-link records..."
"$MYSQL_BIN" --defaults-extra-file="$CLIENT_CNF" --batch -e "
SELECT
  l.ID AS link_id,
  l.Issue_ID AS source_task_id,
  src.Issue_Key AS source_issue_key,
  l.Target_Issue_ID AS target_task_id,
  tgt.Issue_Key AS target_issue_key,
  l.Name AS relation_name,
  l.Description AS relation_description,
  l.Direction AS direction,
  src.Project_ID AS source_project_id,
  tgt.Project_ID AS target_project_id,
  src.Title AS source_title,
  src.Description_Text AS source_description_text,
  tgt.Title AS target_title,
  tgt.Description_Text AS target_description_text
FROM Issue_Link l
JOIN Issue src ON src.ID = l.Issue_ID
JOIN Issue tgt ON tgt.ID = l.Target_Issue_ID
WHERE l.Target_Issue_ID IS NOT NULL;
" > "$OUT_DIR/task_links.tsv"

echo "Exporting summary tables..."
"$MYSQL_BIN" --defaults-extra-file="$CLIENT_CNF" --batch -e "
SELECT COUNT(*) AS issue_count FROM Issue;
SELECT COUNT(*) AS issue_with_story_point_count FROM Issue WHERE Story_Point IS NOT NULL;
SELECT COUNT(*) AS issue_link_count FROM Issue_Link;
SELECT Type AS task_type, COUNT(*) AS count FROM Issue GROUP BY Type ORDER BY count DESC;
SELECT Priority AS priority, COUNT(*) AS count FROM Issue GROUP BY Priority ORDER BY count DESC;
SELECT Name AS relation_name, Description AS relation_description, Direction AS direction, COUNT(*) AS count
FROM Issue_Link
GROUP BY Name, Description, Direction
ORDER BY count DESC;
" > "$OUT_DIR/tawos_summary.tsv"

echo "Export complete:"
ls -lh "$OUT_DIR"/tasks.tsv "$OUT_DIR"/task_links.tsv "$OUT_DIR"/tawos_summary.tsv
