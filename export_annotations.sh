#!/bin/bash

current_date=$(date +"%Y-%m-%d-%H-%M")
random_string=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

export $(cat .env | xargs)

curl -X GET \
    "$LABEL_STUDIO_URL/api/projects/5/export?exportType=JSON_MIN" \
    -H "Authorization: Token $API_KEY" \
    --output "project-5-at-$current_date-$random_string.json"

curl -X GET \
    "$LABEL_STUDIO_URL/api/projects/5/export?exportType=JSON" \
    -H "Authorization: Token $API_KEY" \
    --output "project-5-full-at-$current_date-$random_string.json"

echo "project-5-at-$current_date-$random_string.json"
echo "project-5-full-at-$current_date-$random_string.json"
