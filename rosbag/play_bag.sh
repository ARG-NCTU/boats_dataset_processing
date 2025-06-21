#!/usr/bin/env bash

BAG_PATH="/home/arg/boats_dataset_processing/bags/0610_JS5/0610_1145/_2025-06-10-11-45-30_0.bag"

TOPICS_CONFIG_PATH="$HOME/boats_dataset_processing/config/topics-raw-camera.txt"

if [ $# -gt 0 ]; then
    BAG_PATH="$1"
    echo "Bag file set to: $BAG_PATH"
else
    echo "No bag file specified, using default: $BAG_PATH"
fi

if [ $# -gt 1 ]; then
    TOPICS_CONFIG_PATH="$2"
    echo "Topics config file set to: $TOPICS_CONFIG_PATH"
else
    echo "No topics config file specified, using default: $TOPICS_CONFIG_PATH"
fi

# Check if the bag file exists
if [ ! -f "$BAG_PATH" ]; then
    echo "Bag file not found: $BAG_PATH"
    exit 1
fi

# Read topics into array, skip empty lines and comments
TOPICS=()
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  TOPICS+=("$line")
done < "$TOPICS_CONFIG_PATH"

# Print topics to be played
echo "✅ Playing the following topics:"
printf ' - %s\n' "${TOPICS[@]}"

# Play the bag file at 30x speed with specified topics
rosbag play "$BAG_PATH" --topics "${TOPICS[@]}"
