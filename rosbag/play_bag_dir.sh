#!/usr/bin/env bash

BAG_ROOT="$HOME/boats_dataset_processing/bags/0610_JS5"

TOPICS_CONFIG_PATH="$HOME/boats_dataset_processing/config/topics-raw-camera.txt"

if [ $# -gt 0 ]; then
    BAG_ROOT="$1"
    echo "Bag file set to: $BAG_ROOT"
else
    echo "No bag file specified, using default: $BAG_ROOT"
fi

if [ $# -gt 1 ]; then
    TOPICS_CONFIG_PATH="$2"
    echo "Topics config file set to: $TOPICS_CONFIG_PATH"
else
    echo "No topics config file specified, using default: $TOPICS_CONFIG_PATH"
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

# Create an array to store all bag paths
mapfile -t BAG_PATHS < <(find "$BAG_ROOT" -type f -name "*.bag" | sort)

# Play all bags in sequence
for BAG_PATH in "${BAG_PATHS[@]}"; do
    echo "===================================="
    echo "Playing: $BAG_PATH"
    echo "===================================="

    # Run rosbag play in the background
    # rosbag play "$BAG_PATH" -r 3 --clock -q 
    rosbag play "$BAG_PATH" --topics "${TOPICS[@]}"
    
    # Wait for a short time to ensure the bag starts playing
    # sleep 2
done

