#!/bin/bash

# ---------- CONFIGURATION ----------
# Debug mode: if true, break after first segment
# DEBUG_BREAK_AFTER_FIRST=true

# Directory of segments (each segment has video + frames)
SEGMENTS_DIR="/scratch/keane/human_behaviour_data/ados_videos_and_frames/sample_1"

# Rubrics JSON file (for module + test_type prompts)
RUBRICS_JSON="/home/keaneong/human-behavior/data/autism/ados_prompts/autism_scoring.json"

# Path to inference.py
INFERENCE_SCRIPT="/home/keaneong/human-behavior/autism/inference/_qwen_vl_inference.py"

# Output directory
OUTPUT_DIR="/home/keaneong/human-behavior/data/autism/ados_outputs/"

# If running non test prompt=True, we will not run the test prompt
# otherwise we will be running the test prompts
NON_TEST_PROMPT_TXT="/home/keaneong/human-behavior/autism/inference/non_test_prompt.txt"
RUN_NON_TEST_PROMPT=true

# Set this flag to true to run ALL module + test_type from rubrics_json
# if this is true, we will not run the non test prompt
RUN_ALL_TESTS=false

# If RUN_ALL_TESTS=true, you can EXCLUDE specific module/test_type below:
# if you want to exclude the entire module, set EXCLUDED_MODULES=("Module Name")
# otherwise, you can exclude specific test types by setting EXCLUDED_TEST_TYPES=("Test Type Name")
EXCLUDED_MODULES=("")
EXCLUDED_TEST_TYPES=("C2. Imagination/Creativity")

# If RUN_ALL_TESTS=false, set the specific test you want to run:
SINGLE_TEST_MODULE="B: Reciprocal Social Interaction"
SINGLE_TEST_TYPE="B3. Facial Expressions Directed to Others"

# If you want to run a specific segment only, set here:
# (Set SPECIFIC_SEGMENT_NAME="" to run all segments)
SPECIFIC_SEGMENT_NAME=""

# ---------- MAIN LOOP ----------

# Read the rubrics JSON into a list of module + test_type
RUBRICS=$(jq -c '.[]' "$RUBRICS_JSON")

# Loop through segments
for SEGMENT_PATH in "$SEGMENTS_DIR"/*; do
    if [ -d "$SEGMENT_PATH" ]; then
        SEGMENT_NAME=$(basename "$SEGMENT_PATH")

        # If SPECIFIC_SEGMENT_NAME is set, skip others
        if [ -n "$SPECIFIC_SEGMENT_NAME" ] && [ "$SEGMENT_NAME" != "$SPECIFIC_SEGMENT_NAME" ]; then
            continue
        fi

        echo "Processing segment: $SEGMENT_NAME"

        if [ "$RUN_NON_TEST_PROMPT" = true ]; then
            # Run manual prompt mode
            python "$INFERENCE_SCRIPT" \
                --config "" \
                --test_segment_dir "$SEGMENT_PATH" \
                --use_video_or_frames "video" \
                --non_test_prompt "$NON_TEST_PROMPT_TXT" \
                --output_path "$OUTPUT_DIR/${SEGMENT_NAME}.json"
        else
            if [ "$RUN_ALL_TESTS" = true ]; then
                # Loop over all entries in the rubrics
                echo "$RUBRICS" | while IFS= read -r entry; do
                    MODULE=$(echo "$entry" | jq -r '.module')
                    TEST_TYPE=$(echo "$entry" | jq -r '."test type"')

                    # Check if excluded
                    if printf '%s\n' "${EXCLUDED_MODULES[@]}" | grep -qFx "$MODULE"; then
                        echo "Skipping EXCLUDED MODULE: $MODULE"
                        continue
                    fi
                    if printf '%s\n' "${EXCLUDED_TEST_TYPES[@]}" | grep -qFx "$TEST_TYPE"; then
                        echo "Skipping EXCLUDED TEST_TYPE: $TEST_TYPE"
                        continue
                    fi

                    echo "Running: MODULE='$MODULE' TEST_TYPE='$TEST_TYPE' on segment '$SEGMENT_NAME'"

                    # Run inference
                    python "$INFERENCE_SCRIPT" \
                        --config "" \
                        --test_segment_dir "$SEGMENT_PATH" \
                        --use_video_or_frames "video" \
                        --rubrics_json "$RUBRICS_JSON" \
                        --module "$MODULE" \
                        --test_type "$TEST_TYPE" \
                        --output_path "$OUTPUT_DIR/${SEGMENT_NAME}.json"

                    # if [ "$DEBUG_BREAK_AFTER_FIRST" = true ]; then
                    #     echo "DEBUG: breaking after first iteration."
                    #     break
                    # fi
                    
                done
            else
                # Single test mode: you can set these manually below:
    
                echo "Running: MODULE='$SINGLE_TEST_MODULE' TEST_TYPE='$SINGLE_TEST_TYPE' on segment '$SEGMENT_NAME'"

                python "$INFERENCE_SCRIPT" \
                    --config "" \
                    --test_segment_dir "$SEGMENT_PATH" \
                    --use_video_or_frames "video" \
                    --rubrics_json "$RUBRICS_JSON" \
                    --module "$MODULE" \
                    --test_type "$TEST_TYPE" \
                    --output_path "$OUTPUT_DIR/${SEGMENT_NAME}__${TEST_TYPE// /_}.json"
            fi
        
        fi
    fi
done