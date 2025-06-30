#!/bin/bash

# ---------- CONFIGURATION ----------
SEGMENTS_DIR="/scratch/keane/human_behaviour_data/ados_videos_and_frames/sample_1"
# Specify multiple segments by name (no commas!)
# SPECIFIC_SEGMENT_NAMES=("response_to_joint_attention_1" "response_to_name" "responsive_social_smile_1" "responsive_social_smile_2" "responsive_social_smile_3" "responsive_social_smile_4")
SPECIFIC_SEGMENT_NAMES=("bubble_play")
RUN_PARALLEL=true   # Set to false to run serially

RUBRICS_JSON="/home/keaneong/human-behavior/data/autism/ados_prompts/autism_scoring.json"
INFERENCE_SCRIPT="/home/keaneong/human-behavior/autism/inference/_qwen_vl_inference.py"
OUTPUT_DIR="/home/keaneong/human-behavior/data/autism/ados_outputs/"

# If running non test prompt=True, we will not run the test prompt
# otherwise we will be running the test prompts
NON_TEST_PROMPT_TXT="/home/keaneong/human-behavior/autism/inference/non_test_prompt.txt"
RUN_NON_TEST_PROMPT=false

# Set this flag to true to run ALL module + test_type from rubrics_json
# if this is true, we will not run the non test prompt
RUN_ALL_TESTS=false

# If RUN_ALL_TESTS=true, you can EXCLUDE specific module/test_type below:
# if you want to exclude the entire module, set EXCLUDED_MODULES=("Module Name")
# otherwise, you can exclude specific test types by setting EXCLUDED_TEST_TYPES=("Test Type Name")
EXCLUDED_MODULES=("")
EXCLUDED_TEST_TYPES=("")

# If RUN_ALL_TESTS=false, set the specific test you want to run:
SINGLE_TEST_MODULE="E: Other Abnormal Behaviors"
SINGLE_TEST_TYPE="E2. Tantrums, Aggression, Negative or Disruptive Behavior"

RUBRICS=$(jq -c '.[]' "$RUBRICS_JSON")

contains_element() {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

process_segment() {
    local SEGMENT_PATH="$1"
    local SEGMENT_NAME
    SEGMENT_NAME=$(basename "$SEGMENT_PATH")

    echo "Processing segment: $SEGMENT_NAME"

    if [ "$RUN_NON_TEST_PROMPT" = true ]; then
        python "$INFERENCE_SCRIPT" \
            --config "" \
            --test_segment_dir "$SEGMENT_PATH" \
            --use_video_or_frames "video" \
            --non_test_prompt "$NON_TEST_PROMPT_TXT" \
            --output_path "$OUTPUT_DIR/${SEGMENT_NAME}.json"
    else
        if [ "$RUN_ALL_TESTS" = true ]; then
            while IFS= read -r entry; do
                MODULE=$(echo "$entry" | jq -r '.module')
                TEST_TYPE=$(echo "$entry" | jq -r '."test type"')
                if printf '%s\n' "${EXCLUDED_MODULES[@]}" | grep -qFx "$MODULE"; then
                    echo "Skipping EXCLUDED MODULE: $MODULE"
                    continue
                fi
                if printf '%s\n' "${EXCLUDED_TEST_TYPES[@]}" | grep -qFx "$TEST_TYPE"; then
                    echo "Skipping EXCLUDED TEST_TYPE: $TEST_TYPE"
                    continue
                fi

                echo "Running: MODULE='$MODULE' TEST_TYPE='$TEST_TYPE' on segment '$SEGMENT_NAME'"
                python "$INFERENCE_SCRIPT" \
                    --config "" \
                    --test_segment_dir "$SEGMENT_PATH" \
                    --use_video_or_frames "video" \
                    --rubrics_json "$RUBRICS_JSON" \
                    --module "$MODULE" \
                    --test_type "$TEST_TYPE" \
                    --output_path "$OUTPUT_DIR/${SEGMENT_NAME}.json"
            done <<< "$RUBRICS"
        else
            echo "Running: MODULE='$SINGLE_TEST_MODULE' TEST_TYPE='$SINGLE_TEST_TYPE' on segment '$SEGMENT_NAME'"
            python "$INFERENCE_SCRIPT" \
                --config "" \
                --test_segment_dir "$SEGMENT_PATH" \
                --use_video_or_frames "video" \
                --rubrics_json "$RUBRICS_JSON" \
                --module "$SINGLE_TEST_MODULE" \
                --test_type "$SINGLE_TEST_TYPE" \
                --output_path "$OUTPUT_DIR/${SEGMENT_NAME}.json"
        fi
    fi
}

for SEGMENT_PATH in "$SEGMENTS_DIR"/*; do
    if [ -d "$SEGMENT_PATH" ]; then
        SEGMENT_NAME=$(basename "$SEGMENT_PATH")
        if [ "${#SPECIFIC_SEGMENT_NAMES[@]}" -ne 0 ]; then
            contains_element "$SEGMENT_NAME" "${SPECIFIC_SEGMENT_NAMES[@]}"
            [ $? -ne 0 ] && continue
        fi

        if [ "$RUN_PARALLEL" = true ]; then
            process_segment "$SEGMENT_PATH" &
        else
            process_segment "$SEGMENT_PATH"
        fi
    fi
done

if [ "$RUN_PARALLEL" = true ]; then
    wait
fi
