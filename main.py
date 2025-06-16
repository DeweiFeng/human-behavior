from loader import VideoDatasetLoader

# Initialize the dataset loader
loader = VideoDatasetLoader("datasets/MELD.Raw")

# Load the test split
test_data = loader.get_test_split()

# Print information about the first example
if test_data:
    first_example = test_data[0]
    print("First example:")
    print(f"Filename: {first_example['filename']}")
    print(f"Video size: {len(first_example['video'])} bytes")
    print(f"Label: {first_example['label']} ({loader.get_emotion_name(first_example['label'])})")

# Print dataset statistics
print("\nDataset statistics:")
print(f"Number of test examples: {len(test_data)}")

# Count examples per emotion
emotion_counts = {}
for example in test_data:
    emotion = loader.get_emotion_name(example['label'])
    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

print("\nEmotion distribution in test set:")
for emotion, count in sorted(emotion_counts.items()):
    print(f"{emotion}: {count} examples")

# Load dev split if needed
dev_data = loader.get_dev_split()
print(f"\nNumber of dev examples: {len(dev_data)}")
