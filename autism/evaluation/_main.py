from compute_accuracy import load_ground_truth, load_test_predictions, evaluate_predictions, compute_accuracies, is_valid
from visualize_accuracy import plot_segment_accuracy, plot_module_accuracy
import os

def main(json_dir: str, csv_path: str, output_dir: str):
    ground_truth = load_ground_truth(csv_path)
    predictions = load_test_predictions(json_dir)
    results_df = evaluate_predictions(predictions, ground_truth)

    filtered_results_df = results_df[
        results_df['pred'].apply(is_valid) & results_df['ground_truth'].apply(is_valid)
    ].copy()

    print(f"Filtered down from {len(results_df)} to {len(filtered_results_df)} rows.")

    # save the filtered results to a CSV file
    filtered_csv_path = os.path.join(output_dir, "results.csv")
    filtered_results_df.to_csv(filtered_csv_path, index=False)
    print(f"Filtered results saved to {filtered_csv_path}")

    overall, by_module, by_test, by_segment = compute_accuracies(filtered_results_df)

    print(f"\nOverall Accuracy: {overall:.3f}\n")
    
    print("Accuracy by Module:")
    print(by_module.to_string())

    print("\nAccuracy by Test Type:")
    print(by_test.to_string())

    plot_segment_accuracy(by_segment, output_dir=output_dir, filetype="pdf")
    plot_module_accuracy(by_module, output_dir=output_dir, filetype="pdf")

if __name__ == "__main__":
    JSON_DIR = "/home/human-behavior/autism/data/ados_outputs"
    CSV_PATH = "/home/human-behavior/autism/data/ados_outputs/ados_scoring_sheet.csv"
    OUTPUT_DIR = "/home/human-behavior/autism/data/ados_scores/"
    main(JSON_DIR, CSV_PATH, OUTPUT_DIR)