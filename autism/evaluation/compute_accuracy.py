import os
import json
import pandas as pd
from typing import Dict, List, Tuple
import re

def load_ground_truth(csv_path: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    df = pd.read_csv(csv_path)
    df.columns = [col.lower() for col in df.columns]
    segment_cols = [col for col in df.columns if col not in {'module', 'test_type', 'description', 'labels'}]

    gt_map = {}
    for _, row in df.iterrows():
        key = (row['module'], row['test_type'])
        seg_labels = {}

        for seg in segment_cols:
            val = row[seg]
            if pd.notna(val):
                if isinstance(val, int):
                    cleaned = str(val)
                elif isinstance(val, float):
                    cleaned = str(int(val))
                else:
                    cleaned = str(val)
                # Now remove non-alphanumeric characters from the final string
                cleaned = re.sub(r'\W+', '', cleaned)
                seg_labels[seg] = cleaned

        gt_map[key] = seg_labels

    return gt_map



def load_test_predictions(json_dir: str) -> List[Dict]:
    def parse_json(filepath: str, filename: str) -> List[Dict]:
        with open(filepath, 'r') as f:
            data = json.load(f)

        segment = os.path.splitext(filename)[0]  # remove .json extension  
        predictions = []
        for item in data:  # iterate through list of dictionaries
            meta_data = item['meta']
            # NOTE: Hard coded to ensure that we are only taking test results
            if 'module' not in meta_data or 'test_type' not in meta_data:
                continue
            else:
                module = item['meta']['module']
                test_type = item['meta']['test_type']
                pred = str(item['outputs'][0])
                predictions.append({
                    'segment': segment,
                    'module': module,
                    'test_type': test_type,
                    'pred': pred
                })
        return predictions

    all_predictions = []
    for fname in os.listdir(json_dir):
        if fname.endswith('.json'):
            filepath = os.path.join(json_dir, fname)
            all_predictions.extend(parse_json(filepath, fname))  # flatten list

    return all_predictions


def evaluate_predictions(predictions: List[Dict], ground_truth: Dict[Tuple[str, str], Dict[str, str]]) -> pd.DataFrame:
    def match(pred: Dict) -> Dict:
        # returns none if the segment cannot be found in the ground truth
        gt_label = ground_truth.get((pred['module'], pred['test_type']), {}).get(pred['segment'])
        return {
            **pred,
            'ground_truth': str(gt_label),
            'correct': str(gt_label) == pred['pred']
        }
    return pd.DataFrame(map(match, predictions))

def is_valid(val):
    return pd.notna(val) and val is not None and val != 'None'


def compute_accuracies(results_df: pd.DataFrame) -> Tuple[float, pd.Series, pd.Series, pd.Series]:
    print("evaluating results for the dataframe", results_df.head(5))

    # this should handle cases where the ground truth for a test 
    # is not filled up because it is not applicable
    # this should also handle cases where we have not annotated 
    # a test segment

    overall = results_df['correct'].mean()
    by_module = results_df.groupby('module')['correct'].mean()
    by_test = results_df.groupby('test_type')['correct'].mean()
    by_segment = results_df.groupby('segment')['correct'].mean()

    return overall, by_module, by_test, by_segment

