from argparse import ArgumentParser
import os
from typing import Dict, List, Tuple
import json

import pandas as pd

OUTPUT_METRICS = {
  'tp',
  'fp',
  'tn',
  'fn',
  'FAKE_F1_SCORE',
  'FBETA_SCORE',
  'KENDALLS_TAU',
  'SPEARMAN_R'
}


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-folder', required=True, help='folder containing multiple screening runs')
    return parser.parse_args()


def main(folder: str):
    if not folder.endswith('/'):
        folder = '{}/'.format(folder)

    results: List[pd.DataFrame] = []
    for metric in os.listdir(folder):
        if not os.path.isdir('{}{}'.format(folder, metric)):
            continue

        targetResults: Dict[str, Dict[Tuple[str, str], float]] = {}
        for target in os.listdir('{}{}'.format(folder, metric)):
            if not os.path.isdir('{}{}/{}'.format(folder, metric, target)):
                continue

            with open('{}{}/{}/0/test_performance.json'.format(folder, metric, target), 'r') as f:
                r = json.load(f)

            targetResults[target] = {(k, metric): v for k, v in r.items() if k in OUTPUT_METRICS}

        results.append(pd.DataFrame.from_dict(targetResults, orient='index'))

    results: pd.DataFrame = pd.concat(results, axis=1).round(2)
    results: pd.DataFrame = results.T.sort_index(axis=0, level=0).T.sort_index(axis=0, ascending=True)
    results.to_excel('{}screeningResults.xlsx'.format(folder), merge_cells=True)


if __name__ == '__main__':
    main(parseArgs().folder)
