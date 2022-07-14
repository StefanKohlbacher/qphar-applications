from argparse import ArgumentParser

import pandas as pd
import numpy as np


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-compareAgainst', required=True,
                        help='results to compare shared pharmacophore screening results against. xlsx file',
                        )
    parser.add_argument('-out', required=True, help='output file')
    return parser.parse_args()


def loadResultsXlsx(path: str) -> pd.DataFrame:
    results: pd.DataFrame = pd.read_excel(path)
    results.dropna(axis=0, how='all', inplace=True)
    results.columns = pd.Series(np.where(np.array(['Unnamed' not in col for col in results.columns]),
                                         results.columns,
                                         None)).fillna(method='ffill')
    return results


def getBestRun(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score run by average of Fake-F1-Score and F-Beta-Score.

    Returns fakeF1Score of df with best parameters.
    :return:
    """
    fakeF1Score = df[[None, 'FAKE_F1_SCORE']]
    fakeF1Score.columns = ['Target', *fakeF1Score.iloc[0].values[1:]]
    fakeF1Score.drop(0, axis=0, inplace=True)
    fakeF1Score.set_index('Target', inplace=True, drop=True)

    fbetaScore = df[[None, 'FBETA_SCORE']]
    fbetaScore.columns = ['Target', *fbetaScore.iloc[0].values[1:]]
    fbetaScore.drop(0, axis=0, inplace=True)
    fbetaScore.set_index('Target', inplace=True, drop=True)
        
    scores = ((fakeF1Score + fbetaScore) / 2).round(3)  # adds columns with same name
    cols = scores.columns.values
    scores.fillna(0, inplace=True)
    scores['best'] = scores.apply(lambda x: np.argmax([x[col] for col in cols]), axis=1)
    scores['bestNames'] = scores['best'].apply(lambda x: cols[x])
    scores['bestScores'] = scores.apply(lambda x: x[x['bestNames']], axis=1)
    return scores[['bestNames', 'bestScores']]


def main(otherResultsPath: str, outputFile: str):
    sharedResults = loadResultsXlsx('./data/sharedPharmacophores/screeningResults.xlsx')
    otherResults = loadResultsXlsx(otherResultsPath)

    # filter best run
    scoredSharedResults = getBestRun(sharedResults)
    scoredOtherResults = getBestRun(otherResults)

    # compare and merge
    scoredSharedResults.rename(columns={'bestNames': 'sharedResults', 'bestScores': 'sharedResultsScores'}, inplace=True)
    scoredOtherResults.rename(columns={'bestNames': 'otherResults', 'bestScores': 'otherResultScores'}, inplace=True)
    merged = pd.concat([scoredSharedResults, scoredOtherResults], axis=1)
    merged.to_csv(outputFile)


if __name__ == '__main__':
    args = parseArgs()
    main(args.compareAgainst, args.out)
