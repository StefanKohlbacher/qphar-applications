"""
Create a baseline model for virtual pharmacophore screening. The model is used to screen a validation set and the hits
are saved for comparison and ranking later on.

The baseline model is generated as a shared pharmacophore from the top n molecules in the training dataset, whereas the
top n are defined my their ranking of activity values.
"""
import logging
import os
from argparse import ArgumentParser
from typing import List, Union, Set, Tuple, Dict

import CDPL.Chem as Chem
import CDPL.Pharm as Pharm
from src.molecule_tools import mol_to_sdf
from src.pharmacophore_tools import loadPharmacophore
from src.qphar import LOOKUPKEYS

from makeOptimalPharmacophore import Screener, Metric, loadReferenceDataset, getScreeningPerformance, saveTopNResults, \
    saveTestResults

IDB_GEN_PATH = '/data/shared/software/ligandscout4.4.3/idbgen'
I_SCREEN_PATH = '/data/shared/software/ligandscout4.4.3/iscreen'
ESPRESSO_PATH = '/data/shared/software/ligandscout4.4.3/espresso'


class Parameters:

    def __init__(self,
                 nrMolecules: int = 3,
                 findBestParameters: bool = True,
                 referenceDataset: str = None,
                 referenceDbPath: str = None,
                 testDataset: str = None,
                 testDbPath: str = None,
                 activityName: str = None,
                 metric: Metric = Metric.ACCURACY,
                 outputPath: str = None,
                 beta: float = 1
                 ):
        self.nrMolecules = nrMolecules
        self.findBestParameters = findBestParameters
        self.referenceDataset = referenceDataset
        self.referenceDbPath = referenceDbPath
        self.testDataset = testDataset
        self.testDbPath = testDbPath
        self.activityName = activityName
        self.metric = metric
        self.outputPath = outputPath
        self.beta = beta


class LsScreener(Screener):

    def __init__(self,
                 dataset: Union[str, List[Chem.BasicMolecule], None] = None,
                 pharmacophore: Union[str, Pharm.BasicPharmacophore, None] = None,
                 hitPath: str = None,
                 ):
        super(LsScreener, self).__init__(dataset=dataset,
                                         pharmacophore=pharmacophore,
                                         hitPath=hitPath,
                                         screeningDbFileFormat='.ldb',
                                         dbGeneratorPath=IDB_GEN_PATH,
                                         dbScreenerPath=I_SCREEN_PATH,
                                         )


class SharedPharmacophoreGenerator:

    tempMolPath = '/tmp/topNMolecules.sdf'
    tempPharmPath = '/tmp/sharedPharmacophore.pml'

    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    def run(self, molecules: List[Chem.BasicMolecule]) -> None:
        """
        Create a shared pharmacophore from the n most active molecules (as defined in parameters) with Ligandscout
        and save the result to a pml file.
        :return:
        """
        logging.info('Sorting molecules by their activity: {}'.format(self.parameters.activityName))

        def sortingFn(mol: Chem.BasicMolecule) -> float:
            return mol.getProperty(LOOKUPKEYS['activity'])

        molecules.sort(key=sortingFn)

        mol_to_sdf(molecules[-self.parameters.nrMolecules:],
                   self.tempMolPath,
                   multiconf=True)

        self.generatedSharedPharmacophoreFromSdf(self.tempMolPath, self.tempPharmPath)

    @staticmethod
    def generatedSharedPharmacophoreFromSdf(inputPath: str, outputPath: str) -> None:
        os.system('{} -t {} -l /tmp/sharedPharmacophoreGeneration.log -p {} -g shared -c import'.format(ESPRESSO_PATH, inputPath, outputPath))

    @staticmethod
    def getPharmacophoreOutputPath() -> str:
        return '{}-1.pml'.format(os.path.splitext(SharedPharmacophoreGenerator.tempPharmPath)[0])


def parseArgs() -> Parameters:
    parser = ArgumentParser()
    parser.add_argument('-output', required=True, type=str, help='folder where shared pharmacophore should be saved')
    parser.add_argument('-findBestParams', action='store_true', default=False, required=False,
                        help='indicates whether to optimize parameters for generating shared pharmacophore')
    parser.add_argument('-nrMolecules', default=3, required=False, type=int,
                        help='nr of molecules to use for shared pharmacophore generation')
    parser.add_argument('-referenceDataset', required=True,
                        help='path of sdf file containing molecules as reference / trainign dataset')
    parser.add_argument('-referenceDbPath', required=False, default=None,
                        help='path of generated screening database from reference dataset')
    parser.add_argument('-testDataset', required=False, default=None,
                        help='sdf file containing molecules to be used as test set')
    parser.add_argument('-testDbPath', required=False, default=None,
                        help='path of generated screening database from test dataset')
    parser.add_argument('-activityName', required=True,
                        help='name of activity property in reference dataset')
    parser.add_argument('-metric', required=False, default=Metric.ACCURACY.value,
                        help='metric to score screening performance of pharmacophore. One of [{}]'.format(
                            e.value for e in Metric))
    parser.add_argument('-beta', required=False, default=1, type=float, help='beta in f-score')
    parser.add_argument('-logLevel', required=False, default='INFO', type=str,
                        help='granularity level of logging')
    args = parser.parse_args()

    logging.getLogger().setLevel(level=logging.getLevelName(args.logLevel))
    logging.basicConfig(level=logging.getLevelName(args.logLevel))

    metric: Metric
    try:
        metric = Metric[args.metric]
    except KeyError:
        raise KeyError('{} not a member of Metric. Must be one of {}'.format(args.metric, [e.value for e in Metric]))

    if args.findBestParams:
        if args.referenceDataset is None or args.referenceDbPath is None or args.activityName is None:
            raise ValueError('When optimising parameters, "referenceDataset", "referenceDbPath", and "activityName" need to be provided. {} was given'.format(
                args.referenceDataset, args.referenceDbPath, args.activityName
            ))

    if args.testDbPath is not None and args.testDataset is None:
        raise ValueError('-testDataset needs to be provided if -testDbPath is provided.')

    return Parameters(findBestParameters=args.findBestParams,
                      nrMolecules=args.nrMolecules,
                      referenceDataset=args.referenceDataset,
                      referenceDbPath=args.referenceDbPath,
                      testDataset=args.testDataset,
                      testDbPath=args.testDbPath,
                      activityName=args.activityName,
                      metric=metric,
                      outputPath=args.output if args.output.endswith('/') else '{}/'.format(args.output),
                      beta=args.beta
                      )


def runParameterOptimization(params: Parameters) -> None:
    molecules = loadReferenceDataset(params.referenceDataset, params.activityName)
    testMolecules = loadReferenceDataset(params.testDataset,
                                         params.activityName) if params.testDataset is not None else None

    screener = LsScreener(dataset=params.referenceDbPath)
    screeningResults: Dict[int, Dict[str, Union[Pharm.BasicPharmacophore, List[Chem.BasicMolecule]]]] = {}

    for n in range(3, 10):  # n most active molecules to use for shared pharmacophore generation
        logging.info('Creating shared pharmacohpore from {} most active molecules'.format(n))
        params.nrMolecules = n

        sharedPharmGenerator = SharedPharmacophoreGenerator(params)
        sharedPharmGenerator.run(molecules)

        screener.setPharmacophore(SharedPharmacophoreGenerator.getPharmacophoreOutputPath())
        screener.run()

        sharedPharm = loadPharmacophore(SharedPharmacophoreGenerator.getPharmacophoreOutputPath())
        hits = screener.loadHits(params.activityName)
        screeningResults[n] = {
            'pharmacophore': sharedPharm,
            'hits': hits,
            'parameters': params,
        }

    screeningPerformance = getScreeningPerformance(molecules, screeningResults, params.metric, params.beta)
    saveTopNResults(len(screeningResults), params.outputPath, screeningPerformance, screeningResults)

    # if test set given, evaluate top n on test set
    if testMolecules is None or params.testDbPath is None:
        return

    logging.info('Evaluating shared pharmacophores on test set')
    screener = LsScreener(dataset=params.testDbPath)
    testScreeningResults: Dict[int, Dict[str, List[Chem.BasicMolecule]]] = {}
    for i, row in screeningPerformance.iterrows():
        pharmacophore = screeningResults[row['index']]['pharmacophore']
        screener.setPharmacophore(pharmacophore)
        screener.run()
        hits = screener.loadHits(params.activityName)
        testScreeningResults[i] = {
            'hits': hits,
        }

    testScreeningPerformance = getScreeningPerformance(testMolecules, testScreeningResults, params.metric, params.beta)
    saveTestResults(params.outputPath, testScreeningPerformance, testScreeningResults)


def main(params: Parameters) -> None:
    molecules = loadReferenceDataset(params.referenceDataset, params.activityName)

    generator = SharedPharmacophoreGenerator(params)
    generator.run(molecules)


if __name__ == '__main__':
    params = parseArgs()

    if params.findBestParameters:
        runParameterOptimization(params)
    else:
        main(params)
