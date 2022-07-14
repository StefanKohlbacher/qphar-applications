"""
Overview

Given a trained qphar, we can deduct the influence of each feature on the activity of a sample. For example does the
feature contribute positively or negatively to the activity? Given this knowledge and the position of the features,
we can deduct an optimal pharmacophore, which yields the highest expected activity based on the model trained with this
data.
We would expect this optimal pharmacophore to be very similar to truly active samples and not similar to less active
samples. With regards to alignment, we would expect to some extend that the less active samples cannot be aligned with
this pharmacophore. This directly relates to virtual pharmacophore screening, which requires alignment of samples.
Therefore, we would expect this pharmacophore fulfill certain following properties:
    - combine features of multiple highly active samples
    - show increased enrichment of active compounds over inactive compounds when used for screening
    - show better enrichment / selectivity in virtual screening than a single pharmacophore from a randomly selected
        active sample
    - show better enrichment / selectivity in virtual screening than a simple merged / shared pharmacophore obtained
        from a selection of active compounds

The optimal pharmacophore will be designed with the following procedure:
    - load and extract trained ml model from qphar-model
    - get importance and contribution of each pharmacophore-feature to activity based on trained ml model
        (feature importance of random forest, coefficients in linear regression)
    - remove features contributing negatively (alternatively place xvols)
    - [optional] adjust size of positively contributing features according to feature importance
"""
from argparse import ArgumentParser
import os
import json
import logging
from typing import List, Set, Dict, Union, Tuple
from enum import Enum
from hashlib import md5

import pandas as pd
import numpy as np
import CDPL.Chem as Chem
import CDPL.Pharm as Pharm
import CDPL.Base as Base
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau, spearmanr

from src.qphar import Qphar, LOOKUPKEYS
from src.pharmacophore_tools import savePharmacophore
from src.molecule_tools import SDFReader, mol_to_sdf

from visualizeActivityGrid import Parameters as VisualizerParameters, Visualizer, FEATURE_TYPES

# enable type analysis for IDE -> not necessary in production
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.cross_decomposition import PLSRegression


PSD_DB_CREATOR_PATH = '/data/shared/software/CDPKit/Apps/psdcreate'
PSD_DB_SCREENER_PATH = '/data/shared/software/CDPKit/Apps/psdscreen'

HASH_LOOKUPKEY = Base.LookupKey.create('id')
FEATURE_CONTRIBUTION_LOOKUPKEY = Base.LookupKey.create('featureContribution')
FEATURE_IMPORTANCE_LOOKUPKEY = Base.LookupKey.create('featureImportance')


class RfContributionType(Enum):
    AVG_PREDICTION = 'AVG_PREDICTION'
    BINARY_THRESHOLD = 'BINARY_THRESHOLD'
    CONTINUOUS_THRESHOLD = 'CONTINUOUS_THRESHOLD'


class Metric(Enum):
    ACCURACY = 'ACCURACY'
    PRECISION = 'PRECISION'
    SENSITIVITY = 'SENSITIVITY'
    SPECIFICITY = 'SPECIFICITY'
    BALANCED_ACCURACY = 'BALANCED_ACCURACY'
    F1_SCORE = 'F1_SCORE'
    FAKE_F1_SCORE = 'FAKE_F1_SCORE'
    FBETA_SCORE = 'FBETA_SCORE'
    ROC_AUC = 'ROC_AUC'
    KENDALLS_TAU = 'KENDALLS_TAU'
    SPEARMAN_R = 'SPEARMAN_R'


class Metrics:

    def __init__(self, tp: int, fp: int, tn: int, fn: int):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def calcAccuracy(self) -> float:
        s = self.tp + self.fp + self.tn + self.fn
        return 0 if s == 0 else (self.tp + self.tn) / s

    def calcPrecision(self) -> float:
        s = self.tp + self.fp
        return 0 if s == 0 else self.tp / s

    def calcSensitivity(self) -> float:  # aka recall
        s = self.tp + self.fn
        return 0 if s == 0 else self.tp / s

    def calcSpecificity(self) -> float:
        s = self.tn + self.fp
        return 0 if s == 0 else self.tn / s

    def calcBalancedAccuracy(self) -> float:
        return (self.calcSensitivity() + self.calcSpecificity()) / 2

    def calcF1Score(self) -> float:
        s = self.tp + (self.fp + self.fn) * 0.5
        return 0 if s == 0 else self.tp / s

    def calcFakeF1Score(self) -> float:
        precision = self.calcPrecision()
        specificity = self.calcSpecificity()
        if precision == 0 or specificity == 0:
            return 0
        return 2 / ((1/precision) + (1/specificity))

    def calcFBetaScore(self, beta: float = 1) -> float:
        s = (1 + beta**2) * self.tp + beta**2 * self.fn + self.fp
        return 0 if s == 0 else (1 + beta**2) * self.tp / s

    @staticmethod
    def calcRocAucScore(yTrue: np.ndarray, yPred: np.ndarray) -> float:
        return roc_auc_score(yTrue, yPred)


class Parameters:

    def __init__(self,
                 modelPath: str = None,
                 outputPath: str = None,
                 weightFeaturesByImportance: bool = False,
                 weightFeaturesAtScreening: bool = False,
                 setXvols: bool = False,
                 featureContributionFromMl: bool = True,
                 minFeatureWeight: float = 0.5,
                 maxFeatureWeight: float = 1,
                 nrOptimalFeatures: int = 6,
                 findBestParameters: bool = False,
                 rfContributionType: RfContributionType = RfContributionType.BINARY_THRESHOLD,
                 screeningDbPath: str = None,
                 referenceDatasetPath: str = None,
                 activityName: str = None,
                 metric: Metric = Metric.ACCURACY,
                 nrTopResults: int = 5,
                 screeningTestDbPath: str = None,
                 referenceTestData: str = None,
                 beta: float = 1
                 ):
        self.modelPath = modelPath
        self.outputPath = outputPath
        self.weightFeaturesByImportance = weightFeaturesByImportance
        self.weightFeaturesAtScreening = weightFeaturesAtScreening
        self.setXvols = setXvols
        self.featureContributionFromMl = featureContributionFromMl
        self.minFeatureWeight = minFeatureWeight
        self.maxFeatureWeight = maxFeatureWeight
        self.nrOptimalFeatures = nrOptimalFeatures
        self.findBestParameters = findBestParameters
        self.rfContributionType = rfContributionType
        self.screeningDbPath = screeningDbPath
        self.referenceDatasetPath = referenceDatasetPath
        self.activityName = activityName
        self.metric = metric
        self.nrTopResults = nrTopResults
        self.screeningTestDbPath = screeningTestDbPath
        self.referenceTestData = referenceTestData
        self.beta = beta


class OptimalPharmacophoreGenerator:

    def __init__(self,
                 parameters: Parameters,
                 ):
        self.parameters = parameters
        self.featureImportanceScaler = MinMaxScaler((parameters.minFeatureWeight, parameters.maxFeatureWeight))
        self.featureImportanceScaler.fit([[0], [1]])

    def run(self, qpharModel: Qphar) -> Union[Pharm.BasicPharmacophore, None]:
        """
        
        :return: 
        """
        logging.info('Generating optimal pharmacophore for modelType {}'.format(qpharModel.modelType))
        qpharPharmacohpore = Pharm.BasicPharmacophore()
        qpharPharmacohpore.assign(qpharModel.cleanedHP)
        mlModel: Union[RandomForestRegressor, Ridge, Lasso, LinearRegression, PLSRegression] = qpharModel.mlModel

        # get feature importance
        featureImportance: List[float]
        if qpharModel.modelType == 'randomForest':
            featureImportance = mlModel.feature_importances_.tolist()
        else:
            raise NotImplementedError('Retrieval of feature importance for {} is currently not supported'.format(
                qpharModel.modelType))
        if featureImportance is None:
            raise ValueError('featureImportance was None, but should be numpy.ndarray')

        # calculate feature contribution
        featureContribution: List[float]
        if self.parameters.featureContributionFromMl:
            featureContribution = self.featureContributionFromMl(qpharModel, mlModel)
        else:
            featureContribution = self.featureContributionFromQphar(qpharPharmacohpore)

        if len(featureContribution) == 0:
            raise ValueError('featureContribution had no elements, but {} were expected'.format(qpharModel.numFeatures))
        nrPositiveContributions = np.sum((np.array(featureContribution) > 0))
        if nrPositiveContributions < self.parameters.nrOptimalFeatures:
            logging.info('Could not create optimal pharmacophore with {} features. Only {} positive contributions found'.format(self.parameters.nrOptimalFeatures, nrPositiveContributions))
            return None

        # process negatively contributing features
        if self.parameters.setXvols:
            logging.info('Converting negatively contributing features to exclusion volumes')
            for featureId, feature in enumerate(qpharPharmacohpore):
                contribution = featureContribution[featureId]
                if contribution < 0:
                    Pharm.setGeometry(feature, Pharm.FeatureGeometry.SPHERE)
                    Pharm.setType(feature, Pharm.FeatureType.X_VOLUME)

                if self.parameters.weightFeaturesAtScreening:
                    self.weightFeatureByImportance(feature, featureImportance[featureId])

        else:
            logging.info('Removing negatively contributing features from pharmacophore')
            for featureId in range(qpharPharmacohpore.numFeatures - 1, -1, -1):
                contribution = featureContribution[featureId]
                if contribution < 0:
                    qpharPharmacohpore.removeFeature(featureId)
                    featureContribution.pop(featureId)
                    featureImportance.pop(featureId)
                    continue

                if self.parameters.weightFeaturesAtScreening:
                    self.weightFeatureByImportance(qpharPharmacohpore.getFeature(featureId),
                                                   featureImportance[featureId])

        # save feature contribution and importance on feature
        for i in range(qpharPharmacohpore.numFeatures):
            feature = qpharPharmacohpore.getFeature(i)
            feature.setProperty(FEATURE_CONTRIBUTION_LOOKUPKEY, featureContribution[i])
            feature.setProperty(FEATURE_IMPORTANCE_LOOKUPKEY, featureImportance[i])

        # make optimal pharmacophore based on feature contribution and importance
        optimalPharmacophore = Pharm.BasicPharmacophore()
        if self.parameters.weightFeaturesByImportance:
            importanceWeightedContributions: np.ndarray = featureContribution * np.abs(featureImportance)
        else:
            importanceWeightedContributions: np.ndarray = np.array(featureContribution)
        sortedArgs = np.argsort(importanceWeightedContributions)
        for i in range(1, min(self.parameters.nrOptimalFeatures, len(sortedArgs)) + 1):
            featureId = sortedArgs[-i]
            contribution = importanceWeightedContributions[featureId]
            if contribution < 0:
                continue
            addedFeature = optimalPharmacophore.addFeature()
            addedFeature.assign(qpharPharmacohpore.getFeature(int(featureId)))

        if self.parameters.setXvols:  # add same number of xvols as was added for optimal features
            for i in range(min(self.parameters.nrOptimalFeatures, len(sortedArgs))):
                featureId = sortedArgs[i]
                feature = qpharPharmacohpore.getFeature(int(featureId))
                if Pharm.getType(feature) != Pharm.FeatureType.X_VOLUME:
                    continue

                addedFeature = optimalPharmacophore.addFeature()
                addedFeature.assign(feature)

        logging.info('Generated optimal pharmacoophore with {} features and {} exclusion volumes'.format(
            self.parameters.nrOptimalFeatures,
            0 if not self.parameters.setXvols else self.parameters.nrOptimalFeatures,
        ))
        return optimalPharmacophore

    def weightFeatureByImportance(self, feature: Pharm.BasicFeature, featureImportance: float) -> None:
        weight = self.featureImportanceScaler.transform([[abs(featureImportance)]])
        Pharm.setWeight(feature, weight[0][0])

    def featureContributionFromQphar(self, qpharPharmacohpore: Pharm.BasicPharmacophore) -> List[float]:
        """
        Determine the feature contribution from qphar features. If the average activity value of a feature is below
        the midpoint of activity ranges, it contributes negatively, otherwise it contributes positively to the activity
        of the pharmacophore.
        :return:
        """
        activities = np.array([np.mean(feature.getProperty(LOOKUPKEYS['activities'])) for feature in qpharPharmacohpore])
        # scaler = MinMaxScaler((-1, 1))
        scaler = StandardScaler()
        scaled = scaler.fit_transform(activities.reshape((-1, 1)))
        return scaled.flatten().tolist()

    def featureContributionFromMl(self,
                                  qpharModel: Qphar,
                                  mlModel: Union[RandomForestRegressor, Ridge, Lasso, LinearRegression, PLSRegression],
                                  ) -> List[float]:
        """
        Determine the feature contribution from the ml model of the qphar model. Either get the coefficients from a
        linear reegression, or determine the weighted average prediction of random forests when the feature is included
        and when the feature is not. Quantified difference can then be interpreted as a positive or negative
        contribution of the feature.
        :return:
        """
        featureContribution = {}  # map feature id to average value and nr samples
        if qpharModel.modelType == 'randomForest':
            # for each node in each tree, check whether the average prediction below the threshold has the higher or
            # lower activity. If lower thresholds are associated with higher predicted activity values, then the
            # feature is beneficial for high activity, otherwise not. Each average prediction is weighted by the nr
            # of samples contained on this side of the tree branch. Furthermore, additional weighting is applied by the
            # threshold which is put into relation of min and max distance. Higher thresholds generally contribute less
            # to predictions, since they are much less likely to have samples on both sides. For example feature values
            # are ranging from 2 to 10 Angström distance. If the threshold is 500 Angström, then it is rather
            # irrelevant, since basically all samples will end up on one side of the tree. Therefore, there is no power
            # to distinguish between samples and it becomes less important. Conversely, a threshold of e.g. 5 Ansgtröm
            # will be much more relevant in this example.

            class Node:

                def __init__(self,
                             nodeId: int,
                             threshold: float,
                             featureId: int,
                             value: float,
                             nrSamples: int,
                             leftChildId: int,
                             rightChildId: int,
                             ):
                    self.nodeId = nodeId
                    self.threshold = threshold
                    self.featureId = featureId
                    self.value = value
                    self.nrSamples = nrSamples
                    self.leftChildId = leftChildId
                    self.rightChildId = rightChildId
                    self.isLeaveNode = leftChildId == rightChildId

            def convertToTree(decisionTree) -> Dict[int, Node]:
                tree = {}
                childrenLeft, childrenRight = decisionTree.tree_.children_left, decisionTree.tree_.children_right
                features, values, thresholds, nrNodeSamples = decisionTree.tree_.feature, decisionTree.tree_.value, decisionTree.tree_.threshold, decisionTree.tree_.n_node_samples
                for nodeId in range(len(childrenLeft)):
                    node = Node(nodeId,
                                thresholds[nodeId],
                                features[nodeId],
                                np.mean(values[nodeId].flatten()),
                                nrNodeSamples[nodeId],
                                childrenLeft[nodeId],
                                childrenRight[nodeId],
                                )
                    tree[nodeId] = node
                return tree

            def getAveragePredictionForNode(node: Node, tree: Dict[int, Node]) -> float:
                if node.isLeaveNode:
                    return node.value

                leftNode, rightNode = tree[node.leftChildId], tree[node.rightChildId]
                leftPrediction = getAveragePredictionForNode(leftNode, tree)
                rightPrediction = getAveragePredictionForNode(rightNode, tree)

                return (leftPrediction * leftNode.nrSamples + rightPrediction * rightNode.nrSamples) / (
                            leftNode.nrSamples + rightNode.nrSamples)

            def getBinaryImportanceByThreshold(node: Node, tree: Dict[int, Node]) -> float:
                return float(tree[node.leftChildId].value < tree[node.rightChildId].value)

            def getContinuousImportanceByThreshold(node: Node, tree: Dict[int, Node]) -> float:
                threshold = node.threshold
                binaryPrediction = getBinaryImportanceByThreshold(node, tree)
                importanceFactor = 1/threshold if threshold >= 1 else threshold**2
                return binaryPrediction * importanceFactor

            for dTree in mlModel.estimators_:
                tree = convertToTree(dTree)
                for nodeId in range(len(tree)):
                    node = tree[nodeId]
                    if node.featureId == -2:  # leave node
                        continue

                    avgPrediction: float
                    if self.parameters.rfContributionType == RfContributionType.AVG_PREDICTION:
                        avgPrediction = getAveragePredictionForNode(node, tree)
                    elif self.parameters.rfContributionType == RfContributionType.BINARY_THRESHOLD:
                        avgPrediction = getBinaryImportanceByThreshold(node, tree)
                    elif self.parameters.rfContributionType == RfContributionType.CONTINUOUS_THRESHOLD:
                        avgPrediction = getContinuousImportanceByThreshold(node, tree)
                    else:
                        raise ValueError('Unhandled contribution type {}'.format(self.parameters.rfContributionType))

                    if node.featureId not in featureContribution.keys():
                        featureContribution[node.featureId] = {
                            'pred': avgPrediction,
                            'nrSamples': node.nrSamples,
                        }
                    else:
                        currentContribution = featureContribution[node.featureId]
                        totalSamples = currentContribution['nrSamples'] + node.nrSamples
                        currentContribution['pred'] = currentContribution['pred'] * (
                                    currentContribution['nrSamples'] / totalSamples) + avgPrediction * (
                                                                  node.nrSamples / totalSamples)
                        currentContribution['nrSamples'] = totalSamples

        else:
            raise NotImplementedError('Calculation of feature contribution for {} is currently not supported'.format(
                qpharModel.modelType))

        # scale activities
        normalizedFeatureContribution = np.array([featureContribution.get(featureId, {'pred': np.nan})['pred'] for featureId in range(qpharModel.cleanedHP.numFeatures)])
        scaler = StandardScaler()
        meanContribution = np.nanmean(normalizedFeatureContribution)
        normalizedFeatureContribution = np.where(np.isnan(normalizedFeatureContribution), meanContribution, normalizedFeatureContribution)
        scaled = scaler.fit_transform(np.reshape(normalizedFeatureContribution, (-1, 1)))
        return scaled.flatten().tolist()

    @staticmethod
    def visualizeFeatures(pharmacophore: Pharm.BasicPharmacophore,
                          outputPath: str,
                          ) -> None:
        visualizer = Visualizer(VisualizerParameters(saveAsKont=True, gridInterval=0.2))
        visualizer.setOutputPath(outputPath)
        visualizer.setPharmacophore(pharmacophore)
        visualizer.calculateXYZSearchSpace(pharmacophore)
        dimensions = visualizer.getDimensions(pharmacophore)

        contributionGrid, importanceGrid = np.zeros(dimensions), np.zeros(dimensions)
        for feature in pharmacophore:
            featureCoordinates = visualizer.getCoordinates(feature)
            featureTolerance = Pharm.getTolerance(feature)
            contribution = feature.getProperty(FEATURE_CONTRIBUTION_LOOKUPKEY)
            importance = feature.getProperty(FEATURE_IMPORTANCE_LOOKUPKEY)
            for i, x in enumerate(visualizer.xCoords):
                for j, y in enumerate(visualizer.yCoords):
                    for k, z in enumerate(visualizer.zCoords):
                        currentCoordinates = np.array([x, y, z])
                        distance = euclideanDistance(currentCoordinates, featureCoordinates)
                        if distance > featureTolerance or distance == 0:
                            continue

                        contributionGrid[i, j, k] = contribution
                        importanceGrid[i, j, k] = importance

        positiveContribution, negativeContribution = visualizer.scaleGrid(contributionGrid)
        positiveImportance, negativeImportance = visualizer.scaleGrid(importanceGrid)
        for grid, name in zip([positiveContribution, negativeContribution, positiveImportance, negativeImportance],
                              ['positiveFeatureContribution', 'negativeFeatureContribution', 'positiveFeatureImportance', 'negativeFeatureImportance']):
            visualizer.writeKontFile('{}{}.kont'.format(visualizer.outputPath, name), grid, name)


class Screener:

    def __init__(self,
                 dataset: Union[str, List[Chem.BasicMolecule], None] = None,
                 pharmacophore: Union[str, Pharm.BasicPharmacophore, None] = None,
                 hitPath: str = None,
                 screeningDbFileFormat: str = '.psd',
                 dbGeneratorPath: str = None,
                 dbScreenerPath: str = None,
                 ):
        self.pharmacophorePath: Union[str, None] = None
        self.databasePath: Union[str, None] = None
        self.hitPath = None
        self.screeningDbFileFormat = screeningDbFileFormat
        self.dbGeneratorPath = PSD_DB_CREATOR_PATH if dbGeneratorPath is None else dbGeneratorPath
        self.dbScreenerPath = PSD_DB_SCREENER_PATH if dbScreenerPath is None else dbScreenerPath
        if hitPath is not None:
            self.setHitPath(hitPath)
        else:
            defaultHitPath = '/tmp/screeningResults.sdf'
            logging.info('hitPath not set. Using default output instead: {}'.format(defaultHitPath))
            self.hitPath = defaultHitPath
        if pharmacophore is not None:
            self.setPharmacophore(pharmacophore)
        if dataset is not None:
            self.setDatabase(dataset)
        self.performedScreen = False

    def run(self) -> None:
        if self.pharmacophorePath is None:
            raise RuntimeError('Pharmacophore must be set before running a screen')
        if self.databasePath is None:
            raise RuntimeError('Database must be set before running a screen')

        logging.info('Screening database {}'.format(self.databasePath))
        logging.info('Using pharmacophore {}'.format(self.pharmacophorePath))
        os.system('{} -d {} -q {} -o {}'.format(self.dbScreenerPath,
                                                self.databasePath,
                                                self.pharmacophorePath,
                                                self.hitPath),
                  )
        self.performedScreen = True
        logging.info('Saved hits to {}'.format(self.hitPath))

    def setDatabase(self, dataset: Union[str, List[Chem.BasicMolecule]]) -> None:
        if isinstance(dataset, str):
            _output = '{}{}'.format(os.path.splitext(dataset)[0], self.screeningDbFileFormat)

            if dataset.endswith('.sdf'):  # create screening DB from sdf
                os.system('{} -i {} -o {}'.format(self.dbGeneratorPath, dataset, _output))

            elif dataset.endswith(self.screeningDbFileFormat):
                pass

            else:
                raise TypeError('Path to screening dataset must be {} or .sdf, but {} was given'.format(
                    self.screeningDbFileFormat, os.path.splitext(dataset)[-1]))

            self.databasePath = _output

        elif isinstance(dataset, list):
            from src.molecule_tools import mol_to_sdf

            molecules = []
            for i, mol in enumerate(dataset):
                if not isinstance(mol, Chem.BasicMolecule):
                    logging.info('Skipping element nr {} because not instance of Chem.BasicMolecule'.format(i))

                has3DCoordinates = True
                for atom in mol.atoms:
                    if not Chem.has3DCoordinates(atom):
                        logging.info('Skipping molecule nr {} due to missing 3D coordinates'.format(i))
                        has3DCoordinates = False
                        break

                if has3DCoordinates:
                    molecules.append(mol)

            mol_to_sdf(molecules, '/tmp/mols_to_screen.sdf', multiconf=True)
            os.system('{} -i /tmp/mols_to_screen.sdf -o /tmp/mols_to_screen{}'.format(self.dbGeneratorPath, self.screeningDbFileFormat))
            self.databasePath = '/tmp/mols_to_screen{}'.format(self.screeningDbFileFormat)

        else:
            raise TypeError('Unhandled dataset given. Must be instance of str or List of Chem.BasicMolecule. {} was given'.format(type(dataset)))

        self.performedScreen = False

    def setPharmacophore(self, pharmacophore: Union[str, Pharm.BasicPharmacophore]) -> None:
        if isinstance(pharmacophore, str):
            if not pharmacophore.endswith('.pml'):
                raise TypeError('Pharmacophore file must be of type .pml, but {} was given'.format(
                    os.path.splitext(pharmacophore)[-1]))

            self.pharmacophorePath = pharmacophore

        elif isinstance(pharmacophore, Pharm.BasicPharmacophore):
            savePharmacophore(pharmacophore, '/tmp/query_pharmacophore.pml')
            self.pharmacophorePath = '/tmp/query_pharmacophore.pml'

        else:
            raise TypeError('Unhandled pharmacophore given. Must be instance of str of Pharm.BasicPharmacophore. {} was given'.format(type(pharmacophore)))

        self.performedScreen = False

    def setHitPath(self, hitPath: str) -> None:
        if not hitPath.endswith('.sdf'):
            raise TypeError('Hits must be saved as .sdf, but {} was provided'.format(hitPath))
        self.hitPath = hitPath
        self.performedScreen = False

    def loadHits(self, activityName: str = None) -> List[Chem.BasicMolecule]:
        """

        :param activityName: Name of activity property to load if exists
        :return:
        """
        if not self.performedScreen:
            raise RuntimeError('Must run screening before loading hits.')

        r = SDFReader(self.hitPath, multiconf=True)
        hitMolecules = [mol for mol in r]
        for i, mol in enumerate(hitMolecules):
            sdb = Chem.getStructureData(mol)
            for p in sdb:
                if activityName in p.header:
                    try:
                        activity = float(p.data)
                    except ValueError:
                        logging.info('Could not convert activity of mol {} to float. Setting activity as str'.format(i))
                        activity = p.data
                    mol.setProperty(LOOKUPKEYS['activity'], activity)

                elif 'hash' in p.header:
                    mol.setProperty(HASH_LOOKUPKEY, p.data)
        logging.info('Loaded {} hits from {}'.format(len(hitMolecules), self.hitPath))
        return hitMolecules


def makeParameterCombinations(searchParams: Dict[str, List[Union[str, int, float, bool]]]
                              ) -> List[Dict[str, Union[str, int, float, bool]]]:
    from itertools import product

    keys = sorted(searchParams.keys())
    combinations = product(*(searchParams[k] for k in keys))
    combinedParameters = [{keys[k]: values[k] for k in range(len(keys))} for values in combinations]
    return combinedParameters


def getScreeningPerformance(referenceDataset: List[Chem.BasicMolecule],
                            screeningResults: Dict[int, Dict[str, Union[Pharm.BasicPharmacophore, List[Chem.BasicMolecule]]]],
                            metric: Metric,
                            beta: float,
                            qpharModel: Qphar = None,
                            ) -> pd.DataFrame:
    logging.info('Analysing screening results...')
    screeningPerformance = evaluateScreening(referenceDataset, screeningResults, beta=beta, qpharModel=qpharModel)
    screeningPerformance.sort_values(metric.value, ascending=False, inplace=True)
    screeningPerformance.reset_index(drop=False, inplace=True)
    return screeningPerformance


def saveTopNResults(nrTopResults: int,
                    outputPath: str,
                    screeningPerformance: pd.DataFrame,
                    screeningResults: Dict[int, Dict[str, Union[Pharm.BasicPharmacophore, List[Chem.BasicMolecule]]]],
                    ) -> None:
    logging.info('Saving top {} results to {}'.format(nrTopResults, outputPath))
    row: pd.Series
    for i, row in screeningPerformance.iloc[:min(nrTopResults, screeningPerformance.shape[0])].iterrows():
        path = '{}{}/'.format('{}/'.format(outputPath) if not outputPath.endswith('/') else outputPath, i)
        if not os.path.isdir(path):
            os.makedirs(path)

        savePharmacophore(screeningResults[row['index']]['pharmacophore'], '{}optimalPharmacophore.pml'.format(path))
        mol_to_sdf(screeningResults[row['index']]['hits'], '{}hits.sdf'.format(path))
        with open('{}/performance.json'.format(path), 'w') as f:
            json.dump(row.to_dict(), f, indent=2)
        with open('{}/parameters.json'.format(path), 'w') as f:
            paramsToSave = {}
            for k, v in screeningResults[row['index']]['parameters'].__dict__.items():
                if isinstance(v, (RfContributionType, Metric)):
                    v = v.value
                paramsToSave[k] = v
            json.dump(paramsToSave, f, indent=2)

        OptimalPharmacophoreGenerator.visualizeFeatures(screeningResults[row['index']]['pharmacophore'], path)


def saveTestResults(outputPath: str,
                    screeningPerformance: pd.DataFrame,
                    screeningResults: Dict[int, Dict[str, Union[Pharm.BasicPharmacophore, List[Chem.BasicMolecule]]]],
                    ) -> None:
    for i, row in screeningPerformance.iterrows():
        if not outputPath.endswith('/'):
            path = '{}/{}/'.format(outputPath, i)
        else:
            path = '{}{}/'.format(outputPath, i)
        if not os.path.isdir(path):
            os.makedirs(path)
        mol_to_sdf(screeningResults[i]['hits'], '{}test_hits.sdf'.format(path))
        with open('{}test_performance.json'.format(path), 'w') as f:
            json.dump(row.to_dict(), f, indent=2)


def runParameterOptimization(qpharModel: Qphar,
                             referenceDataset: List[Chem.BasicMolecule],
                             screeningDbPath: str,
                             outputPath: str,
                             activityName: str,
                             metric: Metric,
                             nrTopResults: int,
                             screeningTestDbPath: str = None,
                             referenceTestDataset: List[Chem.BasicMolecule] = None,
                             beta: float = 1,
                             ) -> None:
    """
    Optimize the following parameters to find an optimal pharmacophore:
        - weightFeaturesByImportance: bool
        - weightFeaturesAtScreening: bool
        - setXvols: bool
        - featureContributionFromMl: bool
        - nrOptimalFeatures: int
        - rfContributionType: RfContributionType
    The quality of the optimal pharmacophore is determined by the its ability to distinguish between high and low
    activity compounds from a given dataset.
    :param qpharModel:
    :param referenceDataset:
    :param screeningDbPath:
    :param outputPath:
    :param activityName:
    :param metric:
    :param nrTopResults: Number of top n results to save.
    :param screeningTestDbPath:
    :param referenceTestDataset:
    :param beta:
    :return:
    """
    # prepare parameter combinations
    searchParams = {
        'weightFeaturesByImportance': [True, False],
        'weightFeaturesAtScreening': [True, False],
        'setXvols': [True, False],
        'featureContributionFromMl': [True, False],
        'nrOptimalFeatures': [4, 5, 6, 7, 8, 9],
        'rfContributionType': [e for e in RfContributionType]
    }
    parameterCombinations = makeParameterCombinations(searchParams)
    logging.info('Testing {} parameter combinations'.format(len(parameterCombinations)))

    # apply and evaluate parameter combinations
    screener = Screener(dataset=screeningDbPath)
    screeningResults: Dict[int, Dict[str, Union[Pharm.BasicPharmacophore, List[Chem.BasicMolecule]]]] = {}
    generatedPharmacophoreHashes: Set[str] = set()
    for i, combination in enumerate(parameterCombinations):
        logging.info('Testing combination {} / {}'.format(i, len(parameterCombinations)))
        params = Parameters(outputPath=outputPath, metric=metric, nrTopResults=nrTopResults, **combination)
        generator = OptimalPharmacophoreGenerator(params)
        optimalPharmacophore = generator.run(qpharModel)
        if optimalPharmacophore is None:
            continue
        pharmHash = makePharmacophoreHash(optimalPharmacophore)
        if pharmHash in generatedPharmacophoreHashes:
            continue

        generatedPharmacophoreHashes.add(pharmHash)
        prediction, alignmentScore = qpharModel.predict([optimalPharmacophore], returnScores=True, ignoreAlign=True)
        screener.setPharmacophore(optimalPharmacophore)
        screener.run()
        hits = screener.loadHits(activityName)
        screeningResults[i] = {
            'pharmacophore': optimalPharmacophore,
            'hits': hits,
            'parameters': params,
            'prediction': prediction,
            'alignmentScore': alignmentScore[0],
        }

    if len(screeningResults) == 0:
        logging.info('No pharmacophores to analyse. Exiting...')
        return

    screeningPerformance = getScreeningPerformance(referenceDataset, screeningResults, metric, beta,
                                                   qpharModel=qpharModel)
    saveTopNResults(nrTopResults, outputPath, screeningPerformance, screeningResults)

    # if test set given, evaluate top n on test set
    if screeningTestDbPath is None:
        return

    logging.info('Evaluating top {} models on test set'.format(nrTopResults))
    screener = Screener(dataset=screeningTestDbPath)
    testScreeningResults: Dict[int, Dict[str, Union[Pharm.BasicPharmacophore, List[Chem.BasicMolecule]]]] = {}
    for i, row in screeningPerformance.iloc[:min(nrTopResults, screeningPerformance.shape[0])].iterrows():
        pharmacophore = screeningResults[row['index']]['pharmacophore']
        screener.setPharmacophore(pharmacophore)
        screener.run()
        hits = screener.loadHits(activityName)
        testScreeningResults[i] = {
            'hits': hits,
        }

    testScreeningPerformance = getScreeningPerformance(referenceTestDataset, testScreeningResults, metric, beta,
                                                       qpharModel=qpharModel)
    saveTestResults(outputPath, testScreeningPerformance, testScreeningResults)


def getClassFromActivity(activity: float, percentiles: List[float]) -> int:
    nrClasses = len(percentiles) + 1
    for i, perc in enumerate(percentiles):
        if activity > perc:
            return i
    return nrClasses


def calculateMetrics(tp: int, fp: int, tn: int, fn: int, beta: float = 1) -> Dict[str, float]:
    metrics = Metrics(tp, fp, tn, fn)
    return {
        Metric.ACCURACY.value: metrics.calcAccuracy(),
        Metric.PRECISION.value: metrics.calcPrecision(),
        Metric.SENSITIVITY.value: metrics.calcSensitivity(),
        Metric.SPECIFICITY.value: metrics.calcSpecificity(),
        Metric.BALANCED_ACCURACY.value: metrics.calcBalancedAccuracy(),
        Metric.F1_SCORE.value: metrics.calcF1Score(),
        Metric.FAKE_F1_SCORE.value: metrics.calcFakeF1Score(),
        Metric.FBETA_SCORE.value: metrics.calcFBetaScore(beta)
    }


def calcRocAuc(trueClasses: List[int],
               screenedClasses: List[int],
               trueMolHashes: List[str],
               hits: List[Chem.BasicMolecule],
               ) -> float:
    hitsHashes = {mol.getProperty(HASH_LOOKUPKEY): i for i, mol in enumerate(hits)}
    predClasses = []
    for trueHash in trueMolHashes:
        if trueHash in hitsHashes.keys():
            index = hitsHashes[trueHash]
            predClasses.append(screenedClasses[index])
        else:
            predClasses.append(0)
    return Metrics.calcRocAucScore(np.array(trueClasses), np.array(predClasses))


def makePharmacophoreHash(pharmacophore: Pharm.BasicPharmacophore) -> str:
    featureTuples = []
    for i in range(pharmacophore.numFeatures):
        f1 = pharmacophore.getFeature(i)
        f1Type = Pharm.getType(f1)
        f1Coords = Chem.get3DCoordinates(f1).toArray()

        for j in range(i+1, pharmacophore.numFeatures):
            f2 = pharmacophore.getFeature(j)
            f2Type = Pharm.getType(f2)
            f2Coords = Chem.get3DCoordinates(f2).toArray()

            distance = round(euclideanDistance(f1Coords, f2Coords), 4)
            if f1Type < f2Type:
                featureTuples.append(np.array([float(f1Type), float(f2Type), distance]).reshape(1, -1))
            else:
                featureTuples.append(np.array([float(f2Type), float(f1Type), distance]).reshape(1, -1))

    featureTuples = np.concatenate(featureTuples, axis=0)
    sortedIndices = np.lexsort((featureTuples[:, 0], featureTuples[:, 1], featureTuples[:, 2]))
    featureTuples = featureTuples[sortedIndices]
    return md5(featureTuples.tobytes()).hexdigest()


def makeMoleculeHash(mol: Chem.BasicMolecule) -> str:
    Chem.makeHydrogenDeplete(mol)
    Chem.calcImplicitHydrogenCounts(mol, True)
    try:
        Chem.calcAtomStereoDescriptors(mol, True)
        Chem.calcBondStereoDescriptors(mol, True)
        Chem.calcCIPPriorities(mol, True)
        Chem.calcAtomCIPConfigurations(mol, True)
        Chem.calcBondCIPConfigurations(mol, True)
        return str(Chem.calcHashCode(mol))
    except:
        return str(Chem.calcHashCode(mol,
                                     atom_flags=Chem.AtomPropertyFlag.TYPE | Chem.AtomPropertyFlag.H_COUNT | Chem.AtomPropertyFlag.FORMAL_CHARGE | Chem.AtomPropertyFlag.AROMATICITY,
                                     bond_flags=Chem.BondPropertyFlag.ORDER | Chem.BondPropertyFlag.TOPOLOGY | Chem.BondPropertyFlag.AROMATICITY
                                     ))


def getRankedHits(qpharModel: Qphar, hits: List[Chem.BasicMolecule]) -> Tuple[np.ndarray, np.ndarray]:
    if len(hits) == 0:
        return np.array([]), np.array([])
    predictions: np.ndarray = qpharModel.predict(hits)
    sortedPredictionIndices = np.argsort(predictions.flatten())
    trueActivities = [mol.getProperty(LOOKUPKEYS['activity']) for mol in hits]
    sortedHitIndices = np.argsort(np.array(trueActivities))
    return sortedHitIndices, sortedPredictionIndices


def calculateKenndalsTau(qpharModel: Qphar,
                         hits: List[Chem.BasicMolecule],
                         ) -> float:
    sortedHitIndices, sortedPredictionIndices = getRankedHits(qpharModel, hits)
    if len(sortedHitIndices) == 0:
        return 0
    kendallsTau = kendalltau(sortedHitIndices, sortedPredictionIndices)[0]
    return kendallsTau


def calculateSpearmanR(qpharModel: Qphar, hits: List[Chem.BasicMolecule]) -> float:
    sortedHitIndices, sortedPredictionIndices = getRankedHits(qpharModel, hits)
    if len(sortedHitIndices) == 0:
        return 0
    spearmanR = spearmanr(sortedHitIndices, sortedPredictionIndices)[0]
    return spearmanR


def evaluateScreening(referenceDataset: List[Chem.BasicMolecule],
                      screeningResults: Dict[int, Dict[str, Union[Pharm.BasicPharmacophore, List[Chem.BasicMolecule]]]],
                      beta: float = 1,
                      qpharModel: Qphar = None,
                      ) -> pd.DataFrame:
    referenceActivities: List[float] = [mol.getProperty(LOOKUPKEYS['activity']) for mol in referenceDataset]
    referenceHashes: List[str] = [mol.getProperty(HASH_LOOKUPKEY) for mol in referenceDataset]
    activePercentile = np.percentile(referenceActivities, 80)
    classes = [int(mol.getProperty(LOOKUPKEYS['activity']) > activePercentile) for mol in referenceDataset]
    nrActives = {'active': sum(classes), 'inactive': len(classes) - sum(classes)}
    metrics = {}

    for i, results in screeningResults.items():
        hits = results['hits']
        screenedClasses = [int(mol.getProperty(LOOKUPKEYS['activity']) > activePercentile) for mol in hits]
        tp = sum(screenedClasses)
        fp = len(screenedClasses) - tp
        fn = nrActives['active'] - tp
        tn = nrActives['inactive'] - fp
        currentScores = {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            Metric.ROC_AUC.value: calcRocAuc(classes, screenedClasses, referenceHashes, hits),
            **calculateMetrics(tp, fp, tn, fn, beta=beta)
        }
        if qpharModel is not None:
            currentScores[Metric.KENDALLS_TAU.value] = calculateKenndalsTau(qpharModel, hits)
            currentScores[Metric.SPEARMAN_R.value] = calculateSpearmanR(qpharModel, hits)
        metrics[i] = currentScores

    return pd.DataFrame.from_dict(metrics, orient='index')


def main(model: Qphar, params: Parameters) -> None:
    """
    Make optimal pharmacophore with set parameters
    :param model:
    :param params:
    :return:
    """
    generator = OptimalPharmacophoreGenerator(params)
    optimalPharmacophore = generator.run(model)

    if not os.path.isdir(params.outputPath):
        os.makedirs(params.outputPath)
    savePharmacophore(optimalPharmacophore, '{}optimalPharmacophore.pml'.format(params.outputPath))
    with open('{}paramaeters.json'.format(params.outputPath), 'w') as f:
        paramsToSave = {}
        for k, v in params.__dict__.items():
            if isinstance(v, (RfContributionType, Metric)):
                v = v.value
            paramsToSave[k] = v
        json.dump(paramsToSave, f, indent=2)


def parseArgs() -> Parameters:
    parser = ArgumentParser()
    parser.add_argument('-model', required=True, type=str, help='path of model to create optimal pharmacophore for')
    parser.add_argument('-output', required=True, type=str, help='folder where optimal pharmacophore should be saved')
    parser.add_argument('-weightFeatures', required=False, default=False, action='store_true',
                        help='weight pharmacophore features by adjusting their size according to the feature importance determined by the ml model')
    parser.add_argument('-setXvols', required=False, default=False, action='store_true',
                        help='replace pharmacophore features impacting activity negatively with exclusion volumes')
    parser.add_argument('-nrOptimalFeatures', default=6, required=False, type=int,
                        help='nr of optimal pharmacophore features to create. The higher the more restrictive the pharmacophore')
    parser.add_argument('-findBestParameters', default=False, required=False, action='store_true',
                        help='iterate over various parameter combinations to find pharmacophore with best selectivity')
    parser.add_argument('-rfContributionType', required=False, default=RfContributionType.BINARY_THRESHOLD.value,
                        help='method to determine contribution of pharmacophore feature to activity for random forest. One of [{}]'.format([e.value for e in RfContributionType]))
    parser.add_argument('-screeningDbPath', required=False, default=None,
                        help='path to dataset to be used for pharmacophore screening')
    parser.add_argument('-screeningTestDbPath', required=False, default=None,
                        help='path to dataset used for validation ')
    parser.add_argument('-referenceData', required=False, default=None,
                        help='path to molecules used as validation of screening results')
    parser.add_argument('-referenceTestData', required=False, default=None,
                        help='path to molecules used as validation of test set')
    parser.add_argument('-activityName', required=False, default=None,
                        help='name of activity property in reference dataset')
    parser.add_argument('-metric', required=False, default=Metric.ACCURACY.value,
                        help='metric to score screening performance of pharmacophore. One of [{}]'.format(
                            e.value for e in Metric))
    parser.add_argument('-beta', required=False, default=1, type=float, help='beta in f-score')
    parser.add_argument('-nrTopResults', required=False, default=5, type=int, help='nr of top n results to save')
    parser.add_argument('-logLevel', required=False, default='INFO', type=str,
                        help='granularity level of logging')
    args = parser.parse_args()

    logging.getLogger().setLevel(level=logging.getLevelName(args.logLevel))
    logging.basicConfig(level=logging.getLevelName(args.logLevel))

    rfContributionType: RfContributionType
    try:
        rfContributionType = RfContributionType[args.rfContributionType]
    except KeyError:
        raise KeyError('{} not a member of RfContributionType. Must be one of {}'.format(args.rfContributionType, [e.value for e in RfContributionType]))

    metric: Metric
    try:
        metric = Metric[args.metric]
    except KeyError:
        raise KeyError('{} not a member of Metric. Must be one of {}'.format(args.metric, [e.value for e in Metric]))

    if args.findBestParameters:
        if args.referenceData is None or args.screeningDbPath is None or args.activityName is None:
            raise ValueError('When optimising parameters, "referenceData", "screeningDbPath", and "activityName" need to be provided. {} was given'.format(
                args.referenceData, args.screeningDbPath, args.activityName
            ))

    if args.screeningTestDbPath is not None and args.referenceTestData is None:
        raise ValueError('-referenceTestData needs to be provided if -screeningTestDbPath is provided.')

    params = Parameters(modelPath=args.model if args.model.endswith('/') else '{}/'.format(args.model),
                        outputPath=args.output if args.output.endswith('/') else '{}/'.format(args.output),
                        weightFeaturesByImportance=args.weightFeatures,
                        nrOptimalFeatures=args.nrOptimalFeatures,
                        setXvols=args.setXvols,
                        findBestParameters=args.findBestParameters,
                        rfContributionType=rfContributionType,
                        screeningDbPath=args.screeningDbPath,
                        referenceDatasetPath=args.referenceData,
                        activityName=args.activityName,
                        metric=metric,
                        nrTopResults=args.nrTopResults,
                        screeningTestDbPath=args.screeningTestDbPath,
                        referenceTestData=args.referenceTestData,
                        beta=args.beta
                        )
    return params


def loadReferenceDataset(path: str, activityName: str) -> List[Chem.BasicMolecule]:
    r = SDFReader(path, multiconf=True)
    molecules = []
    for i, mol in enumerate(r):
        sdb = Chem.getStructureData(mol)
        hasActivity = True
        for p in sdb:
            if activityName in p.header:
                try:
                    activity = float(p.data)
                except ValueError:
                    logging.info('Cannot convert activity of mol nr {} to float. Skipping'.format(i))
                    hasActivity = False
                    break
                mol.setProperty(LOOKUPKEYS['activity'], activity)
            elif 'hash' in p.header:
                mol.setProperty(HASH_LOOKUPKEY, p.data)
        if hasActivity:
            molecules.append(mol)
    return molecules


def euclideanDistance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.linalg.norm(point2-point1, ord=2)


if __name__ == '__main__':
    params = parseArgs()

    model = Qphar()
    model.load(params.modelPath)

    if params.findBestParameters:
        molecules = loadReferenceDataset(params.referenceDatasetPath, params.activityName)
        testMolecules = loadReferenceDataset(params.referenceTestData, params.activityName) if params.referenceTestData is not None else None
        runParameterOptimization(model,
                                 molecules,
                                 params.screeningDbPath,
                                 params.outputPath,
                                 params.activityName,
                                 params.metric,
                                 params.nrTopResults,
                                 screeningTestDbPath=params.screeningTestDbPath,
                                 referenceTestDataset=testMolecules,
                                 beta=params.beta,
                                 )
    else:
        main(model, params)
