"""
Prepare analysis of optimal pharmacophores to be visualized.
Visualizes the following form of grids:
    - importance of pharmacophore feature for prediction in qphar model
    - contribution type (positive or negative) of pharmacophore feature in prediction of qphar model. Magnitude of
    contribution is displayed by saturation (alpha-values) of each grid point.
Output as grid files which can be read by LigandScout.
"""
from argparse import ArgumentParser
import os
import logging
from enum import Enum
from typing import List, Dict, Union, Tuple

import CDPL.Pharm as Pharm
import CDPL.Math as Math
import CDPL.Chem as Chem
import numpy as np

from src.qphar import Qphar
from src.pharmacophore_tools import loadPharmacophore


MODEL_BASE_PATH = './data/hypogen_models/hypogen_qphar/results/'


class FEATURE_TYPES(Enum):
    AROMATIC = Pharm.FeatureType.AROMATIC
    HYDROPHOBIC = Pharm.FeatureType.HYDROPHOBIC
    H_BOND_ACCEPTOR = Pharm.FeatureType.H_BOND_ACCEPTOR
    H_BOND_DONOR = Pharm.FeatureType.H_BOND_DONOR
    NEG_IONIZABLE = Pharm.FeatureType.NEG_IONIZABLE
    POS_IONIZABLE = Pharm.FeatureType.POS_IONIZABLE
    X_VOLUME = Pharm.FeatureType.X_VOLUME


class Parameters:

    def __init__(self,
                 target: str = None,
                 pharmacophorePath: str = None,
                 outputPath: str = None,
                 searchSpace: float = 2.0,
                 gridInterval: float = 0.5,
                 margin: float = 2,
                 saveAsKont: bool = False,
                 ):
        self.target = target
        self.pharmacophorePath = pharmacophorePath
        self.outputPath = outputPath
        self.searchSpace = searchSpace
        self.gridInterval = gridInterval
        self.margin = margin
        self.saveAsKont = saveAsKont


class Visualizer:

    def __init__(self, params: Parameters):
        self.params = params

        self.model: Union[None, Qphar] = None
        self.pharmacophore: Union[None, Pharm.BasicPharmacophore] = None
        self.baseActivity: Union[None, float] = None
        self.outputPath: Union[None, str] = self.params.outputPath

        # define storage
        self.dimensions: Union[Tuple[int, int, int], None] = None
        self.xCoords: Union[None, np.ndarray] = None
        self.yCoords: Union[None, np.ndarray] = None
        self.zCoords: Union[None, np.ndarray] = None

    def setOutputPath(self, outputPath: str) -> str:
        if not os.path.isdir(outputPath):
            os.makedirs(outputPath)
        if not outputPath.endswith('/'):
            outputPath = '{}/'.format(outputPath)

        self.outputPath = outputPath
        return self.outputPath

    def setPharmacophore(self, pharmacophore: Pharm.BasicPharmacophore) -> Pharm.BasicPharmacophore:
        self.pharmacophore = pharmacophore
        return self.pharmacophore

    def setModel(self, model: Qphar) -> Qphar:
        self.model = model
        return self.model

    @staticmethod
    def loadModel(path: str) -> Qphar:
        model = Qphar()
        model.load(path)
        return model

    def run(self) -> None:
        if self.pharmacophore is None or self.model is None:
            raise ValueError('Pharmacophore and model must not be None')

        self.baseActivity = self.model.predict(self.pharmacophore, ignoreAlign=True)
        self.dimensions = self.getDimensions(self.pharmacophore)
        self.calculateXYZSearchSpace(self.pharmacophore)

        grids: Dict[FEATURE_TYPES, np.ndarray] = {}
        for featureType in FEATURE_TYPES:
            featuresToEvaluate = [feature for feature in self.pharmacophore
                                  if Pharm.getType(feature) == featureType.value]
            if len(featuresToEvaluate) == 0:
                # grids[featureType] = np.zeros(self.dimensions)
                continue

            grid = self.evaluateFeatures(featuresToEvaluate)
            grids[featureType] = grid

        positiveGrids: Dict[FEATURE_TYPES, np.ndarray] = {}
        negativeGrids: Dict[FEATURE_TYPES, np.ndarray] = {}
        for featureType, grid in grids.items():
            scaledGrid = self.scaleGrid(grid)
            positiveGrids[featureType] = scaledGrid[0]
            negativeGrids[featureType] = scaledGrid[1]

        self.saveGrids(positiveGrids, suffix='positive')
        self.saveGrids(negativeGrids, suffix='negative')

    def writeKontFile(self, path: str, grid: np.ndarray, gridName: str) -> None:
        with open(path, 'w') as f:
            index = 0
            gridValues = []
            for k, z in enumerate(self.zCoords):
                for i, x in enumerate(self.xCoords):
                    for j, y in enumerate(self.yCoords):
                        index += 1
                        value = grid[i, j, k]
                        gridValues.append(value)
                        f.write('{:>7}   {:>7.3f} {:>7.3f} {:>7.3f}\n'.format(index,
                                                                              x,
                                                                              y,
                                                                              z,
                                                                              ))

            f.write('{:>8}\n'.format(gridName))
            for value in gridValues:
                f.write('{:>8.3f}\n'.format(value))

    def saveGrid(self, grid: np.ndarray, featureType: FEATURE_TYPES, suffix: str = None) -> None:
        outputName = "{}{}-grid{}.{}".format(self.outputPath,
                                             featureType.name,
                                             '' if suffix is None else '-{}'.format(suffix),
                                             'csv' if not self.params.saveAsKont else 'kont')
        if self.params.saveAsKont:
            self.writeKontFile(outputName, grid, featureType.name)

        else:
            # TODO: reshape grid to (n, 3)
            # np.savetxt(outputName,
            #            grid,
            #            fmt="%f",
            #            delimiter=",",
            #            )
            raise NotImplementedError

    def saveGrids(self, grids: Dict[FEATURE_TYPES, np.ndarray], suffix: str = None) -> None:
        for featureType, grid in grids.items():
            if not np.any(grid):
                logging.info('Skipping grid {} due to no values'.format(featureType))
                continue

            self.saveGrid(grid, featureType, suffix=suffix)

    @staticmethod
    def scaleGrid(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        emptyGrid = np.zeros(grid.shape)
        if not np.any(grid):
            return emptyGrid, emptyGrid

        positiveGrid = np.where(grid > 0, grid, emptyGrid)
        negativeGrid = np.where(grid < 0, grid, emptyGrid)

        if np.sum(positiveGrid.flatten()) > 0:
            positiveScaled = positiveGrid / np.max(positiveGrid.flatten())
        else:
            positiveScaled = emptyGrid

        if np.sum(negativeGrid.flatten()) < 0:
            negativeScaled = -(-negativeGrid / np.min(negativeGrid.flatten()))
        else:
            negativeScaled = emptyGrid

        return positiveScaled, negativeScaled

    @staticmethod
    def getCoordinates(entity: Union[Pharm.BasicPharmacophore, Pharm.BasicFeature]) -> np.ndarray:
        if isinstance(entity, Pharm.BasicPharmacophore):
            coords = Math.Vector3DArray()
            Chem.get3DCoordinates(entity, coords)
            return coords.toArray(False)

        elif isinstance(entity, Pharm.BasicFeature):
            return Chem.get3DCoordinates(entity).toArray()

        else:
            raise TypeError('Unknown type {} to calculate coordinates for. Must be one of [Pharm.BasicFeature, Pharm.BasicPharmacophore]'.format(type(entity)))

    def getDimensions(self,
                      pharmacophore: Pharm.BasicPharmacophore,
                      ) -> Tuple[int, int, int]:
        coords = self.getCoordinates(pharmacophore)
        diff = np.abs(np.max(coords, axis=0) - np.min(coords, axis=0))
        dimensions = np.ceil((diff + self.params.margin) / self.params.gridInterval).astype(int)
        return dimensions[0], dimensions[1], dimensions[2]

    def calculateXYZSearchSpace(self,
                                pharmacophore: Pharm.BasicPharmacophore,
                                ) -> Tuple[np.ndarray]:
        coords = self.getCoordinates(pharmacophore)
        dimensions = self.getDimensions(pharmacophore)
        searchSpaces: List[np.ndarray] = []
        for i in range(3):
            linSpace: np.ndarray = np.linspace(int(np.floor(np.min(coords, axis=0)[i] - self.params.margin)),
                                               int(np.ceil(np.max(coords, axis=0)[i] + self.params.margin)),
                                               dimensions[i])
            searchSpaces.append(linSpace)
        self.xCoords = searchSpaces[0]
        self.yCoords = searchSpaces[1]
        self.zCoords = searchSpaces[2]
        return tuple(searchSpaces)

    def evaluateSingleFeature(self,
                              feature: Pharm.BasicFeature,
                              ) -> np.ndarray:
        activityGrid = np.zeros(self.dimensions)
        featureIndex = self.pharmacophore.getFeatureIndex(feature)
        initialCoordinates = self.getCoordinates(feature)

        modifiedPharmacophores: List[Pharm.BasicPharmacophore] = []
        gridPositions: List[Tuple[int, int, int]] = []
        for i, x in enumerate(self.xCoords):
            for j, y in enumerate(self.yCoords):
                for k, z in enumerate(self.zCoords):
                    currentCoordinates = np.array([x, y, z])
                    distance = euclideanDistance(initialCoordinates, currentCoordinates)
                    if distance > self.params.searchSpace or distance == 0:
                        continue

                    copiedPharmacophore = Pharm.BasicPharmacophore()
                    copiedPharmacophore.assign(self.pharmacophore)
                    Chem.set3DCoordinates(copiedPharmacophore.getFeature(featureIndex), currentCoordinates)
                    modifiedPharmacophores.append(copiedPharmacophore)
                    gridPositions.append((i, j, k))

        logging.info('Evaluating {} pharmacophores'.format(len(modifiedPharmacophores)))
        predictions = self.model.predict(modifiedPharmacophores, ignoreAlign=True)
        activityDifferences = predictions - self.baseActivity
        for i, indices in enumerate(gridPositions):
            activityGrid[indices] = activityDifferences[i]
        return activityGrid.round(3)

    def evaluateFeatures(self,
                         features: List[Pharm.BasicFeature],
                         ) -> np.ndarray:
        if self.dimensions is None or self.xCoords is None or self.yCoords is None or self.zCoords is None:
            raise ValueError('dimensions or searchSpaces is None.')

        grids: List[np.ndarray] = [self.evaluateSingleFeature(feature) for feature in features]
        grid = np.stack(grids, axis=-1)
        grid = np.nanmean(np.where(grid > 0, grid, np.nan), axis=-1)
        return grid


def euclideanDistance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.linalg.norm(point2-point1, ord=2)


def main(params: Parameters):
    vis = Visualizer(params)
    pharmacophore = loadPharmacophore(params.pharmacophorePath)
    vis.setPharmacophore(pharmacophore)
    vis.setModel(Visualizer.loadModel('{}{}/model/'.format(MODEL_BASE_PATH, params.target)))
    vis.run()


def parseArgs() -> Parameters:
    parser = ArgumentParser()
    parser.add_argument('-target', required=True, type=str, help='name of target')
    parser.add_argument('-pharmacophore', required=True, type=str, help='pml file of optimal pharmacophore')
    parser.add_argument('-output', required=True, type=str, help='folder name where visualization files will be saved')
    parser.add_argument('-saveAsKont', required=False, default=False, action='store_true',
                        help='whether to save files as ".kont". default is ".csv"')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    return Parameters(target=args.target,
                      pharmacophorePath=args.pharmacophore,
                      outputPath=args.output,
                      saveAsKont=args.saveAsKont,
                      )


if __name__ == '__main__':
    main(parseArgs())
