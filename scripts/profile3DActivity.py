from argparse import ArgumentParser
import logging
import os
from typing import List, Tuple, Set, Union, Dict

import CDPL.Pharm as Pharm
import CDPL.Chem as Chem
import CDPL.Math as Math
import numpy as np

from src.qphar import Qphar
from src.molecule_tools import SDFReader, mol_to_sdf
from src.pharmacophore_tools import loadPharmacophore, savePharmacophore, getPharmacophore
from visualizeActivityGrid import FEATURE_TYPES


class Activity3DProfiler:

    def __init__(self,
                 model: Qphar,
                 margin: float = 2.0,
                 gridInterval: float = 0.5,
                 searchSpace: float = 2.0,
                 defaultFeatureGeometry: Pharm.FeatureGeometry = Pharm.FeatureGeometry.SPHERE,
                 defaultFeatureTolerance: float = 1.5,
                 defaultOptionalFlag: bool = False,
                 defaultDisabledFlag: bool = False,
                 ):
        self.model = model
        self.margin = margin
        self.searchSpace = searchSpace
        self.gridInterval = gridInterval
        self.defaultFeatureGeometry = defaultFeatureGeometry
        self.defaultFeatureTolerance = defaultFeatureTolerance
        self.defaultOptionalFlag = defaultOptionalFlag
        self.defaultDisabledFlag = defaultDisabledFlag

        # define storage
        self.dimensions: Union[Tuple[int, int, int], None] = None
        self.xCoords: Union[None, np.ndarray] = None
        self.yCoords: Union[None, np.ndarray] = None
        self.zCoords: Union[None, np.ndarray] = None

    @staticmethod
    def getCoordinates(entity: Union[Pharm.BasicPharmacophore, Pharm.BasicFeature, Chem.BasicMolecule]) -> np.ndarray:
        if isinstance(entity, Pharm.BasicPharmacophore) or isinstance(entity, Chem.BasicMolecule):
            coords = Math.Vector3DArray()
            Chem.get3DCoordinates(entity, coords)
            return coords.toArray(False)

        elif isinstance(entity, Pharm.BasicFeature):
            return Chem.get3DCoordinates(entity).toArray()

        else:
            raise TypeError(
                'Unknown type {} to calculate coordinates for. Must be one of [Pharm.BasicFeature, Pharm.BasicPharmacophore]'.format(
                    type(entity)))

    def getDimensions(self,
                      entity: Union[Pharm.BasicPharmacophore, Chem.BasicMolecule],
                      ) -> Tuple[int, int, int]:
        coords = self.getCoordinates(entity)
        diff = np.abs(np.max(coords, axis=0) - np.min(coords, axis=0))
        dimensions = np.ceil((diff + self.margin) / self.gridInterval).astype(int)
        return dimensions[0], dimensions[1], dimensions[2]

    def calculateXYZSearchSpace(self,
                                entity: Union[Pharm.BasicPharmacophore, Chem.BasicMolecule],
                                ) -> Tuple[np.ndarray]:
        coords = self.getCoordinates(entity)
        dimensions = self.getDimensions(entity)
        searchSpaces: List[np.ndarray] = []
        for i in range(3):
            linSpace: np.ndarray = np.linspace(int(np.floor(np.min(coords, axis=0)[i] - self.margin)),
                                               int(np.ceil(np.max(coords, axis=0)[i] + self.margin)),
                                               dimensions[i])
            searchSpaces.append(linSpace)
        self.xCoords = searchSpaces[0]
        self.yCoords = searchSpaces[1]
        self.zCoords = searchSpaces[2]
        return tuple(searchSpaces)

    def addFeature(self,
                   pharmacophore: Pharm.BasicPharmacophore,
                   coordinates: np.ndarray,
                   featureType: FEATURE_TYPES,
                   ) -> Pharm.BasicPharmacophore:
        feature = pharmacophore.addFeature()
        Pharm.setType(feature, featureType.value)
        Chem.set3DCoordinates(feature, coordinates)
        Pharm.setGeometry(feature, self.defaultFeatureGeometry)
        Pharm.setTolerance(feature, self.defaultFeatureTolerance)
        Pharm.setOptionalFlag(feature, self.defaultOptionalFlag)
        Pharm.setDisabledFlag(feature, self.defaultDisabledFlag)
        return pharmacophore

    def probePharmacophore(self,
                           pharmacophore: Pharm.BasicPharmacophore,
                           basePrediction: float,
                           sameFeatureTypeOnly: bool = False,
                           **kwargs
                           ) -> Dict[FEATURE_TYPES, np.ndarray]:
        self.dimensions = self.getDimensions(pharmacophore)
        self.calculateXYZSearchSpace(pharmacophore)

        grids: Dict[FEATURE_TYPES, np.ndarray] = {}
        for featureType in FEATURE_TYPES:
            if featureType.value == Pharm.FeatureType.X_VOLUME:  # skip because not included in model
                continue

            logging.info('Processing {}'.format(featureType.name))
            if sameFeatureTypeOnly:
                featuresToEvaluate = [feature for feature in pharmacophore if Pharm.getType(feature) == featureType.value]
            else:
                featuresToEvaluate = [feature for feature in pharmacophore]

            if len(featuresToEvaluate) == 0:
                continue

            logging.info('Processing {} features'.format(len(featuresToEvaluate)))
            featureGrids: List[np.ndarray] = []
            for feature in featuresToEvaluate:
                grid: np.ndarray = np.zeros(self.dimensions)
                featureCoordinates = self.getCoordinates(feature)
                featureIndex = pharmacophore.getFeatureIndex(feature)

                probePharmacophores: List[Pharm.BasicPharmacophore] = []
                gridPositions: List[Tuple[int, int, int]] = []
                for i, x in enumerate(self.xCoords):
                    for j, y in enumerate(self.yCoords):
                        for k, z in enumerate(self.zCoords):
                            currentCoordinates = np.array([x, y, z])
                            distance = self.euclideanDistance(featureCoordinates, currentCoordinates)
                            if distance > self.searchSpace:
                                continue

                            tempPharmacophore = Pharm.BasicPharmacophore()
                            tempPharmacophore.assign(pharmacophore)
                            probeFeature = tempPharmacophore.getFeature(featureIndex)
                            Chem.set3DCoordinates(probeFeature, currentCoordinates)
                            if Pharm.getType(probeFeature) != featureType:
                                Pharm.setType(probeFeature, featureType.value)
                            probePharmacophores.append(tempPharmacophore)
                            gridPositions.append((i, j, k))

                logging.info('Predicting {} pharmacophores'.format(len(probePharmacophores)))
                predictions = self.model.predict(probePharmacophores, ignoreAlign=True)
                activityDifferences = predictions - basePrediction
                for i, indices in enumerate(gridPositions):
                    grid[indices] = activityDifferences[i]

                featureGrids.append(grid.round(3))

            stackedFeatureGrids: np.ndarray = np.stack(featureGrids, axis=-1)
            stackedFeatureGrids = np.nanmean(np.where(stackedFeatureGrids != 0, stackedFeatureGrids, np.nan), axis=-1)

            grids[featureType] = stackedFeatureGrids

        return grids

    def probeFullMolecule(self,
                          molecule: Chem.BasicMolecule,
                          pharmacophore: Pharm.BasicPharmacophore,
                          basePrediction: float,
                          **kwargs
                          ) -> Dict[FEATURE_TYPES, np.ndarray]:
        self.dimensions = self.getDimensions(molecule)
        self.calculateXYZSearchSpace(molecule)
        moleculeCoordinates = self.getCoordinates(molecule)

        grids: Dict[FEATURE_TYPES, np.ndarray] = {}
        for featureType in FEATURE_TYPES:
            if featureType.value == Pharm.FeatureType.X_VOLUME:  # skip because not included in model
                continue

            logging.info('Processing {}'.format(featureType.name))
            grid: np.ndarray = np.zeros(self.dimensions)
            probePharmacophores: List[Pharm.BasicPharmacophore] = []
            gridPositions: List[Tuple[int, int, int]] = []
            for i, x in enumerate(self.xCoords):
                for j, y in enumerate(self.yCoords):
                    for k, z in enumerate(self.zCoords):
                        currentCoordinates = np.array([x, y, z])
                        distances = np.linalg.norm(moleculeCoordinates - currentCoordinates, ord=2, axis=1)
                        minDistance = np.min(distances, axis=0)
                        if minDistance > self.searchSpace:
                            continue

                        tempPharmacophore = Pharm.BasicPharmacophore()
                        tempPharmacophore.assign(pharmacophore)
                        self.addFeature(tempPharmacophore, currentCoordinates, featureType)
                        probePharmacophores.append(tempPharmacophore)
                        gridPositions.append((i, j, k))

            logging.info('Predicting {} pharmacophores'.format(len(probePharmacophores)))
            predictions = self.model.predict(probePharmacophores, ignoreAlign=True)
            activityDifferences = predictions - basePrediction
            for i, indices in enumerate(gridPositions):
                grid[indices] = activityDifferences[i]

            grids[featureType] = grid

        return grids

    @staticmethod
    def euclideanDistance(point1: np.ndarray, point2: np.ndarray) -> float:
        return np.linalg.norm(point2 - point1, ord=2)

    def saveGrid(self, grid: np.ndarray, path: str, gridName: str) -> None:
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

    def saveGrids(self, grids: Dict[FEATURE_TYPES, np.ndarray], path: str, suffix: str) -> None:
        if not os.path.isdir(path):
            os.makedirs(path)

        for featureType, grid in grids.items():
            if not np.any(grid):
                logging.info('Skipping grid {} due to no values'.format(featureType))
                continue
            self.saveGrid(grid,
                          '{}{}-{}.kont'.format(path, featureType.name, suffix),
                          '{}-{}'.format(featureType.name, suffix)
                          )

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

    def splitGrids(self, grids: Dict[FEATURE_TYPES, np.ndarray]) -> Tuple[Dict[FEATURE_TYPES, np.ndarray],
                                                                          Dict[FEATURE_TYPES, np.ndarray]]:
        positiveGrids, negativeGrids = {}, {}
        for featureType, grid in grids.items():
            scaledGrid = self.scaleGrid(grid)
            positiveGrids[featureType] = scaledGrid[0]
            negativeGrids[featureType] = scaledGrid[1]
        return positiveGrids, negativeGrids

    def processPharmacophore(self,
                             pharmacophore: Pharm.BasicPharmacophore,
                             outputPath: str,
                             name: str,
                             alignPharmacophore: bool = False,
                             **kwargs
                             ) -> None:
        if alignPharmacophore:
            basePrediction, alignedPharmacophore = self.model.predict(pharmacophore, returnAlignedPharmacophores=True)
            alignedPharmacophore = alignedPharmacophore[0]
            if alignedPharmacophore is None:
                raise ValueError('Could not align pharmacophore.')
        else:
            basePrediction = self.model.predict(pharmacophore)[0]
            alignedPharmacophore = pharmacophore

        grids = self.probePharmacophore(alignedPharmacophore, basePrediction=basePrediction, **kwargs)
        logging.info('Saving grids to {}{}/'.format(outputPath, name))
        positiveGrids, negativeGrids = self.splitGrids(grids)
        self.saveGrids(positiveGrids, '{}{}/'.format(outputPath, name), 'positive')
        self.saveGrids(negativeGrids, '{}{}/'.format(outputPath, name), 'negative')
        savePharmacophore(alignedPharmacophore, '{}{}/alignedPharmacophore.pml'.format(outputPath, name))

    def processMolecules(self,
                         molecules: List[Chem.BasicMolecule],
                         outputPath: str,
                         fullMolecule: bool = False,
                         **kwargs
                         ) -> None:
        basePredictions, alignedPharmacophores = self.model.predict(molecules, returnAlignedPharmacophores=True)

        for basePred, pharm, mol in zip(basePredictions, alignedPharmacophores, molecules):
            name = Chem.getName(mol)
            logging.info('Probing molecule {}'.format(name))

            if pharm is None:
                logging.info('Could not align molecule {}. Skipping'.format(name))
                continue

            alignedMol = self.alignMoleculeToPharmacopore(mol, pharm)
            sdb = Chem.getStructureData(alignedMol)
            sdb.addEntry(' <prediction>', str(basePred))
            Chem.setStructureData(mol, sdb)
            if fullMolecule:
                grids = self.probeFullMolecule(mol, pharm, basePred, **kwargs)
            else:
                grids = self.probePharmacophore(pharm, basePrediction=basePred, **kwargs)
            logging.info('Saving grids to {}{}/'.format(outputPath, name))
            positiveGrids, negativeGrids = self.splitGrids(grids)
            self.saveGrids(positiveGrids, '{}{}/'.format(outputPath, name), 'positive')
            self.saveGrids(negativeGrids, '{}{}/'.format(outputPath, name), 'negative')
            savePharmacophore(pharm, '{}{}/alignedPharmacophore.pml'.format(outputPath, name))
            mol_to_sdf([alignedMol],
                       '{}{}/alignedMolecule.sdf'.format(outputPath, name),
                       multiconf=False)

    @staticmethod
    def alignMoleculeToPharmacopore(molecule: Chem.BasicMolecule,
                                    pharmacophore: Pharm.BasicPharmacophore,
                                    ) -> Chem.BasicMolecule:
        aligner = Pharm.PharmacophoreAlignment(True)
        scorer = Pharm.PharmacophoreFitScore()

        aligner.addFeatures(pharmacophore, True)
        bestScore, bestConf = 0, 0
        bestTfMatrix = Math.Matrix4D()
        for conf in range(Chem.getNumConformations(molecule)):
            Chem.applyConformation(molecule, conf)
            pharm = getPharmacophore(molecule, fuzzy=True)
            aligner.addFeatures(pharm, False)

            while aligner.nextAlignment():
                tfMatrix = aligner.getTransform()
                score = scorer(pharmacophore, pharm, tfMatrix)

                if score is not None:
                    if score > bestScore:
                        bestScore = score
                        bestConf = conf
                        bestTfMatrix.assign(tfMatrix)

            aligner.clearEntities(False)

        if bestTfMatrix is not None:
            Chem.applyConformation(molecule, bestConf)
            Chem.transform3DCoordinates(molecule, bestTfMatrix)
        return molecule


def main(pmlPath: str = None,
         sdfPath: str = None,
         outputPath: str = None,
         modelPath: str = None,
         **kwargs
         ):
    model = Qphar()
    model.load(modelPath)
    profiler = Activity3DProfiler(model)

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    if not outputPath.endswith('/'):
        outputPath = '{}/'.format(outputPath)

    if sdfPath is not None:
        molecules = []
        r = SDFReader(sdfPath, multiconf=True)
        for i, mol in enumerate(r):
            sdb = Chem.getStructureData(mol)
            _nr, _name = '', ''
            # name = ''
            for p in sdb:
                if 'Compound' in p.header:
                    _nr = p.data
                elif 'Name' in p.header:
                    _name = p.data
                # if 'Nr' in p.header:  # TODO: make generic
                #     name = p.data
            name = '{}_{}'.format(_nr, _name)
            Chem.setName(mol, name)
            molecules.append(mol)

        profiler.processMolecules(molecules, outputPath, **kwargs)

    if pmlPath is not None:
        pharm = loadPharmacophore(pmlPath)
        profiler.processPharmacophore(pharm, outputPath, os.path.splitext(os.path.basename(pmlPath))[0], **kwargs)


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-pml', required=False, default=None,
                        help='path to pharmacophore pml file. Must be given if sdf is not given.')
    parser.add_argument('-sdf', required=False, default=None,
                        help='path to molecule sdf file. Must be given if pml is not given.')
    parser.add_argument('-output', required=True, help='folder where results are saved')
    parser.add_argument('-model', required=True, help='path of qphar model')
    parser.add_argument('-alignPharmacophore', required=False, default=False, action='store_true',
                        help='whether to align the provided pharmacophore. False per default.')
    parser.add_argument('-sameTypeOnly', default=False, required=False, action='store_true',
                        help='whether to make activity grid for features of same type only. False per default.')
    parser.add_argument('-fullMolecule', default=False, action='store_true', required=False,
                        help='probe entire molecule instead of pharmacophore features only. SearchRadius still applies')
    args = parser.parse_args()

    if args.pml is None and args.sdf is None:
        raise ValueError('Either pml or sdf file must be provided, neither was.')

    if args.alignPharmacophore and args.pml is None:
        logging.info('No pharmacophore provided. Ignoring alignPharmacophore == True')

    if args.fullMolecule and not args.sdf:
        raise ValueError('Cannot probe full molecule if sdf not given. Please provide SDF file.')

    logging.basicConfig(level=logging.INFO)
    return args


if __name__ == '__main__':
    args = parseArgs()
    main(pmlPath=args.pml,
         sdfPath=args.sdf,
         outputPath=args.output,
         modelPath=args.model,
         alignPharmacophore=args.alignPharmacophore,
         sameFeatureTypeOnly=args.sameTypeOnly,
         fullMolecule=args.fullMolecule,
         )
