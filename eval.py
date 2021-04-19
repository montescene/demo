import sys
import numpy as np
import open3d as o3d
import os
import torch
import csv
import cv2
from copy import deepcopy
from utils import *
from pytorch3d.ops.knn import knn_points

v_level = o3d.utility.VerbosityLevel.Error
o3d.utility.set_verbosity_level(v_level)

from monte_carlo_model_search.mc_utilsObjects import *

from absl import flags
from absl import app
FLAGS = flags.FLAGS

flags.DEFINE_string('scan2cad', '/media/shreyas/ssd2/Dataset/scan2cad_download_link/full_annotations.json', 'scan2cad annotation path')
flags.DEFINE_boolean('download_scenes', False, 'Download Validation scenes (takes time)')
flags.DEFINE_string('shapenet_dir', '/media/shreyas/4aa82be1-14a8-47f7-93a7-171e3ebac2b0/Datasets/ShapeNetCore.v2', 'shapenet dir')


IOUThresh = 0.5
COMPARE_WITH = 'mcts'#'votenet_baseline' #'mcts'
model2scanCoordinateChangeMatrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32).T
#left multiplying by this makes the model in scannet aligned coordinate system

finalCandidatesFileName = 'FinalCandidates.pickle'

class MCTSObjs():
    def __init__(self, sceneID, runName='monte_carlo', finalCandidatesFileName = 'FinalCandidates.pickle'):
        mctsPickleFile = join(SCANS_DIR, sceneID, runName, finalCandidatesFileName)

        with open(mctsPickleFile, 'rb') as f:
            self.candList = pickle.load(f)

        self.sceneID = sceneID
        self.sceneMesh = o3d.io.read_triangle_mesh(join(SCANS_DIR, sceneID, '%s_vh_clean_2.ply' % (sceneID)))
        self.sceneMesh = alignPclMesh(sceneID, self.sceneMesh)


        self.predObjList = self.getMCTSObjs()

        self.iouAllCands, self.candIntersectionPairsCnt = 0, 0


    def getMCTSObjs(self):
        # all objects are in axis aligned coordinate system
        axis_align_matrix = getAxisAlignmentMat(self.sceneID)
        mctsObjectsList = []
        for ii, cand in enumerate(self.candList[0:]):
            assert isinstance(cand, ObjCandidate)
            if 'ESC' in cand.modelID or 'END' in cand.modelID:
                continue
            catID = cand.modelID.split('/')[0]
            modelID = cand.modelID.split('/')[1]

            catID = getStandardShapenetCatID(catID, modelID, FLAGS.shapenet_dir)
            objMesh = o3d.io.read_triangle_mesh(join(FLAGS.shapenet_dir,
                                                     catID, modelID,
                                                     'models', 'model_normalized.obj'))
            objVert = np.asarray(objMesh.vertices)
            objPoseMat = cand.objPoseMat
            bb3DRest = getObj3DBB(objVert)

            scale = cand.scale.copy()
            scale = scale / np.abs(np.array(
                [bb3DRest[2, 0] - bb3DRest[0, 0], bb3DRest[1, 1] - bb3DRest[0, 1],
                 bb3DRest[4, 2] - bb3DRest[0, 2]]))
            S = np.eye(4)
            S[0:3, 0:3] = np.diag(scale)

            q = quaternion.from_rotation_matrix(objPoseMat[:3, :3]).components
            q = np.quaternion(q[0], q[1], q[2], q[3])
            R = np.eye(4)
            R[0:3, 0:3] = quaternion.as_rotation_matrix(q)

            T = np.eye(4)
            T[0:3, 3] = objPoseMat[:3, 3]

            M = T.dot(R).dot(S)

            objMesh = objMesh.transform(M)
            bb3DTrans = bb3DRest.dot(M.T[:3, :3]) + M[:3, 3]

            scaleForUnitBB = np.array([scale[0], scale[2], scale[1]])
            RotTransMat = T.dot(R)
            RotTransMat = RotTransMat.dot(model2scanCoordinateChangeMatrix.T)
            rot = cv2.Rodrigues(RotTransMat[:3, :3])[0].squeeze()  # [2]
            trans = RotTransMat[:3, 3]
            objBBRestZUp = bb3DRest.dot(model2scanCoordinateChangeMatrix.T[:3, :3])
            objBBRestZupCenter = np.mean(objBBRestZUp, axis=0)
            objBBRestZUpCentered = objBBRestZUp - objBBRestZupCenter
            objBBRestZupCenteredScale = np.array(
                [np.max(objBBRestZUpCentered[:, 0]) - np.min(objBBRestZUpCentered[:, 0]),
                 np.max(objBBRestZUpCentered[:, 1]) - np.min(objBBRestZUpCentered[:, 1]),
                 np.max(objBBRestZUpCentered[:, 2]) - np.min(objBBRestZUpCentered[:, 2])])
            normalizedBBScale = scaleForUnitBB * objBBRestZupCenteredScale
            normalizedBBTrans = RotTransMat[:3, :3].dot(np.diag(scaleForUnitBB)).dot(objBBRestZupCenter) + trans
            normalizedBBRot = rot

            unitBB = np.array([[-0.5, -0.5, -0.5],
                               [-0.5, -0.5, 0.5],
                               [0.5, -0.5, -0.5],
                               [0.5, -0.5, 0.5],
                               [-0.5, 0.5, -0.5],
                               [-0.5, 0.5, 0.5],
                               [0.5, 0.5, -0.5],
                               [0.5, 0.5, 0.5]
                               ])
            orientedBB = (unitBB * normalizedBBScale).dot(cv2.Rodrigues(normalizedBBRot)[0].T) + normalizedBBTrans

            objMeshNew = o3d.geometry.TriangleMesh()
            vertTrans = np.array(objMesh.vertices)
            objMeshNew.vertices = o3d.utility.Vector3dVector(vertTrans)
            objMeshNew.triangles = objMesh.triangles
            vertCols = vertTrans[:, :3] - np.mean(vertTrans[:, :3], 0)
            objMeshNew.vertex_colors = o3d.utility.Vector3dVector(
                vertCols - np.min(vertCols, 0) / (np.max(vertCols, 0) - np.min(vertCols, 0)))

            
            
            mctsObj = s2cAnnoData(modelID=join(catID, modelID), orientedBB=orientedBB,
                                  objPoseMat=M, catName=ShapenetIDToName[catID], transMesh=objMeshNew,
                        pointsInsideBB=None,
                                  rot=normalizedBBRot, scale=normalizedBBScale, trans=normalizedBBTrans)

            mctsObjectsList.append(mctsObj)

        return mctsObjectsList



class S2CAnno():
    def __init__(self, sceneID):
        filename_json = FLAGS.scan2cad
        self.jsonAnno = jsonRead(filename_json)

        self.sceneID = sceneID
        self.sceneMesh = o3d.io.read_triangle_mesh(join(SCANS_DIR, sceneID, '%s_vh_clean_2.ply' % (sceneID)))
        self.sceneMesh = alignPclMesh(sceneID, self.sceneMesh)
        self.segMeshGtNyu = o3d.io.read_point_cloud(join(SCANS_DIR, sceneID, '%s_vh_clean_2.labels.ply' % (sceneID)))
        self.segMeshGtNyu = alignPclMesh(sceneID, self.segMeshGtNyu)

        self.s2cAnnoObjList = self.getScan2CadAnnotations()
        self.s2cAnnoObjListOrig = deepcopy(self.s2cAnnoObjList)




    def getScan2CadAnnotations(self):
        # all objects are in axis aligned coordinate system
        axis_align_matrix = getAxisAlignmentMat(self.sceneID)
        s2cObjectsList = []
        for anno in self.jsonAnno:
            if self.sceneID == anno['id_scan']:
                t = anno["trs"]["translation"]
                q = anno["trs"]["rotation"]
                s = anno["trs"]["scale"]
                MScene = make_M_from_tqs(t, q, s)

                for model in anno['aligned_models']:
                    catID = model['catid_cad']
                    modelID = model['id_cad']
                    if catID not in ShapenetIDToName.keys():
                        continue
                    t = model["trs"]["translation"]
                    q = model["trs"]["rotation"]
                    s = model["trs"]["scale"]
                    Mcad = make_M_from_tqs(t, q, s)
                    Mcad = axis_align_matrix.dot(np.linalg.inv(MScene).dot(Mcad))

                    objMesh = o3d.io.read_triangle_mesh(
                        join(FLAGS.shapenet_dir, catID, modelID, 'models', 'model_normalized.obj'))
                    objBBRest = getObj3DBB(np.asarray(objMesh.vertices))

                    scale = np.array([s[0], s[2], s[1]])
                    T = np.eye(4)
                    T[0:3, 3] = t
                    R = np.eye(4)
                    q = np.quaternion(q[0], q[1], q[2], q[3])
                    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
                    RotTransMat = axis_align_matrix.dot(np.linalg.inv(MScene).dot(T.dot(R)))
                    RotTransMat = RotTransMat.dot(model2scanCoordinateChangeMatrix.T)
                    rot = cv2.Rodrigues(RotTransMat[:3,:3])[0].squeeze()#[2]
                    trans = RotTransMat[:3,3]
                    objBBRestZUp = objBBRest.dot(model2scanCoordinateChangeMatrix.T[:3,:3])
                    objBBRestZupCenter = np.mean(objBBRestZUp, axis=0)
                    objBBRestZUpCentered = objBBRestZUp - objBBRestZupCenter
                    objBBRestZupCenteredScale = np.array([np.max(objBBRestZUpCentered[:, 0]) - np.min(objBBRestZUpCentered[:, 0]),
                                                  np.max(objBBRestZUpCentered[:, 1]) - np.min(objBBRestZUpCentered[:, 1]),
                                                  np.max(objBBRestZUpCentered[:, 2]) - np.min(objBBRestZUpCentered[:, 2])])
                    normalizedBBScale = scale* objBBRestZupCenteredScale
                    normalizedBBTrans = RotTransMat[:3,:3].dot(np.diag(scale)).dot(objBBRestZupCenter) + trans
                    normalizedBBRot = rot

                    unitBB = np.array([[-0.5, -0.5, -0.5],
                                       [-0.5, -0.5, 0.5],
                                       [0.5, -0.5, -0.5],
                                       [0.5, -0.5, 0.5],
                                       [-0.5, 0.5, -0.5],
                                       [-0.5, 0.5, 0.5],
                                       [0.5, 0.5, -0.5],
                                       [0.5, 0.5, 0.5]
                                       ])
                    orientedBB = (unitBB*normalizedBBScale).dot(cv2.Rodrigues(normalizedBBRot)[0].T) + normalizedBBTrans

                    CenteringMat = np.eye(4)
                    CenteringMat[:3,3] = objBBRestZupCenter

                    objBBRest = np.concatenate([objBBRest, np.ones((objBBRest.shape[0], 1))], axis=1)
                    gtBB = objBBRest.dot(Mcad.T)[:, :3]
                    objMesh = objMesh.transform(Mcad)

                    objMeshNew = o3d.geometry.TriangleMesh()
                    vertTrans = np.array(objMesh.vertices)
                    objMeshNew.vertices = o3d.utility.Vector3dVector(vertTrans)
                    objMeshNew.triangles = objMesh.triangles
                    vertCols = vertTrans[:, :3] - np.mean(vertTrans[:, :3], 0)
                    objMeshNew.vertex_colors = o3d.utility.Vector3dVector(
                        vertCols - np.min(vertCols, 0) / (np.max(vertCols, 0) - np.min(vertCols, 0)))

                    nyuLabelID = nyuName2ID[ShapenetIDToName[catID]]
                    if self.sceneID == 'scene0222_00' and catID == '04256520':
                        nyuLabelID = nyuName2ID['bed']

                    if nyuLabelID not in aliasDict.keys():
                        continue

                    s2cObj = s2cAnnoData(modelID=join(catID, modelID), orientedBB=orientedBB,#gtBB.copy(),
                                         objPoseMat=Mcad, catName=ShapenetIDToName[catID], transMesh=objMeshNew,
                                         pointsInsideBB=getPointsInsideOrientedBB(self.segMeshGtNyu, gtBB, nyuLabelList=aliasDict[nyuLabelID])[0],
                                         rot=normalizedBBRot, scale=normalizedBBScale, trans=normalizedBBTrans)

                    s2cObjectsList.append(s2cObj)

        return s2cObjectsList


    def getBestIOUObjAnno(self, obj):
        '''
        Get the matching object in the scan2cad annotation and return the IOU and chamfer distance with that
        :param obj: query object
        :return:
        '''
        assert isinstance(obj, s2cAnnoData)

        bestIOU = 0
        bestObjInd = None
        for i, s2cObj in enumerate(self.s2cAnnoObjList):
            iou, _, _ = getOrientedBBIntersection(s2cObj.orientedBB.copy(), obj.orientedBB.copy(), isAdjustBottomZ=True)
            if iou > bestIOU:
                bestIOU = iou
                bestObjInd = i

        if bestIOU >IOUThresh:
            bestObj = self.s2cAnnoObjList.pop(bestObjInd)
            # self.visAnno(bestObj, addBB=[obj.orientedBB])
            # self.visAnno(obj)
            gtModelPoints = np.expand_dims(bestObj.transMesh.sample_points_uniformly(10000).points, 0)
            gtModelPoints = torch.tensor(gtModelPoints).cuda()

            predModelPoints = np.expand_dims(obj.transMesh.sample_points_uniformly(10000).points, 0)
            predModelPoints = torch.tensor(predModelPoints).cuda()

            gtScanPoints = torch.tensor(np.expand_dims(bestObj.pointsInsideBB, 0)).cuda()

            def getChamferDist(x,y):
                '''
                Computer chamfer distance
                :param x:
                :param y:
                :return:
                '''
                xlengths = torch.full(
                    (x.shape[0],), x.shape[1], dtype=torch.int64, device=x.device
                )
                ylengths = torch.full(
                    (y.shape[0],), y.shape[1], dtype=torch.int64, device=y.device
                )
                x_nn = knn_points(x, y, lengths1=xlengths, lengths2=ylengths, K=1)
                y_nn = knn_points(y, x, lengths1=ylengths, lengths2=xlengths, K=1)

                cham_x = x_nn.dists[..., 0]  # (N, P1)
                cham_y = y_nn.dists[..., 0]  # (N, P2)

                return cham_x, cham_y


            s2cChamfer = getChamferDist(gtScanPoints, gtModelPoints)[0].cpu().detach().numpy().squeeze()
            s2cChamfer = np.mean(np.sort(s2cChamfer))

            predChamfer = getChamferDist(gtScanPoints, predModelPoints)[0].cpu().detach().numpy().squeeze()
            predChamfer = np.mean(np.sort(predChamfer))#[-numPoints:])

            if np.isnan(s2cChamfer):
                print('Mismatch between Scan2Cad catID and ScanNet catID for scene %s'%(self.sceneID))
                s2cChamfer = 0.
                predChamfer = 0.

            return bestIOU, bestObj, s2cChamfer, predChamfer
        else:
            return None, None, None, None



def main(argv):
    if FLAGS.download_scenes:
        downloadValScenes()

    sceneIDList = []
    with open('scannetv2_val.txt', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            sceneIDList.append(row[0].strip())

    sceneIDList = [s for s in sceneIDList if '_00' in s]


    truePositiveCntAllScenes = {'chair': 0,
                       'table': 0,
                       'sofa': 0,
                       'bed': 0}
    falsePositiveCntAllScenes = truePositiveCntAllScenes.copy()
    falseNegativeCntAllScenes = truePositiveCntAllScenes.copy()
    catAPAllScenes = truePositiveCntAllScenes.copy()
    catARAllScenes = truePositiveCntAllScenes.copy()

    s2cChamferDistCatAllScenes = {'chair': 0,
                         'table': 0,
                         'sofa': 0,
                         'bed': 0}
    mctsChamferDistCatAllScenes = s2cChamferDistCatAllScenes.copy()

    rotErr = {'chair': 0,
                         'table': 0,
                         'sofa': 0,
                         'bed': 0}
    scaleErr = {'chair': np.zeros((3,)),
                         'table': np.zeros((3,)),
                         'sofa': np.zeros((3,)),
                         'bed': np.zeros((3,))}
    transErr = {'chair': 0,
                         'table': 0,
                         'sofa': 0,
                         'bed': 0}


    for sceneID in sceneIDList:
        # sceneID = 'scene0207_00'


        if sceneID in ['scene0414_00', 'scene0019_00']:
            continue

        # if sceneID not in CHALLENGING_SCENES_LIST:
        #     continue

        if not os.path.exists(join(SCANS_DIR, sceneID, 'monte_carlo', finalCandidatesFileName)):
            print('%s has no monte carlo results..'%(sceneID))
            continue


        downloadScanNetScene(sceneID)

        print('Getting eval metrics for %s...'%(sceneID))

        # get scan2cad annotations
        s2c = S2CAnno(sceneID)

        # get MCSS outputs
        predObjs = MCTSObjs(sceneID)


        # some inits
        truePositiveCnt = {'chair':0,
                           'table':0,
                           'sofa':0,
                           'bed':0}
        falsePositiveCnt = truePositiveCnt.copy()
        falseNegativeCnt = truePositiveCnt.copy()
        catAP = truePositiveCnt.copy()
        catAR = truePositiveCnt.copy()

        s2cChamferDistCat = {'chair':0,
                           'table':0,
                           'sofa':0,
                           'bed':0}
        mctsChamferDistCat = s2cChamferDistCat.copy()


        # get TP, FP and FN
        for mctsObj in predObjs.predObjList:
            # get the corresponding object in scan2cad annotations
            bestIOU, bestS2CObj, s2cChamfer, predChamfer = s2c.getBestIOUObjAnno(mctsObj)

            if bestIOU is None:
                falsePositiveCnt[mctsObj.catName] += 1
            else:
                # aggregate the results
                truePositiveCnt[mctsObj.catName] += 1
                s2cChamferDistCat[mctsObj.catName] += s2cChamfer
                mctsChamferDistCat[mctsObj.catName] += predChamfer

                rotErr[mctsObj.catName] += abs(calc_rotation_diff(quaternion.from_rotation_vector(bestS2CObj.rot), quaternion.from_rotation_vector(mctsObj.rot)))
                transErr[mctsObj.catName] += np.linalg.norm(bestS2CObj.trans-mctsObj.trans)
                scaleErr[mctsObj.catName] += np.abs(bestS2CObj.scale-mctsObj.scale)

        for s2cObj in s2c.s2cAnnoObjList:
            if s2cObj.catName in falseNegativeCnt.keys():
                falseNegativeCnt[s2cObj.catName] += 1


        # for all the scenes (aggregation)
        for catName in s2cChamferDistCat.keys():
            s2cChamferDistCatAllScenes[catName] += s2cChamferDistCat[catName]
            mctsChamferDistCatAllScenes[catName] += mctsChamferDistCat[catName]

        # for the current scene
        for catName in s2cChamferDistCat.keys():
            if truePositiveCnt[catName] > 0:
                s2cChamferDistCat[catName] = s2cChamferDistCat[catName] / truePositiveCnt[catName]
                mctsChamferDistCat[catName] = mctsChamferDistCat[catName] / truePositiveCnt[catName]
            else:
                s2cChamferDistCat[catName] = np.nan
                mctsChamferDistCat[catName] = np.nan

        # for all the scenes (aggregation)
        for catName in catAP.keys():
            truePositiveCntAllScenes[catName] += truePositiveCnt[catName]
            falsePositiveCntAllScenes[catName] += falsePositiveCnt[catName]
            falseNegativeCntAllScenes[catName] += falseNegativeCnt[catName]


        # for the current scene
        for catName in catAP.keys():
            if (truePositiveCnt[catName] + falsePositiveCnt[catName]) > 0:
                catAP[catName] = truePositiveCnt[catName] / (truePositiveCnt[catName] + falsePositiveCnt[catName])
            else:
                catAP[catName] = np.nan

            if (truePositiveCnt[catName] + falseNegativeCnt[catName]) > 0:
                catAR[catName] = truePositiveCnt[catName] / (truePositiveCnt[catName] + falseNegativeCnt[catName])
            else:
                catAR[catName] = np.nan

        # dump per scene stats in the output folder
        outDir = join(SCANS_DIR, sceneID, 'monte_carlo', 'evalMetricsMCTS')

        if not os.path.exists(outDir):
            os.mkdir(outDir)
        with open(join(outDir, 's2cChamferDistCat.json'), 'w') as f:
            f.write(json.dumps({'myDict':s2cChamferDistCat}))
        with open(join(outDir, 'mctsChamferDistCat.json'), 'w') as f:
            f.write(json.dumps({'myDict':mctsChamferDistCat}))
        with open(join(outDir, 'catAP.json'), 'w') as f:
            f.write(json.dumps({'myDict':catAP}))
        with open(join(outDir, 'catAR.json'), 'w') as f:
            f.write(json.dumps({'myDict':catAR}))


    # for all scenes (aggregation)
    for catName in s2cChamferDistCatAllScenes.keys():
        if truePositiveCntAllScenes[catName] > 0:
            s2cChamferDistCatAllScenes[catName] = s2cChamferDistCatAllScenes[catName] / truePositiveCntAllScenes[catName]
            mctsChamferDistCatAllScenes[catName] = mctsChamferDistCatAllScenes[catName] / truePositiveCntAllScenes[catName]
        else:
            s2cChamferDistCatAllScenes[catName] = np.nan
            mctsChamferDistCatAllScenes[catName] = np.nan

    for catName in catAPAllScenes.keys():
        if (truePositiveCntAllScenes[catName] + falsePositiveCntAllScenes[catName]) > 0:
            catAPAllScenes[catName] = truePositiveCntAllScenes[catName] / (truePositiveCntAllScenes[catName] + falsePositiveCntAllScenes[catName])
        else:
            catAPAllScenes[catName] = np.nan

        if (truePositiveCntAllScenes[catName] + falseNegativeCntAllScenes[catName]) > 0:
            catARAllScenes[catName] = truePositiveCntAllScenes[catName] / (truePositiveCntAllScenes[catName] + falseNegativeCntAllScenes[catName])
        else:
            catARAllScenes[catName] = np.nan


    for catName in s2cChamferDistCatAllScenes.keys():
        if truePositiveCntAllScenes[catName] > 0:
            rotErr[catName] = rotErr[catName] / truePositiveCntAllScenes[catName]
            transErr[catName] = transErr[catName] / truePositiveCntAllScenes[catName]
            scaleErr[catName] = scaleErr[catName] / truePositiveCntAllScenes[catName]
        else:
            s2cChamferDistCatAllScenes[catName] = np.nan
            mctsChamferDistCatAllScenes[catName] = np.nan
        scaleErr[catName] = list(scaleErr[catName])


    # dump outputs for all scenes
    outDir = 'outputs/evalAllScenesMCTS_testrun_%fIOU'%(IOUThresh)

    if not os.path.exists(outDir):
        os.mkdir(outDir)
    with open(join(outDir, 's2cChamferDistCat.json'), 'w') as f:
        f.write(json.dumps({'myDict': s2cChamferDistCatAllScenes}))
    with open(join(outDir, 'mctsChamferDistCat.json'), 'w') as f:
        f.write(json.dumps({'myDict': mctsChamferDistCatAllScenes}))
    with open(join(outDir, 'rotErrors.json'), 'w') as f:
        f.write(json.dumps({'myDict': rotErr}))
    with open(join(outDir, 'transErrors.json'), 'w') as f:
        f.write(json.dumps({'myDict': transErr}))
    with open(join(outDir, 'scaleErrors.json'), 'w') as f:
        f.write(json.dumps({'myDict': scaleErr}))



    with open(join(outDir, 'catAP.json'), 'w') as f:
        f.write(json.dumps({'myDict': catAPAllScenes}))
    with open(join(outDir, 'catAR.json'), 'w') as f:
        f.write(json.dumps({'myDict': catARAllScenes}))

    with open(join(outDir, 'truePositiveCntAllScenes.json'), 'w') as f:
        f.write(json.dumps({'myDict': truePositiveCntAllScenes}))
    with open(join(outDir, 'falseNegativeCntAllScenes.json'), 'w') as f:
        f.write(json.dumps({'myDict': falseNegativeCntAllScenes}))
    with open(join(outDir, 'falsePositiveCntAllScenes.json'), 'w') as f:
        f.write(json.dumps({'myDict': falsePositiveCntAllScenes}))


if __name__ == '__main__':
    app.run(main)



