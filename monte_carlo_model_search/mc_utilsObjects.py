import numpy as np
import open3d as o3d
import os
from os.path import join
import quaternion
import pickle
from collections import namedtuple
from monte_carlo_model_search.line_mesh import LineMesh
from enum import Enum
from monte_carlo_model_search.nyuName2ID import nyuName2ID
from shapely.geometry import Polygon
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

SCANS_DIR= 'outputs/scans/'
# SHAPENETCOREV2_DIR = '/media/shreyas/4aa82be1-14a8-47f7-93a7-171e3ebac2b0/Datasets/ShapeNetCore.v2'

class CandidateType(Enum):
    OBJECT = 1
    LAYOUT = 2
    SKIP = 3
    END = 4


ObjCandidate_nt = namedtuple('ObjCandidate',
                           ['modelID',
                            'gridSearchFrameIDs',
                            'objPoseMat',
                            'scale',
                            'orientedBB',
                            'objLabelNYUFinal',
                            'transMesh',
                            'bbAxisAligned',
                            'bbInstanceID',
                            'viewRenderings',
                            # 'priorScore',
                            'candType',
                            'bb2dPerView',
                            'viewWeights'])
ObjCandidate_nt.__new__.__defaults__ = (None,) * len(ObjCandidate_nt._fields)



# ShapenetIDToName = {'03001627': 'chair',
#                      '04379243': 'desk',
#                      '04256520': 'sofa',
#                      '02818832': 'bed'}

ShapenetIDToName = {'03001627': 'chair',
                                 '04379243': 'table',
                                 '04256520': 'sofa',
                                 '02818832': 'bed',
                    '02808440': 'bathtub'}

aliasDict = {nyuName2ID['chair']: [nyuName2ID['sofa'], nyuName2ID['chair'], nyuName2ID['toilet'], nyuName2ID['otherprop']],
             nyuName2ID['sofa']: [nyuName2ID['chair'], nyuName2ID['sofa']],
             nyuName2ID['table']: [nyuName2ID['desk'], nyuName2ID['table']],
             nyuName2ID['desk']: [nyuName2ID['desk'], nyuName2ID['table']],
             nyuName2ID['bed']: [nyuName2ID['bed']]}

s2cAnnoData = namedtuple('s2cData',
                         ['modelID',
                          'orientedBB',
                          'objPoseMat',
                          'catName',
                          'transMesh',
                          'pointsInsideBB',
                          'scale',
                          'trans',
                          'rot'
                         ])
s2cAnnoData.__new__.__defaults__ = (None,) * len(s2cAnnoData._fields)




class ObjCandidate(ObjCandidate_nt):
    def __init__(self, **kwargs):
        super(ObjCandidate_nt, self).__init__()

        self.candName = getCandName(self)

        self._hash = hash(self.candName)

        self.priorScore = 0.1

    def __eq__(self, other):
        return self._hash == other._hash

    def __hash__(self):
        return self._hash

    def replace(self, **kwargs):
        new_ObjCandidate = self._replace(**kwargs)
        new_ObjCandidate._hash = hash(getCandName(new_ObjCandidate))
        new_ObjCandidate.priorScore = self.priorScore
        new_ObjCandidate.candName = getCandName(new_ObjCandidate)

        return new_ObjCandidate

    @staticmethod
    def copy_without_mesh(other):
        other_obj_dict = other._asdict()
        other_obj_dict.pop('transMesh', None)
        new_object = ObjCandidate(transMesh=None, **other_obj_dict)
        return new_object

def getCandName(candidate):
    assert isinstance(candidate, ObjCandidate)
    return candidate.modelID + '_' + str(candidate.bbInstanceID)


class MCTSObjs():
    def __init__(self, sceneID, SHAPENETCOREV2_DIR):
        self.SHAPENETCOREV2_DIR = SHAPENETCOREV2_DIR
        mctsPickleFile = join(SCANS_DIR, sceneID, 'monte_carlo', 'FinalCandidates.pickle')

        with open(mctsPickleFile, 'rb') as f:
            self.candList = pickle.load(f)

        self.sceneID = sceneID
        self.sceneMesh = o3d.io.read_triangle_mesh(join(SCANS_DIR, sceneID, '%s_vh_clean_2.ply' % (sceneID)))
        self.sceneMesh = alignPclMesh(sceneID, self.sceneMesh)

        # self.segMeshMinkNyu = o3d.io.read_point_cloud(join(SEGMENTATIONS_DIR, sceneID, '%s_segmented.ply' % (sceneID)))
        # self.segMeshMinkNyu = alignPclMesh(sceneID, self.segMeshMinkNyu)

        if self.sceneID == 'scene0169_00' and False:
            segmentPclCols = np.array(self.segMeshMinkNyu.colors)
            segmentPclCols[
                np.round(segmentPclCols[:, 0] * 255) == nyuName2ID['cabinet']] = nyuName2ID['table'] / 255.
            self.segMeshMinkNyu.colors = o3d.utility.Vector3dVector(segmentPclCols)

        self.mctsObjList = self.getMCTSObjs()

    def getMCTSObjs(self):
        # all objects are in axis aligned coordinate system
        axis_align_matrix = getAxisAlignmentMat(self.sceneID)
        mctsObjectsList = []
        cmap = label_colormap(len(self.candList) + 1)
        # cmap = paster_cmap()

        for cand_ind, cand in enumerate(self.candList):
            assert isinstance(cand, ObjCandidate)
            if 'ESC' in cand.modelID or 'END' in cand.modelID:
                continue
            catID = cand.modelID.split('/')[0]
            modelID = cand.modelID.split('/')[1]

            catID = getStandardShapenetCatID(catID, modelID, self.SHAPENETCOREV2_DIR)
            objMesh = o3d.io.read_triangle_mesh(join(self.SHAPENETCOREV2_DIR,
                                                     catID, modelID,
                                                     'models', 'model_normalized.obj'))
            objVert = np.asarray(objMesh.vertices)
            objPoseMat = cand.objPoseMat
            bb3DRest = getObj3DBB(objVert)

            scale = cand.scale
            # scale = np.array([scale[0], scale[2], scale[1]])
            scale = scale / np.abs(np.array(
                [bb3DRest[2, 0] - bb3DRest[0, 0], bb3DRest[1, 1] - bb3DRest[0, 1],
                 bb3DRest[4, 2] - bb3DRest[0, 2]]))
            S = np.eye(4)
            S[0:3, 0:3] = np.diag(scale)

            q = quaternion.from_rotation_matrix(objPoseMat[:3, :3]).components
            # cv2.Rodrigues(bb.optResults.objPoseMat[:3, :3])[0].squeeze()).as_quat()
            q = np.quaternion(q[0], q[1], q[2], q[3])
            R = np.eye(4)
            R[0:3, 0:3] = quaternion.as_rotation_matrix(q)

            T = np.eye(4)
            T[0:3, 3] = objPoseMat[:3, 3]

            M = T.dot(R).dot(S)

            objMesh = objMesh.transform(M)
            bb3DTrans = bb3DRest.dot(M.T[:3, :3]) + M[:3, 3]

            objMeshNew = o3d.geometry.TriangleMesh()
            vertTrans = np.array(objMesh.vertices)
            objMeshNew.vertices = o3d.utility.Vector3dVector(vertTrans)
            objMeshNew.triangles = objMesh.triangles
            vertCols = vertTrans[:, :3] - np.mean(vertTrans[:, :3], 0)
            objMeshNew.vertex_colors = o3d.utility.Vector3dVector(
                vertCols - np.min(vertCols, 0) / (np.max(vertCols, 0) - np.min(vertCols, 0)))

            nyuLabelID = nyuName2ID[ShapenetIDToName[catID]]

            # pointsInsideBB = \
            #     getPointsInsideOrientedBB(self.segMeshMinkNyu, bb3DTrans, nyuLabelList=aliasDict[nyuLabelID])[0]

            pointsInsideBB = None

            if False:
                sceneMeshInsideBBMask = \
                    getPointsInsideOrientedBB(self.sceneMesh, bb3DTrans)[1]
                sceneMeshInsideBB = o3d.geometry.PointCloud()
                sceneMeshInsideBB.points = o3d.utility.Vector3dVector(
                    np.array(self.sceneMesh.vertices)[sceneMeshInsideBBMask])
                sceneMeshInsideBB.colors = o3d.utility.Vector3dVector(
                    np.array(self.sceneMesh.vertex_colors)[sceneMeshInsideBBMask])
                pcdInsideBB = transferSegBtwPcds(sceneMeshInsideBB, pointsInsideBB)

                pointsInsideBBCols = np.array(pcdInsideBB.colors)
                badColsMask = np.abs(np.mean(pointsInsideBBCols, 1) - np.mean(np.mean(pointsInsideBBCols, 1))) > 0.15
                pointsInsideBBCols[badColsMask] = 0.5 * np.mean(pointsInsideBBCols, 0) + 0.5 * pointsInsideBBCols[
                    badColsMask]
                pcdInsideBB.colors = o3d.utility.Vector3dVector(pointsInsideBBCols)

                objMeshNew = transferSegBtwPcds(pcdInsideBB, objMeshNew)

                vertCols = np.array(objMeshNew.vertices)[:, :3] - np.mean(np.array(objMeshNew.vertices)[:, :3], 0)
                vertCols = vertCols - np.min(vertCols, 0) / (np.max(vertCols, 0) - np.min(vertCols, 0))
                vertCols = vertCols - 0.5
                vertCols = 0.2 * vertCols + np.array(objMeshNew.vertex_colors) - 0.15  # + np.array([0.,0.,0.15])

                objMeshNew.vertex_colors = o3d.utility.Vector3dVector(np.clip(vertCols, 0, 1))
            elif False:
                clr = cmap[cand_ind + 1]

                vertex_n = np.array(objMeshNew.vertex_colors).shape[0]
                vertex_colors = np.ones((vertex_n, 3)) * clr[None, :]
                objMeshNew.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            elif True:
                clr = cmap[np.random.randint(0, cmap.shape[0])]
                vertCols = np.array(objMeshNew.vertices)[:, :3] - np.mean(np.array(objMeshNew.vertices)[:, :3], 0)
                vertCols = vertCols - np.min(vertCols, 0) / (np.max(vertCols, 0) - np.min(vertCols, 0))
                # vertCols = vertCols - 0.5
                # vertCols = vertCols*0.6 + clr
                objMeshNew.vertex_colors = o3d.utility.Vector3dVector(np.clip(vertCols, 0, 1))
            elif False:
                cmap = label_colormap(40 + 1)
                from preprocessing.layout import sem_seg_utils as ss
                vertCols = np.array(objMeshNew.vertices) * 0 + ss.SCANNET_COLORMAP[nyuLabelID][None, :]
                objMeshNew.vertex_colors = o3d.utility.Vector3dVector(np.clip(vertCols, 0, 1))

            # objMeshNew.compute_vertex_normals()
            # objMeshNew.paint_uniform_color([1, 0.706, 0])
            # objMeshNew.vertex_colors = objMeshNew.vertex_normals

            mctsObj = s2cAnnoData(modelID=join(catID, modelID), orientedBB=bb3DTrans.copy(), objPoseMat=M,
                                  catName=ShapenetIDToName[catID], transMesh=objMeshNew,
                                  pointsInsideBB=pointsInsideBB)

            mctsObjectsList.append(mctsObj)
            # self.visAnno(mctsObj)

        return mctsObjectsList

    def visAnno(self, mctsObj=None):
        if mctsObj is not None:
            assert isinstance(mctsObj, s2cAnnoData)
            geometryList = []
            lineSets = drawOpen3dLines([mctsObj.orientedBB])
            for l in lineSets:
                geometryList.append(l)

            geometryList.append(self.sceneMesh)
            o3d.visualization.draw_geometries(geometryList)
            o3dPcl = o3d.geometry.PointCloud()
            o3dPcl.points = o3d.utility.Vector3dVector(mctsObj.pointsInsideBB)
            o3dPcl.colors = o3d.utility.Vector3dVector(np.zeros((mctsObj.pointsInsideBB.shape[0], 3)) * 1.0)
            # o3d.visualization.draw_geometries([o3dPcl])
            o3d.visualization.draw_geometries([mctsObj.transMesh])
        else:
            geometryList = []
            for obj in self.mctsObjList:
                geometryList.append(obj.transMesh)
            o3d.visualization.draw_geometries(geometryList)


def getPointsInsideOrientedBB(pcl, bb, nyuLabelList=None):
    nyu2palette = paletteNyu()
    if isinstance(pcl, o3d.geometry.TriangleMesh):
        points = np.array(pcl.vertices).copy()
        colors = (np.array(pcl.vertex_colors)*255).astype(np.int)
    elif isinstance(pcl, o3d.geometry.PointCloud):
        points = np.array(pcl.points).copy()
        colors = (np.array(pcl.colors).copy()*255).astype(np.int)
    elif isinstance(pcl, np.ndarray):
        points = pcl.copy()
        colors = None
    else:
        raise NotImplementedError

    # assumes [0,2,4,6] corresponds to bottom points
    bottomBB = bb[::2].copy()
    sortedInd = np.argsort(bottomBB[:,0])
    cornerSeq = [0,2,3,1]#[sortedInd[0], sortedInd[1], sortedInd[3], sortedInd[2]]
    lineEqsList = []
    ptsDirList = []
    ptsInOutMask = np.ones((points.shape[0]))
    for i in range(4):
        currInd = cornerSeq[i]
        nexInd = cornerSeq[(i+1)%4]
        if np.abs((bottomBB[nexInd,0] - bottomBB[currInd,0])) < 0.01:
            lineEqsList.append(np.array([1, 0, -bottomBB[nexInd,0]]))
        else:
            m = (bottomBB[nexInd, 1] - bottomBB[currInd, 1]) / (bottomBB[nexInd, 0] - bottomBB[currInd, 0])
            c = bottomBB[nexInd, 1] - m * bottomBB[nexInd, 0]
            lineEqsList.append(np.array([m, -1, c]))
        otherInd = [ind for ind in cornerSeq if ind not in [currInd, nexInd]]
        ptsDir = lineEqsList[-1][:2].dot(bottomBB[otherInd,:2].T) + lineEqsList[-1][2]
        # if not (np.all(ptsDir>=0) or np.all(ptsDir<=0)):
        #     a = 10
        ptsDotProd = lineEqsList[-1][:2].dot(points[:,:2].T) + lineEqsList[-1][2]
        if np.all(ptsDir>=0):
            ptsInOutMask = np.logical_and(ptsInOutMask, ptsDotProd >= 0)
        else:
            ptsInOutMask = np.logical_and(ptsInOutMask, ptsDotProd <= 0)

    ptsZinsideBB = np.logical_and(points[:,2]<=np.max(bb[:,2]), points[:,2]>=np.min(bb[:,2]))
    pointsInsideBBMask = np.logical_and(ptsInOutMask, ptsZinsideBB)

    if nyuLabelList is not None and colors is not None:
        pointsWithColMask = np.zeros((points.shape[0]))
        for nyuLabel in nyuLabelList:
            paletteLabel = nyu2palette[nyuLabel]
            tmp = np.logical_and(colors[:,0]==paletteLabel[0], colors[:,1]==paletteLabel[1])
            tmp = np.logical_and(tmp, colors[:,2]==paletteLabel[2])
            pointsWithColMask = np.logical_or(pointsWithColMask, tmp)
        pointsInsideBBMask = np.logical_and(pointsInsideBBMask, pointsWithColMask)

    pointsInsideBB = points[pointsInsideBBMask]


    return pointsInsideBB, pointsInsideBBMask

def getObj3DBB(verts):
    '''
    The order of the corners returned are:
    [Front-Left-Bottom,
     Front-Left-Top,
     Front-Right-Bottom,
     Front-Right-Top,
     Back-Left-Bottom,
     Back-Left-Top,
     Back-Right-Bottom,
     Back-Right-Top,
    ]
    :param verts:
    :return:
    '''
    assert len(verts.shape) == 2
    assert verts.shape[1] <= 4

    bb3d = np.array([[np.max(verts[:, 0]), np.min(verts[:, 1]), np.max(verts[:, 2])],
                     [np.max(verts[:, 0]), np.max(verts[:, 1]), np.max(verts[:, 2])],
                     [np.max(verts[:, 0]), np.min(verts[:, 1]), np.min(verts[:, 2])],
                     [np.max(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 2])],
                     [np.min(verts[:, 0]), np.min(verts[:, 1]), np.max(verts[:, 2])],
                     [np.min(verts[:, 0]), np.max(verts[:, 1]), np.max(verts[:, 2])],
                     [np.min(verts[:, 0]), np.min(verts[:, 1]), np.min(verts[:, 2])],
                     [np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 2])],
                     ]
                    )

    bb3d = np.array([[np.min(verts[:, 0]), np.min(verts[:, 1]), np.max(verts[:, 2])],
                     [np.min(verts[:, 0]), np.max(verts[:, 1]), np.max(verts[:, 2])],
                     [np.max(verts[:, 0]), np.min(verts[:, 1]), np.max(verts[:, 2])],
                     [np.max(verts[:, 0]), np.max(verts[:, 1]), np.max(verts[:, 2])],
                     [np.min(verts[:, 0]), np.min(verts[:, 1]), np.min(verts[:, 2])],
                     [np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 2])],
                     [np.max(verts[:, 0]), np.min(verts[:, 1]), np.min(verts[:, 2])],
                     [np.max(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 2])],
                     ]
                    )

    return bb3d

def alignPclMesh(sceneID, pclMesh):

    axis_align_matrix = getAxisAlignmentMat(sceneID)


    if isinstance(pclMesh, o3d.geometry.TriangleMesh):
        verts = np.array(pclMesh.vertices)
        newVerts = np.ones((verts.shape[0],4))
        newVerts[:,:3] = verts
        newVerts = newVerts.dot(axis_align_matrix.T)
        pclMesh.vertices = o3d.utility.Vector3dVector(newVerts[:,:3])
    elif isinstance(pclMesh, o3d.geometry.PointCloud):
        points = np.array(pclMesh.points)
        newPoints = np.ones((points.shape[0], 4))
        newPoints[:, :3] = points
        newPoints = newPoints.dot(axis_align_matrix.T)
        pclMesh.points = o3d.utility.Vector3dVector(newPoints[:, :3])
    else:
        raise NotImplementedError

    return pclMesh


def getAxisAlignmentMat(sceneID):
    axis_align_matrix = np.eye(4)
    metaFile = os.path.join(SCANS_DIR, sceneID,
                            sceneID + '.txt')  # includes axisAlignment info for the train set scans.
    if os.path.exists(metaFile):#int(sceneID[5:9]) < 707:

        assert os.path.exists(metaFile), '%s' % metaFile
        with open(metaFile) as f:
            lines = f.readlines()
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) \
                                     for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

    if os.path.exists(os.path.join(SCANS_DIR, sceneID,
                                   sceneID + '_add.npy')):
        addTrans = np.load(os.path.join(SCANS_DIR, sceneID,
                                        sceneID + '_add.npy'))
        axis_align_matrix = addTrans.dot(axis_align_matrix)

    return axis_align_matrix

def label_colormap(N=256):

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap

def drawOpen3dCylLines(bbListIn,col=None):
    # draw the BBs
    lines = [[0, 1], [0, 2], [1, 3], [2, 3],
             [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7],

             [1, 0], [2, 0], [3, 1], [3, 2],
             [5, 4], [6, 4], [7, 5], [7, 6],
             [4, 0], [5, 1], [6, 2], [7, 3]
             ]
    line_sets = []

    for bb in bbListIn:
        points = bb

        # if bb.instanceID != 14:
        #     continue
        # if bb.classPred not in [4,6]:
        #     continue


        if col is None:
            col = [0,0,1]
        colors = [col for i in range(len(lines))]
        if False:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            line_sets.append(line_set)
        else:

            line_mesh1 = LineMesh(points, lines, colors, radius=0.03)
            line_mesh1_geoms = line_mesh1.cylinder_segments
            line_sets = line_mesh1_geoms[0]
            for l in line_mesh1_geoms[1:]:
                line_sets = line_sets + l
            # line_sets.append(line_mesh1_geoms)

    return line_sets

def getOrientedBBIntersection(bb1, bb2, isAdjustBottomZ=False):
    '''

    :param bb1: gt bb
    :param bb2: pred bb
    :return:
    '''
    assert isinstance(bb1, np.ndarray)
    assert isinstance(bb2, np.ndarray)
    assert len(bb1.shape) == 2
    assert len(bb2.shape) == 2
    assert bb1.shape[1] == 3
    assert bb2.shape[1] == 3
    assert bb1.shape[0] == bb1.shape[0]

    if isAdjustBottomZ:
        bb2[::2,2] = bb1[::2,2].copy()

    bbMax = np.max(bb1, axis=0)
    bbRefMax = np.max(bb2, axis=0)
    min_max = np.array([bbMax, bbRefMax]).min(0)

    max_max = np.array([bbMax, bbRefMax]).max(0)

    bbMin = np.min(bb1, axis=0)
    bbRefMin = np.min(bb2, axis=0)
    max_min = np.array([bbMin, bbRefMin]).max(0)

    min_min = np.array([bbMin, bbRefMin]).min(0)

    if not ((min_max > max_min).all()):
        return 0, 0 ,0

    xOverlap = min_max[0] - max_min[0]
    yOverlap = min_max[1] - max_min[1]
    zOverlap = min_max[2] - max_min[2]

    p = Polygon([(bb1[0, 0], bb1[0, 1]), (bb1[2, 0], bb1[2, 1]), (bb1[6, 0], bb1[6, 1]), (bb1[4, 0], bb1[4, 1])])
    q = Polygon([(bb2[0, 0], bb2[0, 1]), (bb2[2, 0], bb2[2, 1]), (bb2[6, 0], bb2[6, 1]), (bb2[4, 0], bb2[4, 1])])

    intersectionArea = p.intersection(q).area * zOverlap

    bb1Area = p.area*(bb1[1,2]-bb1[0,2])
    bb2Area = q.area*(bb2[1,2]-bb2[0,2])

    unionArea = bb1Area + bb2Area - intersectionArea

    iou3d = intersectionArea / unionArea

    return iou3d, intersectionArea, unionArea

# helper function to calculate difference between two quaternions
def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation

def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M

def paletteNyu():
    palette = create_color_palette()
    nyu2palette = {}
    palette2nyu = {}
    for i in range(len(palette)):
        nyu2palette[i] = palette[i]
        palette2nyu[palette[i]] = (i,i,i)

    return nyu2palette

# color palette for nyu40 labels
def create_color_palette():
    return [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),
       (247, 182, 210),		# desk
       (66, 188, 102),
       (219, 219, 141),		# curtain
       (140, 57, 197),
       (202, 185, 52),
       (51, 176, 203),
       (200, 54, 131),
       (92, 193, 61),
       (78, 71, 183),
       (172, 114, 82),
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),
       (153, 98, 156),
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),
       (227, 119, 194),		# bathtub
       (213, 92, 176),
       (94, 106, 211),
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]

def getStandardShapenetCatID(catID, modelID, SHAPENETCOREV2_DIR):
    if len(catID.split('_')) > 1:
        catIDsOrig = catID.split('_')
        if 'toilet' in catIDsOrig:
            catIDsOrig.append('03001627')
        found = False
        for kk in range(len(catIDsOrig)):
            if os.path.exists(join(SHAPENETCOREV2_DIR, catIDsOrig[kk], modelID, 'models',
                                   'model_normalized.obj')):
                found = True
                break

        if not found:
            print(catIDsOrig, modelID)
            assert False
        catID = catIDsOrig[kk]

    return catID