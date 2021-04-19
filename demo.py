from monte_carlo_model_search.mc_utilsObjects import *
from utils import downloadScanNetScene
from copy import deepcopy
import random

from absl import flags
from absl import app
FLAGS = flags.FLAGS

flags.DEFINE_string('scene', 'scene0000_00', 'scene ID')
flags.DEFINE_string('shapenet_dir', '/media/shreyas/4aa82be1-14a8-47f7-93a7-171e3ebac2b0/Datasets/ShapeNetCore.v2', 'shapenet dir')

def visualize(sceneID):

    mctsObjs = MCTSObjs(sceneID, FLAGS.shapenet_dir)

    # Create mesh with Bounding Boxes
    lineGeometryList = []
    geometryList = []
    for obj in mctsObjs.mctsObjList[0:]:
        geometryList.append(obj.transMesh)
        col = [0, 0, 1]
        lineSets = drawOpen3dCylLines([obj.orientedBB], col)
        lineGeometryList.append(lineSets)

    finalMesh = deepcopy(mctsObjs.sceneMesh)
    for i in range(0, len(geometryList)):
        finalMesh += lineGeometryList[i]
        finalMesh += mctsObjs.mctsObjList[i].transMesh

    objMesh = deepcopy(geometryList[0])
    for i in range(1, len(geometryList)):
        objMesh += geometryList[i]

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=640, height=480, left=0, top=0,
                      visible=True)
    vis.get_render_option().mesh_show_back_face = True

    sceneMeshVert = np.array(mctsObjs.sceneMesh.vertices)
    trans = np.zeros((3,)) * 1.0
    trans[0] = np.max(sceneMeshVert[:, 0]) - np.min(sceneMeshVert[:, 0])
    vis.add_geometry(mctsObjs.sceneMesh)
    vis.add_geometry(objMesh.translate(trans * 1.1))
    vis.add_geometry(finalMesh.translate(trans * 1.1 * 2))

    vis.run()


def main(argv):
    if FLAGS.scene == 'scene0000_00':

        with open('scannetv2_val.txt', 'r') as f:
            sceneIDs = f.readlines()
        sceneIDs = [s.strip() for s in sceneIDs if ('_00' in s)]
        sceneIDs = [s for s in sceneIDs if os.path.exists(os.path.join(SCANS_DIR, s, 'monte_carlo', 'FinalCandidates.pickle'))]
    else:
        sceneIDs = [FLAGS.scene]

    while (True):
        sceneID = random.choice(sceneIDs)
        downloadScanNetScene(sceneID)
        visualize(sceneID)




if __name__ == '__main__':
    app.run(main)
