from monte_carlo_model_search.mc_utilsObjects import *
from utils import downloadScanNetScene
from copy import deepcopy
import random

from absl import flags
from absl import app

from LayoutStruct.LayoutStruct import LayoutStruct
from LayoutStruct.LayoutStructVis import LayoutStructVis

FLAGS = flags.FLAGS

flags.DEFINE_string('scene', 'scene0000_00', 'scene ID')
flags.DEFINE_string('shapenet_dir', '/media/shreyas/4aa82be1-14a8-47f7-93a7-171e3ebac2b0/Datasets/ShapeNetCore.v2', 'shapenet dir')
flags.DEFINE_bool('add_objects', False, 'add objects to the scene')
flags.DEFINE_bool('add_layout', False, 'add room layout to the scene')

def visualize(sceneID, add_objects=True, add_layout=True):
    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=640, height=480, left=0, top=0,
                      visible=True)
    vis.get_render_option().mesh_show_back_face = True

    if add_objects:
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



        sceneMeshVert = np.array(mctsObjs.sceneMesh.vertices)
        trans = np.zeros((3,)) * 1.0
        trans[0] = np.max(sceneMeshVert[:, 0]) - np.min(sceneMeshVert[:, 0])
        vis.add_geometry(mctsObjs.sceneMesh)
        vis.add_geometry(objMesh.translate(trans * 1.1))
        vis.add_geometry(finalMesh.translate(trans * 1.1 * 2))

    if add_layout:
        sceneLayoutPath = os.path.join("outputs/scans/", sceneID, 'monte_carlo/FinalLayout.pickle')
        if not os.path.isfile(sceneLayoutPath):
            print("WARNING: Layout missing for scene %s" % sceneID)
        else:
            with open(sceneLayoutPath, 'rb') as f:
                layout_struct = pickle.load(f) # type: LayoutStruct

            wireframe = LayoutStructVis().get_layout_wireframe(layout_struct.comp_list)

            vis.add_geometry(wireframe)

    vis.run()


def main(argv):
    add_objects = FLAGS.add_objects
    add_layout = FLAGS.add_layout

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
        visualize(sceneID, add_objects=add_objects, add_layout=add_layout)




if __name__ == '__main__':
    app.run(main)
