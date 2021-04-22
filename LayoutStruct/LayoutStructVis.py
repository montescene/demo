import open3d as o3d
import numpy as np

from LayoutStruct.LayoutStruct import LayoutStruct
from LayoutStruct.line_mesh import LineMesh

class LayoutStructVis:
    def __init__(self):
        pass

    def get_layout_wireframe(self, comp_list, clr=np.array([1., 0, 0])):
        # Add edges
        # ------------------------------------

        wireframe = o3d.geometry.TriangleMesh()
        for comp_ind, comp in enumerate(comp_list):

            comp_poly = comp.poly

            clr = clr
            comp_corners = []
            comp_edges = []

            comp_poly = np.round(comp_poly, decimals=3)
            for corner_ind, comp_corner in enumerate(comp_poly):
                if corner_ind < len(comp_poly) - 2:
                    comp_corners.append(comp_corner)
                    comp_edges.append([corner_ind, corner_ind + 1])
                # elif corner_ind == len(comp_poly) - 2:
                # comp_corners.append(comp_corner)
                # comp_edges.append([corner_ind, 0])
                elif corner_ind == len(comp_poly) - 2:
                    comp_corners.append(comp_corner)
                    comp_edges.append([0, corner_ind])
            # comp_edges.append([0,2])

            comp_corners = np.array(comp_corners)
            comp_edges = np.array(comp_edges)

            compidate_edges_o3d = o3d.geometry.LineSet()
            compidate_edges_o3d.points = o3d.utility.Vector3dVector(comp_corners)
            compidate_edges_o3d.lines = o3d.utility.Vector2iVector(comp_edges)
            compidate_edges_o3d.colors = \
                o3d.utility.Vector3dVector(np.ones_like(comp_corners) * clr[None, :])

            # print("comp_corners", comp_corners)
            # print("comp_edges", comp_edges)
            # print("clr", list(np.ones_like(comp_corners) * clr[None, :]))
            # line_mesh1 = LineMesh(comp_corners, list(comp_edges), list(np.ones_like(comp_corners) * clr[None, :]),
            #                       radius=0.02)
            line_mesh1 = LineMesh(comp_corners, list(comp_edges), clr[:],
                                  radius=0.02)
            line_mesh1_geoms = line_mesh1.cylinder_segments

            # line_sets.append(line_mesh1_geoms)

            for g in line_mesh1_geoms:
                wireframe += g

            # break

        return wireframe

    def visualize_layout(self, layout):
        """

        :param layout:
        :type layout: LayoutStruct
        :return:
        """

        wireframe = self.get_layout_wireframe(layout.comp_list)

        o3d.visualization.draw_geometries([wireframe])