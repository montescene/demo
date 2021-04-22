import multiprocessing as mp
import os
import glob
from shutil import copyfile
import pickle
import json
import numpy as np
import open3d as o3d
import sys
from absl import flags
from absl import app

from LayoutStruct.LayoutStruct import LayoutStruct
from LayoutStruct.utils import *

FLAGS = flags.FLAGS

flags.DEFINE_string('annotations_path', '/media/sinisa/Sinisa_hdd_data/Sinisa_Projects/corridor_localisation/Datasets/cvpr_2021_layout_annotations/final_annotations/', 'room layout annotations path')
flags.DEFINE_string('solutions_path', '/home/sinisa/Sinisa_Projects/indoor_localisation/montescene/demo/outputs/scans/', 'path to retrieved layouts')

x=10000
sys.setrecursionlimit(x)

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_metrics(pred, gt, pred_polys, gt_polys, thresh=0.4):

    verts_match_pred2gt_list = [[]] * pred.shape[0]
    verts_match_dist_gt = np.array([np.inf] * gt.shape[0])
    verts_match_dist_pred = np.array([np.inf] * pred.shape[0])

    verts_found = np.array([False] * gt.shape[0])
    verts_match_gt2pred = np.array([-1] * gt.shape[0])
    verts_match_pred2gt = np.array([-1] * pred.shape[0])

    # V1
    for verts_ind, verts in enumerate(gt):
        for pred_ind, pred_verts in enumerate(pred):
            if verts_match_pred2gt[pred_ind] != -1:
                continue
            dist = (np.linalg.norm(np.abs(verts - pred_verts), ord=2))
            if dist < thresh:
                if dist < verts_match_dist_gt[verts_ind]:
                    if not verts_found[verts_ind]:
                        verts_found[verts_ind] = True
                        verts_match_dist_gt[verts_ind] = dist
                        verts_match_gt2pred[verts_ind] = pred_ind
                        verts_match_pred2gt[pred_ind] = verts_ind
                        verts_match_pred2gt_list[pred_ind].append(verts_ind)
                    else:
                        verts_match_pred2gt[verts_match_gt2pred[verts_ind]] = -1

                        verts_found[verts_ind] = True
                        verts_match_dist_gt[verts_ind] = dist
                        verts_match_gt2pred[verts_ind] = pred_ind
                        verts_match_pred2gt[pred_ind] = verts_ind
                        verts_match_pred2gt_list[pred_ind].append(verts_ind)

    # Corner recall
    corner_recall = np.sum(verts_found) / float(verts_found.shape[0])

    # Corner precision
    corner_precision = sum(verts_found) / float(np.sum(verts_found) + np.sum(verts_match_pred2gt == -1))


    pred_polys_walls = []
    for pred_poly in pred_polys:
        pred_vts_coord = pred[pred_poly]
        is_floor_ceil = np.all(np.abs(pred_vts_coord[:,2] - pred_vts_coord[0,2]) < 0.02)
        if is_floor_ceil:
            continue
        pred_polys_walls.append(pred_poly)
    pred_polys_walls = np.array(pred_polys_walls)

    gt_polys_walls = []
    for gt_poly in gt_polys:
        gt_vts_coord = gt[gt_poly]
        is_floor_ceil = np.all(np.abs(gt_vts_coord[:,2] - gt_vts_coord[0,2]) < 0.02)
        if is_floor_ceil:
            continue
        gt_polys_walls.append(gt_poly)
    gt_polys_walls = np.array(gt_polys_walls)

    gt_c_polys = []
    for gt_ind, gt_poly in enumerate(gt_polys_walls):
        gt_c = gt[list(gt_poly) + [gt_poly[0]]]
        gt_c_polys.append(gt_c)


    def sort_by_area(x):
        x_homo = np.ones((x.shape[0], 1))
        x_homo = np.concatenate((x, x_homo), axis=1)

        x_plane = fit_plane_LSE(x_homo)

        x_2d = project_poly2plane_2d(x, x_plane)
        x_2d_shapely = poly_2d_np2shapely(x_2d)

        return x_2d_shapely.area
    gt_c_polys.sort(reverse=True, key=sort_by_area)


    poly_ious = [0] * gt_polys_walls.shape[0]
    gt_poly_correct = [True] * gt_polys_walls.shape[0]
    poly_ious_pred2gt = [-1] * pred_polys_walls.shape[0]
    poly_ious_gt2pred = [-1] * gt_polys_walls.shape[0]
    for gt_ind, gt_c in enumerate(gt_c_polys):

        gt_c_homo = np.ones((gt_c.shape[0], 1))
        gt_c_homo = np.concatenate((gt_c, gt_c_homo), axis=1)

        gt_plane = fit_plane_LSE(gt_c_homo)
        gt_c_2d = project_poly2plane_2d(gt_c, gt_plane)
        gt_c_shapely = poly_2d_np2shapely(gt_c_2d)

        if not gt_c_shapely.is_simple:
            # Incorrect polygon skip
            gt_poly_correct[gt_ind] = False
            continue

        for pred_ind, pred_poly in enumerate(pred_polys_walls):
            if poly_ious_pred2gt[pred_ind] != -1:
                continue
            pred_c = pred[list(pred_poly) + [pred_poly[0]]]

            pred_c_2d = project_poly2plane_2d(pred_c, gt_plane)

            _, dists = project_points_to_plane_xy(pred_c, gt_plane, norm=False)
            if np.mean(np.abs(dists)) > 0.1:
                continue

            pred_c_shapely = poly_2d_np2shapely(pred_c_2d)


            if not pred_c_shapely.is_simple:
                continue


            intersection = pred_c_shapely.intersection(gt_c_shapely).area
            union = pred_c_shapely.area + gt_c_shapely.area - intersection

            iou = intersection / union
            if iou > poly_ious[gt_ind]:
                if poly_ious_gt2pred[gt_ind] != -1:
                    poly_ious_pred2gt[poly_ious_gt2pred[gt_ind]] = -1

                poly_ious[gt_ind] = iou
                poly_ious_gt2pred[gt_ind] = pred_ind
                poly_ious_pred2gt[pred_ind] = gt_ind

    poly_ious = 2 * sum(poly_ious) / float(sum(gt_poly_correct) + pred_polys_walls.shape[0])

    return corner_recall, corner_precision, poly_ious

def get_poly2vert(poly_coords, vert_coords):
    annotations_polys = []
    for poly_c in poly_coords:
        poly = [-1] * poly_c.shape[0]

        for p_ind, p_c in enumerate(poly_c):
            for v_ind, v_c in enumerate(vert_coords):
                if np.all(p_c == v_c):
                    poly[p_ind] = v_ind
                    break
        annotations_polys.append(poly)

    annotations_polys = np.array(annotations_polys)
    return annotations_polys

def main(argv):
    f = open(
        "./scannetv2_val.txt",
        "r")
    lines = f.readlines()
    lines.sort()

    corners_recall_per_scene_list_mcss2our = []
    corners_recall_per_scene_dict_mcss2our = {}
    corners_prec_per_scene_list_mcss2our = []
    corners_prec_per_scene_dict_mcss2our = {}
    poly_iou_per_scene_list_mcss2our = []
    poly_iou_per_scene_dict_mcss2our = {}

    for line in lines:
        scene_name = line[:-1]
        decimals = 2


        # Invalid scenes
        if scene_name[-3:] != "_00":
            continue
        if scene_name in ["scene0077_00", "scene0164_00", "scene0550_00"]:
            continue

        print("---" + scene_name + "---")


        # Load Annotations
        # --------------------------------------------------------------------------------------------------------------

        annotations_poly_path = os.path.join(FLAGS.annotations_path, "json_dir/", "scene_" + scene_name[5:-3] + ".json")
        if not os.path.isfile(annotations_poly_path):
            continue

        # Parse Ours
        # --------------------------------------------------------------------------------------------------------------
        solution_folder = os.path.join(FLAGS.solutions_path, scene_name, "monte_carlo")
        solution_file = os.path.join(solution_folder, "FinalLayout.pickle")

        with open(solution_file,
                  'rb') as f:
            layout_struct = pickle.load(f)  # type: LayoutStruct

        finalCandidateList = layout_struct.comp_list
        axis_align_matrix = layout_struct.axis_align_matrix

        mcts_poly_coord_list = []

        # Parse Annotations
        # --------------------------------------------------------------------------------------------------------------

        for cand in finalCandidateList:
            poly = cand.poly
            poly = np.array(poly)
            poly[:, 2] = (poly[:, 2] > np.mean(poly[:, 2])) * 2.5

            poly_round = np.round(poly * 10, decimals=decimals) / 10
            poly_round = np.round(poly_round, decimals=decimals)
            # poly_round = np.round(poly, decimals=decimals)

            mcts_poly_coord_list.append(poly_round)

        mcts_poly_coord_np = np.array(mcts_poly_coord_list)
        mcts_poly_verts = mcts_poly_coord_np.reshape((-1, 3))

        mcts_poly_verts = np.unique(mcts_poly_verts, axis=0)
        mcts_polys = get_poly2vert(mcts_poly_coord_np, mcts_poly_verts)

        with open(annotations_poly_path) as f:
            annotations_data = json.load(f)

        annotations_verts = []
        annotations_polys_coords = []
        for key in annotations_data.keys():
            poly = annotations_data[key]
            # poly = np.array(poly)
            poly = np.round(np.array(poly), decimals=decimals)

            annotations_verts += list(poly[:-1])
            annotations_polys_coords.append(poly)

            if poly.shape[0] != 5:
                print(poly.shape)
                # assert False
                # continue

        annotations_verts = np.array(annotations_verts)
        annotations_verts = np.unique(annotations_verts, axis=0)

        annotations_polys_coords = np.array(annotations_polys_coords)

        annotations_polys = get_poly2vert(annotations_polys_coords, annotations_verts)

        ones = np.ones((annotations_verts.shape[0], 1))
        annotations_verts = np.concatenate([annotations_verts, ones], axis=1)
        annotations_verts = np.dot(axis_align_matrix, annotations_verts.T)
        annotations_verts = annotations_verts[:3, :] / annotations_verts[3, :][None, :]
        annotations_verts = annotations_verts.T
        annotations_verts = np.round(annotations_verts, decimals=decimals)
        annotations_verts[:, 2] = (annotations_verts[:, 2] > np.mean(annotations_verts[:, 2])) * 2.5

        # MCSS2OURS
        # --------------------------------------------------------------------------------------------------------------

        corner_recall_mcss2our, corner_prec_mcss2our, poly_iou_mcss2our = \
            get_metrics(mcts_poly_verts, annotations_verts, mcts_polys[:, :-1], annotations_polys[:, :-1],
                        thresh=0.4)

        corners_recall_per_scene_list_mcss2our.append(corner_recall_mcss2our)
        corners_recall_per_scene_dict_mcss2our[scene_name] = corner_recall_mcss2our
        corners_prec_per_scene_list_mcss2our.append(corner_prec_mcss2our)
        corners_prec_per_scene_dict_mcss2our[scene_name] = corner_prec_mcss2our

        poly_iou_per_scene_list_mcss2our.append(poly_iou_mcss2our)
        poly_iou_per_scene_dict_mcss2our[scene_name] = poly_iou_mcss2our

        print("--MCSS2OUR--")
        print("corner recall : ", corner_recall_mcss2our)
        print("corner prec : ", corner_prec_mcss2our)
        print("poly iou : ", poly_iou_mcss2our)

    print("---Results---")
    for line in lines:
        scene_name = line[:-1]

        if scene_name not in corners_prec_per_scene_list_mcss2our:
            continue

        print("---" + scene_name + "---")

        print("---MCSS2OUR---")
        print("Scene %s verts recall  %0.3f" % (scene_name, corners_recall_per_scene_dict_mcss2our[scene_name]))
        print("Scene %s verts precision  %0.3f" % (scene_name, corners_prec_per_scene_dict_mcss2our[scene_name]))
        print("Scene %s poly iou  %0.3f" % (scene_name, poly_iou_per_scene_list_mcss2our[scene_name]))

    print("---MCSS2OUR---")

    avg_recall = sum(corners_recall_per_scene_list_mcss2our) / float(len(corners_recall_per_scene_list_mcss2our))
    avg_prec = sum(corners_prec_per_scene_list_mcss2our) / float(len(corners_prec_per_scene_list_mcss2our))
    print("Avg. corner recall  %0.3f" % (avg_recall))
    print("Avg. corner prec  %0.3f" % (avg_prec))
    avg_iou = sum(poly_iou_per_scene_list_mcss2our) / float(len(poly_iou_per_scene_list_mcss2our))
    print("Avg. poly iou  %0.3f" % (avg_iou))


if __name__ == '__main__':
    app.run(main)