import open3d as o3d

class LayoutComp:
    def __init__(self, comp_dict):
        """
        :param comp_dict: Component dictionary {"type": type, "plane": plane, "poly": poly}
        """

        self.type = comp_dict["type"]
        self.plane = comp_dict["plane"]
        self.poly = comp_dict["poly"]


class LayoutStruct:
    def __init__(self, comp_list, axis_align_matrix):
        self.comp_list = comp_list
        self.axis_align_matrix = axis_align_matrix
