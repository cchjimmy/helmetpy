import trimesh
import numpy as np
import math
import typing
# import scipy
import pymeshfix


def clean_mesh(input_path: str, output_path: str):
    pymeshfix.clean_from_file(input_path, output_path)

    mesh = trimesh.load_mesh(output_path)
    # mesh.remove_degenerate_faces()
    # mesh.remove_infinite_values()
    # mesh.remove_unreferenced_vertices()

    # mesh = trimesh.smoothing.filter_humphrey(mesh)
    mesh = trimesh.smoothing.filter_laplacian(mesh)
    # mesh = trimesh.smoothing.filter_mut_dif_laplacian(mesh)
    # mesh = trimesh.smoothing.filter_taubin(mesh)

    # mesh.update_faces(np.sum((mesh.triangles_center-mesh.centroid) *
    #                   mesh.face_normals, axis=1) > 0.95)
    # mesh = mesh.convex_hull
    mesh.export(output_path)


def plot_vertex_colors(input_path: str, colors: np.ndarray):
    mesh = trimesh.load_mesh(input_path)
    mesh.visual.vertex_colors = colors
    mesh.show()


def plot_mesh_vertex_defects(input_path: str):
    mesh = trimesh.load_mesh(input_path)
    defects = trimesh.curvature.vertex_defects(mesh)
    colors = []
    for defect in defects:
        colors.append([defect]*3)
    plot_vertex_colors(input_path, colors)


def plot_mesh_vertex_grads(input_path: str):
    mesh = trimesh.load_mesh(input_path)
    plot_vertex_colors(input_path, mesh.vertex_normals.copy())


def crop_mesh(
    input_path: str,
    output_path: str,
    # relative to head scan height (z axis, + -> -)
    helmet_height: float = 0.7,
    ear_left: float = 0.5,  # relative to head scan depth (y axis, + -> -)
    ear_radius: float = 0.1,  # relative to head scan depth (y axis, + -> -)
    ear_height: float = 0.3,  # relative to helmet height (y axis, - -> +)
    # relative to head scan depth (y axis, + -> -)
    sideburn_width: float = 0.1,
):
    mesh = trimesh.load_mesh(input_path)
    bb = mesh.bounding_box
    bb_size = bb.bounds[1]-bb.bounds[0]
    origin = bb.centroid
    box = trimesh.primitives.Box(extents=bb_size)
    box_pos = np.array([0, 0, -helmet_height*(1-ear_height)])
    box.apply_translation(origin+bb_size*box_pos)

    radius = bb_size[1]*ear_radius
    ear_hole = trimesh.primitives.Cylinder(radius, bb_size[0]*1.1)
    ear_hole.apply_transform(
        trimesh.transformations.rotation_matrix(0.5*math.pi, [0, 1, 0]))
    ear_hole_pos = np.array([0, -ear_left, 0])
    ear_hole.apply_translation(
        origin+bb_size*(0.5*np.array([0, 1, 1])+ear_hole_pos+box_pos))
    mesh = mesh.difference(ear_hole)

    sideburn = trimesh.primitives.Box(
        extents=bb_size*[1.1, sideburn_width, helmet_height*ear_height*2])
    sideburn_pos = np.array([0, -(ear_radius+sideburn_width*0.5), 0])
    sideburn.apply_translation(
        origin + bb_size*(0.5*np.array([0, 1, 1])+sideburn_pos+box_pos+ear_hole_pos))
    box = box.difference(sideburn)

    back = trimesh.primitives.Box(
        extents=bb_size*[1.1, (ear_left-ear_radius)*1.1, helmet_height*ear_height*2])
    back_pos = np.array([0, (ear_left-ear_radius)*1.1*0.5+ear_radius, 0])
    back.apply_translation(
        origin+bb_size*(0.5*np.array([0, 1, 1])+back_pos+box_pos+ear_hole_pos))
    box = box.difference(back)

    mesh = mesh.difference(box)

    mesh = trimesh.util.concatenate(mesh, [ear_hole, back, sideburn, box, bb])

    mesh.export(output_path)


def arrange_mesh(input_path: str, output_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(input_path)

    origin = mesh.bounding_box.centroid

    # cut into 4 parts
    left = mesh.copy().slice_plane(
        plane_normal=[1, 0, 0], plane_origin=origin)

    right = mesh.copy().slice_plane(
        plane_normal=[-1, 0, 0], plane_origin=origin)

    bl = left.copy().slice_plane(
        plane_normal=[0, 1, 0], plane_origin=origin)
    ul = left.copy().slice_plane(
        plane_normal=[0, -1, 0], plane_origin=origin)
    br = right.copy().slice_plane(
        plane_normal=[0, 1, 0], plane_origin=origin)
    ur = right.copy().slice_plane(
        plane_normal=[0, -1, 0], plane_origin=origin)

    mirror = np.eye(4)
    mirror[0, 0] = -1

    parts: trimesh.Trimesh[4] = []

    bottom = bl if bl.area > br.area else br
    parts.append(bottom)
    parts.append(bottom.copy().apply_transform(mirror))

    distance = abs(parts[0].bounding_box.centroid -
                   parts[1].bounding_box.centroid)

    diff = ((parts[0].bounds[1]-parts[0].bounds[0]) -
            distance)*distance/np.linalg.norm(distance)*0.5

    parts[0].apply_translation(diff)
    parts[1].apply_translation(-diff)

    up = ul if ul.area > ur.area else ur
    parts.append(up)
    parts.append(up.copy().apply_transform(mirror))

    parts[2].apply_translation(diff)
    parts[3].apply_translation(-diff)

    helmet = trimesh.util.concatenate(parts)
    print("before merge:", len(helmet.vertices))
    helmet.merge_vertices()
    print("after merge:", len(helmet.vertices))
    helmet.fix_normals()

    helmet.export(output_path)


def apply_thickness(input_path: str, output_path: str, thickness: float):
    if thickness <= 0:
        raise ValueError("thickness must be > 0.")

    mesh = trimesh.load_mesh(input_path)

    inner = mesh.copy()
    inner.invert()
    outer = mesh.copy()

    outer.vertices += outer.vertex_normals * thickness

    vertices = np.concatenate((inner.vertices, outer.vertices))
    faces = []

    inner_boundaries = identify_boundaries(inner)
    offset = len(inner.vertices)
    for i in range(len(inner_boundaries.entities)):
        inner_points = inner_boundaries.entities[i].points
        points_len = len(inner_points)
        for j in range(points_len):
            quad = make_quad(
                inner_points[j] + offset,
                inner_points[j],
                inner_points[(j+1) % points_len]+offset,
                inner_points[(j+1) % points_len])
            faces.append(quad[0])
            faces.append(quad[1])

    faces = np.concatenate((inner.faces, outer.faces + offset, faces))

    trimesh.Trimesh(vertices=vertices, faces=faces).export(output_path)


def make_quad(tl, tr, bl, br) -> np.ndarray:
    return np.array([[tl, tr, bl], [bl, tr, br]])


def identify_boundaries(mesh: trimesh.Trimesh) -> trimesh.path.Path3D:
    path3d_param = trimesh.path.exchange.misc.faces_to_path(mesh)
    return trimesh.path.Path3D(**path3d_param)


def show_vert_norms(input_path: str, scale: float = 1):
    mesh = trimesh.load_mesh(input_path)
    lines = [[mesh.vertices[i], mesh.vertices[i]+mesh.vertex_normals[i]*scale]
             for i in range(len(mesh.vertices))]
    trimesh.load_path(lines).show()


def align_mesh(input_path: str, output_path: str, landmarks: typing.List[float]):
    landmarks = np.array(landmarks)

    if (landmarks.shape != (3, 3)):
        raise ValueError("Landmarks must have exactly three 3D points.")

    # credit: from mesh to meaning, ch.4.7, p.104
    p1 = landmarks[0, :]  # right tragus
    p2 = landmarks[1, :]  # nasion
    p3 = landmarks[2, :]  # left tragus

    centroid = 1/3*(p1+p2+p3)

    normal = np.cross(p3-p2, p1-p2)
    normal /= np.linalg.norm(normal)
    forward = -centroid+p2
    forward /= np.linalg.norm(forward)

    target_normal = np.array([0, 0, 1])
    target_forward = np.array([0, -1, 0])

    r1 = trimesh.transformations.rotation_matrix(
        math.acos(np.dot(normal, target_normal)), np.cross(normal, target_normal))
    r2 = trimesh.transformations.rotation_matrix(
        math.acos(np.dot(forward, target_forward)), np.cross(forward, target_forward))

    mesh = trimesh.load_mesh(input_path)

    mesh.apply_transform(r1@r2).export(output_path)


def create_slices(
    input_path: str,
    plane_normal: typing.List[float],
    plane_origin: typing.List[float],
    slice_count: int
) -> typing.List[trimesh.path.Path3D]:
    plane_origin = np.array(plane_origin)

    # normalize normal
    plane_normal = np.asanyarray(plane_normal, dtype=float)
    mag = np.linalg.norm(plane_normal)
    plane_normal /= mag

    mesh = trimesh.load_mesh(input_path)

    max_point = None
    max_num = -math.inf

    for vertex in mesh.vertices:
        dot = np.dot(plane_normal, vertex)
        if dot < max_num:
            continue
        max_num = dot
        max_point = vertex

    distance = np.linalg.norm(max_point-plane_origin)
    unit_distance = distance/slice_count

    sections = mesh.section_multiplane(
        plane_normal=plane_normal, plane_origin=plane_origin, heights=np.arange(stop=distance, step=unit_distance))

    for i in range(len(sections)):
        if sections[i] is None:
            continue
        sections[i] = sections[i].to_3D()

    return sections


def sample_path_3d(
    path: trimesh.path.Path3D,
    samples: int = 100
) -> typing.List[np.array]:
    vertex_sequence = path.discrete[0]

    path_len = 0
    for i in range(len(vertex_sequence)-1):
        path_len += np.linalg.norm(
            vertex_sequence[i+1] -
            vertex_sequence[i])

    interval = path_len / samples
    sum_len = 0
    verts = []

    for i in range(len(vertex_sequence)-1):
        p0 = vertex_sequence[i]
        p1 = vertex_sequence[i+1]
        elm_len = np.linalg.norm(p1-p0)
        sum_len += elm_len
        target_len = (len(verts)+1)*interval
        if sum_len < target_len:
            continue
        diff = sum_len - target_len
        percent = diff/elm_len
        verts.append(p0+(p1-p0)/elm_len*(1-percent))

    return verts


def centroid(input_path: str) -> np.ndarray:
    return trimesh.load_mesh(input_path).centroid


def to_circular_path_3D(verts: typing.List[np.array]) -> trimesh.path.Path3D:
    path_3d = trimesh.path.Path3D()
    path_3d.vertices = verts
    path_3d.entities = [trimesh.path.entities.Line(
        points=list(range(len(verts)))+[0])]
    return path_3d


def find_cephalic_index(path: trimesh.path.Path3D) -> float:
    planar, to_3D = path.to_2D()
    size = planar.bounds[1]-planar.bounds[0]
    return size[0]*100/size[1]


def generate_helmet(input_path: str, output_path: str):
    clean_mesh(input_path, output_path)
    slices = create_slices(output_path, plane_normal=[
        0, 0, 1], plane_origin=centroid(output_path), slice_count=10)
    sampled = [sample_path_3d(slices[i], samples=10)
               for i in range(len(slices))]

    print(find_cephalic_index(to_circular_path_3D(sampled[0])))
    print(find_cephalic_index(slices[0]))

    mesh = trimesh.PointCloud(vertices=np.vstack(tuple(sampled)))

    mesh = mesh.convex_hull

    mesh.update_faces([np.linalg.norm(
        np.dot(mesh.face_normals[i], [0, 0, 1])
    ) < 0.95
        for i in range(len(mesh.face_normals))
    ])

    mesh.export(output_path)

    apply_thickness(output_path, output_path, thickness=4)
