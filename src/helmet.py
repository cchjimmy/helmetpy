import trimesh
import numpy as np
import math
import typing
import pymeshfix


def clean_mesh(input_path: str, output_path: str):
    pymeshfix.clean_from_file(input_path, output_path)
    mesh = trimesh.load_mesh(output_path)
    mesh = trimesh.smoothing.filter_laplacian(mesh)
    mesh.export(output_path)


def thicken(input_path: str, output_path: str, thickness: float):
    if thickness <= 0:
        raise ValueError("thickness must be > 0.")

    mesh = trimesh.load_mesh(input_path)

    inner_faces = mesh.faces.copy()
    inner_faces[:, [0, 1]] = inner_faces[:, [1, 0]]  # inverts winding
    outer_verts = mesh.vertices.copy()
    outer_verts += mesh.vertex_normals * thickness

    vertices = np.concatenate((mesh.vertices, outer_verts))
    faces = []

    boundaries = identify_boundaries(mesh)
    offset = len(mesh.vertices)
    for i in range(len(boundaries.entities)):
        points = boundaries.entities[i].points
        faces.extend(mesh_strip(points, points+offset))

    faces = np.concatenate((inner_faces, mesh.faces + offset, faces))

    trimesh.Trimesh(vertices=vertices, faces=faces).export(output_path)


def make_quad(tl, tr, bl, br) -> (typing.List, typing.List):
    return [tl, tr, bl], [bl, tr, br]


def mesh_strip(indices1: np.ndarray, indices2: np.ndarray) -> np.ndarray:
    if (indices1.shape != indices2.shape):
        raise ValueError("The two inputs don't have the same shape.")

    faces = []

    line_len = len(indices1)
    for i in range(line_len-1):
        r1, r2 = make_quad(
            indices1[i],
            indices1[i+1],
            indices2[i],
            indices2[i+1])
        faces.append(r1)
        faces.append(r2)

    return faces


def identify_boundaries(mesh: trimesh.Trimesh) -> trimesh.path.Path3D:
    path3d_param = trimesh.path.exchange.misc.faces_to_path(mesh)
    return trimesh.path.Path3D(**path3d_param)


def arrange_mesh(input_path: str, output_path: str):
    mesh = trimesh.load_mesh(input_path)

    origin = mesh.bounding_box.centroid

    # cut into 4 parts
    front = mesh.slice_plane(
        plane_normal=[0, -1, 0], plane_origin=origin)

    back = mesh.slice_plane(
        plane_normal=[0, 1, 0], plane_origin=origin)

    fl = front.slice_plane(
        plane_normal=[-1, 0, 0], plane_origin=origin)
    fr = front.slice_plane(
        plane_normal=[1, 0, 0], plane_origin=origin)

    mirror = np.eye(4)
    mirror[0, 0] = -1  # mirror y-z plane

    parts: trimesh.Trimesh[3] = []

    parts.append(back)

    big = fl if fl.area > fr.area else fr
    parts.append(big)
    parts.append(big.copy().apply_transform(mirror))

    diff_centroid = parts[1].bounding_box.centroid - \
        parts[2].bounding_box.centroid

    diff = (((parts[1].bounds[1]-parts[1].bounds[0])+(parts[2].bounds[1]-parts[2].bounds[0]))*0.5 -
            abs(diff_centroid))*diff_centroid/np.linalg.norm(diff_centroid)

    if fl.area > fr.area:
        parts[2].apply_translation(-diff)
    else:
        parts[1].apply_translation(diff)

    helmet = trimesh.util.concatenate(parts)
    print("before merge:", len(helmet.vertices))
    helmet.merge_vertices()
    print("after merge:", len(helmet.vertices))
    helmet.fix_normals()

    helmet.export(output_path)


def align_mesh(input_path: str, output_path: str, landmarks: np.ndarray) -> np.ndarray:
    '''
    Mesh is aligned to the plane constructed from landmarks
    Returns homogeneous transformation matrix

    landmarks: three 3D points in space
    '''
    landmarks = np.array(landmarks)

    if (landmarks.shape != (3, 3)):
        raise ValueError("Landmarks must have exactly three 3D points.")

    # credit: from mesh to meaning, ch.4.7, p.104
    p1 = landmarks[0, :]  # right tragus
    p2 = landmarks[1, :]  # nasion
    p3 = landmarks[2, :]  # left tragus

    centroid = (p1+p2+p3)/3

    normal = np.cross(p3-p2, p1-p2)
    normal /= np.linalg.norm(normal)
    forward = p2-centroid
    forward /= np.linalg.norm(forward)

    target_normal = np.array([0, 0, 1])
    target_forward = np.array([0, -1, 0])

    r1 = trimesh.transformations.rotation_matrix(
        math.acos(np.dot(normal, target_normal)), np.cross(normal, target_normal))

    normal = (r1@np.append(normal, [1]))[:3]
    forward = (r1@np.append(forward, [1]))[:3]

    r2 = trimesh.transformations.rotation_matrix(
        math.acos(np.dot(forward, target_forward)), np.cross(forward, target_forward))

    mesh = trimesh.load_mesh(input_path)

    transform = r2@r1
    mesh.apply_transform(transform).export(output_path)

    return transform


def transform(vertices: np.ndarray, matrix_4x4: np.ndarray) -> np.ndarray:
    return np.matvec(matrix_4x4, np.insert(arr=vertices, obj=3, values=[1], axis=1))[:, :3]


def plane_fit(vertices: np.ndarray) -> (np.ndarray, np.ndarray):
    '''
    Returns plane_origin, plane_normal
    '''
    return trimesh.points.plane_fit(vertices)


def create_slices(
    input_path: str,
    plane_normal: typing.List[float],
    plane_origin: typing.List[float],
    n_slices: int
) -> typing.List[trimesh.path.Path3D]:
    plane_origin = np.array(plane_origin)

    # normalize normal
    plane_normal = np.array(plane_normal, dtype=float)
    mag = np.linalg.norm(plane_normal)
    plane_normal /= mag

    mesh = trimesh.load_mesh(input_path)
    max_point = mesh.vertices[np.argmax(np.dot(mesh.vertices, plane_normal.T))]

    distance = np.linalg.norm(np.dot(plane_normal, max_point-plane_origin))
    unit_distance = distance/n_slices

    sections = mesh.section_multiplane(
        plane_normal=plane_normal,
        plane_origin=plane_origin,
        heights=np.arange(stop=distance, step=unit_distance))

    for i in range(len(sections)):
        if sections[i] is None:
            continue
        sections[i] = sections[i].to_3D()

    return sections


def sample_circular_path_3D(
        path: trimesh.path.Path3D,
        origin: np.ndarray,
        begin_dir: np.ndarray,
        n_samples: int = 100
):
    planar, to_3D = path.to_2D()
    vertex_sequence = planar.discrete[0]

    # project origin to path plane
    plane_origin, plane_normal = trimesh.points.plane_fit(path.vertices)
    to_plane = trimesh.points.project_to_plane(
        [origin, origin+begin_dir], plane_origin=plane_origin, plane_normal=plane_normal, return_planar=True)
    origin = to_plane[0]
    d = to_plane[1] - origin
    begin_dir = d/np.linalg.norm(d)

    verts = []

    # find beginning
    begin_index = 0
    vert_len = len(vertex_sequence)
    for i in range(vert_len-1):
        p1 = vertex_sequence[i]
        p2 = vertex_sequence[i+1]
        mag = np.linalg.norm(p2-p1)
        intersects, intersection = intersect_rays_2D(
            origin, begin_dir, p1, (p2-p1)/mag)
        if not intersects or np.linalg.norm(intersection-p1) > mag:
            continue
        begin_index = i
        intersection = to_3D@np.append(intersection, [0, 1])
        verts.append(intersection[:3])
        break

    unit_radian = 2*math.pi/n_samples
    target_radian = unit_radian
    s = math.sin(target_radian)
    c = math.cos(target_radian)
    normal = np.array([[c, -s], [s, c]])@begin_dir
    for i in range(vert_len-1):
        p1 = vertex_sequence[(i+begin_index) % (vert_len-1)]
        p2 = vertex_sequence[(i+begin_index+1) % (vert_len-1)]
        mag = np.linalg.norm(p2-p1)
        intersects, intersection = intersect_rays_2D(
            origin, normal, p1, (p2-p1)/mag)
        if not intersects or np.linalg.norm(intersection-p1) > mag:
            continue
        intersection = to_3D@np.append(intersection, [0, 1])
        verts.append(intersection[:3])
        target_radian = unit_radian * len(verts)
        s = math.sin(target_radian)
        c = math.cos(target_radian)
        normal = np.array([[c, -s], [s, c]])@begin_dir
        if len(verts) == n_samples:
            break

    return verts


def intersect_rays_2D(r1_origin, r1_dir, r2_origin, r2_dir) -> (bool, np.ndarray):
    # credits: https://stackoverflow.com/questions/2931573/determining-if-two-rays-intersect
    r1_dir = np.array(r1_dir)/np.linalg.norm(r1_dir)
    r2_dir = np.array(r2_dir)/np.linalg.norm(r2_dir)
    r1_origin = np.array(r1_origin)
    r2_origin = np.array(r2_origin)
    d = r2_origin-r1_origin
    det = np.cross(r1_dir, r2_dir)
    if det == 0:
        return False, np.array([])
    u = np.cross(d, r2_dir)/det
    v = np.cross(d, r1_dir)/det
    return u > 0 and v > 0, r1_origin+r1_dir*u


def centroid(input_path: str) -> np.ndarray:
    return trimesh.load_mesh(input_path).centroid


def find_cephalic_index(path: trimesh.path.Path3D) -> float:
    planar, to_3D = path.to_2D()
    size = planar.bounds[1]-planar.bounds[0]
    return size[0]*100/size[1]


def gaussian(x: float = 0, a: float = 1, b: float = 0, c: float = 1) -> float:
    '''
    credit: https://en.wikipedia.org/wiki/Gaussian_function
    x - function input
    a - height
    b - x translation
    c - width, bump dies off at around b+-c/0.3
    '''
    return a*math.exp(-(x-b)**2/(2*c**2))


def inflate(vertices: np.ndarray, origin: np.ndarray, normal: np.ndarray, height: float = 1, radius: float = 1) -> np.ndarray:
    vertices = vertices.copy()
    d = np.linalg.norm(x=vertices-origin, axis=1)
    cond = d < radius
    candidates = vertices[cond]
    for i in range(len(candidates)):
        candidates[i] += normal * \
            gaussian(d[cond][i], height, 0, radius*0.3)
    vertices[cond] = candidates
    return vertices


def show_norms(mesh: trimesh.Trimesh, scale: float = 1):
    lines = []
    for i in range(len(mesh.vertices)):
        lines.append([mesh.vertices[i], mesh.vertices[i] +
                     mesh.vertex_normals[i]*scale])
    trimesh.Scene(geometry=[trimesh.load_path(lines), mesh]).show()


def average_vectors(vectors: np.ndarray) -> np.ndarray:
    return sum(vectors) / len(vectors)


def generate_helmet(
        input_path: str,
        output_path: str,
        cut_origin: np.ndarray,
        cut_normal: np.ndarray,
        n_slices: int = 10,
        n_samples: int = 25,
):
    clean_mesh(input_path, output_path)

    x = np.array([1, 0, 0])
    slices = create_slices(output_path, plane_normal=cut_normal,
                           plane_origin=cut_origin, n_slices=n_slices)

    sampled = [
        sample_circular_path_3D(
            path=slices[i],
            origin=centroid(output_path),
            begin_dir=x,
            n_samples=n_samples)
        for i in range(len(slices))
    ]

    indices = np.arange(n_samples)
    # indices = np.append(indices, [0])
    faces = [
        mesh_strip(indices+n_samples*i, indices+n_samples*(i+1))
        for i in range(len(sampled)-1)
    ]

    trimesh.Trimesh(
        vertices=np.vstack(tuple(sampled)),
        faces=np.vstack(tuple(faces)))\
        .export(output_path)

    arrange_mesh(output_path, output_path)

    thicken(output_path, output_path, thickness=4)

    # show_norms(trimesh.load_mesh(output_path), 5)
