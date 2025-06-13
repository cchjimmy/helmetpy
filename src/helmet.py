import trimesh
import numpy as np
import math


def load_mesh(stlPath: str) -> trimesh.Trimesh:
    return trimesh.load_mesh(stlPath)


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.remove_degenerate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()

    # trimesh.smoothing.filter_humphrey(mesh, iterations=100)
    trimesh.smoothing.filter_laplacian(mesh)
    # trimesh.smoothing.filter_mut_dif_laplacian(mesh)
    # trimesh.smoothing.filter_taubin(mesh, iterations=1000, lamb=0.5)

    # mesh.update_faces(np.sum((mesh.triangles_center-mesh.centroid) *
    #                   mesh.face_normals, axis=1) > 0.95)

    return mesh.convex_hull


def crop_mesh(mesh: trimesh.Trimesh, radius: float = 20) -> trimesh.Trimesh:
    bb = mesh.bounding_box
    bb_size = bb.bounds[1]-bb.bounds[0]
    box = trimesh.primitives.Box(extents=bb_size)
    box.apply_translation(-bb_size*[0, 0, 0.5])

    cylinder = trimesh.primitives.Cylinder(radius, bb_size[0]*1.1)
    cylinder.apply_transform(
        trimesh.transformations.rotation_matrix(0.5*math.pi, [0, 1, 0]))
    # cylinder.apply_translation([0, 0.75*radius, 0])
    # mesh = mesh.difference(cylinder)

    sideburn = trimesh.primitives.Box(
        extents=[bb_size[0]*1.1, radius, bb_size[2]])
    sideburn.apply_translation([0, -radius*1.5, 0])
    box = box.difference(sideburn)

    back = trimesh.primitives.Box(
        extents=[bb_size[0]*1.1, (bb_size[1]*0.5-radius)*1.1, bb_size[2]])
    back.apply_translation([0, bb_size[1]*0.25+radius*0.5, 0])
    box = box.difference(back)

    # mesh = mesh.difference(box)

    mesh = trimesh.util.concatenate(mesh, [cylinder, box])

    return mesh


def arrange_mesh(
        mesh: trimesh.Trimesh,
) -> trimesh.Trimesh:
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

    return helmet


def apply_thickness(mesh: trimesh.Trimesh, thickness: float) -> trimesh.Trimesh:
    if thickness <= 0:
        raise ValueError("thickness must be > 0.")
    inner = mesh.copy()
    inner.invert()
    outer = mesh.copy()
    outer.vertices += outer.vertex_normals*thickness
    combined = trimesh.util.concatenate(inner, outer)
    return combined


def show_vert_norms(mesh: trimesh.Trimesh, scale: float = 1):
    lines = [0]*len(mesh.vertices)
    for i in range(len(mesh.vertices)):
        lines[i] = [mesh.vertices[i], mesh.vertices[i] +
                    mesh.vertex_normals[i]*scale]
    trimesh.load_path(lines).show()


def generateHelm(ctScanPath: str, outputPath: str) -> trimesh.Trimesh:
    helmet = load_mesh(ctScanPath)
    helmet = clean_mesh(helmet)
    helmet = crop_mesh(helmet)
    helmet.show(flags={"wireframe": True})
    helmet = arrange_mesh(helmet)
    helmet = apply_thickness(helmet, 10)

    # export mesh
    helmet.export(outputPath)

    # visualization
    helmet.show(flags={'wireframe': True})  # requires scipy and pyglet
    # helmet.show()

    return helmet
