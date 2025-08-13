from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor
import pyvista
import helmet
import typing
import tempfile
import tomllib
import io
from enum import Enum


class MeshNames(Enum):
    HELMET = "helmet"
    ORIGINAL = "original"


class ButtonNames(Enum):
    IMPORT = "Import Mesh"
    SAVE = "Save Mesh"
    ALIGN = "Align Mesh"
    GENERATE = "Generate Helmet"
    REVERT = "Revert"
    INFLATE = "Inflate"
    CYCLE_VIEWS = "Cycle Views"
    TOGGLE_ORIGINAL = "Toggle Original"
    SAVE_LANDMARKS = "Save Landmarks"
    LOAD_LANDMARKS = "Load Landmarks"


class HelmetGui(QtWidgets.QWidget):
    def __init__(self, args):
        super().__init__()

        with io.open(file="./config.toml", mode="rb") as config:
            self.config = tomllib.load(config)

        self.setWindowTitle("Helmet GUI")

        self.input_path = args.input
        self.output_path = args.output
        self.representation = pyvista.plotting.opts.RepresentationType.SURFACE

        self.init_plotter()

        self.layout = QtWidgets.QHBoxLayout()

        self.setLayout(self.layout)

        if (self.input_path):
            mesh = pyvista.read(self.input_path)
            self.add_mesh(mesh, MeshNames.ORIGINAL, show=True)

        self.layout.addWidget(self.plotter)

        self.buttons = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.buttons)

        self.add_button(ButtonNames.IMPORT, self.import_mesh)
        self.add_button(ButtonNames.SAVE, self.save_mesh)
        self.add_button(ButtonNames.ALIGN, self.align_mesh,
                        "Select three points in order: temporal right, glabella, temporal left.")
        self.add_button(ButtonNames.GENERATE, self.generate_helmet,
                        tool_tip="Must align mesh first.", enabled=False)
        self.add_button(ButtonNames.CYCLE_VIEWS, self.cycle_representations)
        self.add_button(ButtonNames.REVERT, self.revert)
        self.add_button(ButtonNames.INFLATE, self.inflate)
        self.add_button(ButtonNames.TOGGLE_ORIGINAL,
                        self.toggle_original_visibility)
        self.add_button(ButtonNames.SAVE_LANDMARKS,
                        self.save_landmarks, enabled=False)
        self.add_button(ButtonNames.LOAD_LANDMARKS, self.load_landmarks,
                        "Load landmarks and align mesh.")

    def add_button(self, name: ButtonNames, cb: typing.Callable[..., None], tool_tip: str = "", enabled: bool = True):
        button = QtWidgets.QPushButton(name.value)
        button.clicked.connect(cb)
        button.setToolTip(tool_tip)
        button.setEnabled(enabled)
        button.setObjectName(name.value)
        self.buttons.addWidget(button)

    def find_button(self, name: ButtonNames) -> QtWidgets.QPushButton:
        return self.findChild(QtWidgets.QPushButton, name.value)

    def init_plotter(self) -> QtInteractor:
        self.plotter = QtInteractor(parent=self)
        self.plotter.add_axes()
        self.plotter.view_xz()

    @QtCore.Slot()
    def import_mesh(self):
        self.input_path = self.open_file_dialog(filter_string="Mesh (*.stl)")
        if self.input_path == "":
            return
        mesh = pyvista.read(self.input_path)
        self.add_mesh(mesh, MeshNames.ORIGINAL, True)
        self.plotter.view_xz()

    def add_mesh(self, mesh: pyvista.PolyData, name: MeshNames, show=True) -> pyvista.Actor:
        actor = self.plotter.add_mesh(
            mesh=mesh, culling="back", name=name.value)
        self.plotter.reset_camera()
        if name == MeshNames.HELMET:
            self.apply_representation(actor)
        actor.visibility = show
        return actor

    def get_mesh(self, name: MeshNames) -> pyvista.PolyData:
        return self.plotter.actors[name.value].mapper.dataset

    @QtCore.Slot()
    def toggle_original_visibility(self):
        self.set_mesh_visibility(
            MeshNames.ORIGINAL, not self.get_mesh_visibility(MeshNames.ORIGINAL))

    def get_mesh_visibility(self, name: MeshNames):
        return self.plotter.actors[name.value].visibility

    def set_mesh_visibility(self, name: MeshNames, show=True):
        self.plotter.actors[name.value].visibility = show

    def apply_representation(self, actor):
        prop = actor.GetProperty()
        rep_type = pyvista.plotting.opts.RepresentationType
        match self.representation:
            case rep_type.POINTS:
                prop.SetRepresentationToPoints()
            case rep_type.WIREFRAME:
                prop.SetRepresentationToWireframe()
            case _:
                prop.SetRepresentationToSurface()

    def open_file_dialog(self, filter_string: str = ""):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open File",
            filter=filter_string
        )
        return file_path

    def save_file_dialog(self, filter_string: str = ""):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent=self, caption="Save File")
        return file_path

    @QtCore.Slot()
    def align_mesh(self):
        self.plotter.disable_picking()
        self.landmarks = []
        self.labels = []
        self.plotter.enable_surface_point_picking(
            callback=self.align_mesh_cb, show_point=False)

    def align_mesh_cb(self, picked_point):
        self.landmarks.append(picked_point)
        self.labels.append(self.plotter.add_point_labels(
            points=[picked_point], labels=[picked_point], show_points=False, name="point"+str(len(self.labels))))
        if len(self.landmarks) < 3:
            return

        with tempfile.NamedTemporaryFile(suffix=".stl") as tmp:
            mesh = self.get_mesh(MeshNames.ORIGINAL)
            mesh.save(tmp.name)
            self.transform = helmet.align_mesh(
                tmp.name, tmp.name, self.landmarks)
            original = pyvista.read(self.input_path)
            original.points = helmet.transform(original.points, self.transform)
            self.add_mesh(original, MeshNames.ORIGINAL, True)

        for label in self.labels:
            self.plotter.remove_actor(label)

        self.plotter.view_xz()
        self.plotter.disable_picking()
        self.find_button(ButtonNames.GENERATE).setEnabled(True)
        self.find_button(ButtonNames.SAVE_LANDMARKS).setEnabled(True)

    @QtCore.Slot()
    def save_mesh(self):
        if not self.output_path:
            self.output_path = self.save_file_dialog()
        if self.output_path == "":
            return
        self.get_mesh(MeshNames.HELMET).save(self.output_path)

    @QtCore.Slot()
    def generate_helmet(self):
        with io.open(file="./config.toml", mode="rb") as config:
            self.config = tomllib.load(config)

        with tempfile.NamedTemporaryFile(suffix=".stl") as tmp:
            self.get_mesh(MeshNames.ORIGINAL).save(tmp.name)
            plane_origin, plane_normal = helmet.plane_fit(
                helmet.transform(self.landmarks, self.transform))
            print("--- Begin helmet generation ---")
            helmet.generate_helmet(
                input_path=tmp.name,
                output_path=tmp.name,
                cut_origin=plane_origin,
                cut_normal=plane_normal,
                n_samples=self.config["sampling"]["n_radial_samples"],
                n_slices=self.config["sampling"]["n_vertical_slices"],
                thickness=self.config["helmet"]["thickness"],
                enlarge_displacement=self.config["helmet"]["enlarge_displacement"]
            )
            print("--- End of helmet generation ---\n")
            self.add_mesh(pyvista.read(tmp.name), MeshNames.HELMET)

        self.find_button(ButtonNames.GENERATE).setEnabled(False)
        self.set_mesh_visibility(MeshNames.ORIGINAL, False)
        self.set_mesh_visibility(MeshNames.HELMET, True)
        self.plotter.reset_camera()
        self.plotter.view_xz()

    @QtCore.Slot()
    def cycle_representations(self):
        rep_type = pyvista.plotting.opts.RepresentationType
        self.representation = (self.representation+1) % len(rep_type)
        if MeshNames.HELMET.value not in self.plotter.actors:
            return
        self.apply_representation(
            actor=self.plotter.actors[MeshNames.HELMET.value])

    @QtCore.Slot()
    def revert(self):
        self.add_mesh(pyvista.read(self.input_path), MeshNames.ORIGINAL)
        self.set_mesh_visibility(MeshNames.HELMET, False)
        self.plotter.view_xz()

    @QtCore.Slot()
    def inflate(self):
        self.plotter.disable_picking()
        self.plotter.enable_surface_point_picking(
            callback=self.inflate_cb, show_point=False)

    def inflate_cb(self, picked_point):
        with io.open("./config.toml", mode="rb") as config:
            self.config = tomllib.load(config)

        mesh = self.get_mesh(MeshNames.HELMET)

        closest_cell = mesh.find_closest_cell(picked_point)

        mesh = self.get_mesh(MeshNames.HELMET)
        mesh.points = helmet.inflate(
            vertices=mesh.points,
            origin=picked_point,
            normal=mesh.cell_normals[closest_cell],
            radius=self.config["inflate"]["radius"],
            height=self.config["inflate"]["height"])

    @QtCore.Slot()
    def save_landmarks(self):
        if "landmarks" not in self:
            return
        save_path = self.save_file_dialog("npy (*.npy)")
        if save_path == "":
            return
        helmet.save_np_array(arr=self.landmarks,
                             save_path=save_path)

    @QtCore.Slot()
    def load_landmarks(self):
        load_path = self.open_file_dialog("npy (*.npy)")
        if load_path == "":
            return
        self.landmarks = helmet.load_np_array(
            load_path=load_path)
        if self.landmarks.shape != (3, 3):
            return
        landmarks = self.landmarks.copy()
        self.landmarks = []
        self.labels = []
        for i in range(len(landmarks)):
            self.align_mesh_cb(landmarks[i])

        self.find_button(ButtonNames.GENERATE).setEnabled(True)
        self.find_button(ButtonNames.SAVE_LANDMARKS).setEnabled(True)
