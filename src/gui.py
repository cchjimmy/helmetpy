from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor
import pyvista
import helmet
import typing
import tempfile
import tomllib
import io


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
            self.mesh = pyvista.read(self.input_path)
            self.show_mesh()
        self.layout.addWidget(self.plotter)

        self.buttons = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.buttons)

        self.add_button("Import Mesh", self.import_mesh)
        self.add_button("Save Mesh", self.save_mesh)
        # self.add_button("Clean Mesh", self.clean_mesh)
        self.add_button("Align Mesh", self.align_mesh,
                        "Select three points in order: right tragus, nasion, left tragus.")
        self.add_button("Generate Helmet", self.generate_helmet)
        self.add_button("Cycle Views", self.cycle_representations)
        self.add_button("Revert", self.revert)
        self.add_button("Inflate", self.inflate)

    def add_button(self, name: str, cb: typing.Callable[..., None], tool_tip: str = ""):
        button = QtWidgets.QPushButton(name)
        button.clicked.connect(cb)
        button.setToolTip(tool_tip)
        self.buttons.addWidget(button)

    def init_plotter(self):
        self.plotter = QtInteractor(parent=self)
        self.plotter.add_axes()
        self.plotter.view_xz()

    @QtCore.Slot()
    def import_mesh(self):
        self.input_path = self.open_file_dialog()
        self.mesh = pyvista.read(self.input_path)
        self.show_mesh()

    def show_mesh(self):
        self.plotter.remove_actor("mesh")
        actor = self.plotter.add_mesh(
            mesh=self.mesh, culling="back", name="mesh")
        self.plotter.reset_camera()
        self.apply_representation(actor)

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

    def open_file_dialog(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open File",
            filter="stl (*.stl)"
        )
        return file_path

    def save_file_dialog(self):
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

        self.plotter.disable_picking()
        with tempfile.NamedTemporaryFile(suffix=".stl") as tmp:
            self.mesh.save(tmp.name)
            helmet.align_mesh(tmp.name, tmp.name, self.landmarks)
            self.mesh.points = pyvista.read(tmp.name).points

        for label in self.labels:
            self.plotter.remove_actor(label)

        self.show_mesh()
        self.plotter.view_xz()

    @QtCore.Slot()
    def save_mesh(self):
        if not self.output_path:
            self.output_path = self.save_file_dialog()
        self.mesh.save(self.output_path)

    @QtCore.Slot()
    def clean_mesh(self):
        with tempfile.NamedTemporaryFile(suffix=".stl") as tmp:
            self.mesh.save(tmp.name)
            helmet.clean_mesh(tmp.name, tmp.name)
            self.mesh = pyvista.read(tmp.name)
        self.show_mesh()

    @QtCore.Slot()
    def generate_helmet(self):
        with tempfile.NamedTemporaryFile(suffix=".stl") as tmp:
            self.mesh.save(tmp.name)
            helmet.generate_helmet(tmp.name, tmp.name)
            self.mesh = pyvista.read(tmp.name)
        self.show_mesh()

    @QtCore.Slot()
    def cycle_representations(self):
        rep_type = pyvista.plotting.opts.RepresentationType
        self.representation = (self.representation+1) % len(rep_type)
        try:
            self.apply_representation(self.plotter.actors["mesh"])
        except KeyError:
            pass

    @QtCore.Slot()
    def revert(self):
        self.mesh = pyvista.read(self.input_path)
        self.show_mesh()
        self.plotter.view_xz()

    @QtCore.Slot()
    def inflate(self):
        self.plotter.disable_picking()
        self.plotter.enable_surface_point_picking(
            callback=self.inflate_cb, show_point=False)

    def inflate_cb(self, picked_point):
        with io.open("./config.toml", mode="rb") as config:
            self.config = tomllib.load(config)

        closest_cell = self.mesh.find_closest_cell(picked_point)

        self.mesh.points = helmet.inflate(
            vertices=self.mesh.points,
            origin=picked_point,
            direction=self.mesh.cell_normals[closest_cell],
            radius=self.config["inflate"]["radius"],
            height=self.config["inflate"]["height"])
        self.show_mesh()
