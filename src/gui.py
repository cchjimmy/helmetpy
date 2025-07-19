from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor
import pyvista
import helmet
import numpy
import typing
import tempfile


class HelmetGui(QtWidgets.QWidget):
    def __init__(self, args):
        super().__init__()

        self.setWindowTitle("Helmet GUI")

        self.input_path = args.input
        self.output_path = args.output

        self.layout = QtWidgets.QHBoxLayout()

        self.setLayout(self.layout)

        self.plotter = QtInteractor(parent=self)
        self.plotter.add_axes()
        if (self.input_path):
            self.mesh = pyvista.read(self.input_path)
            self.show_mesh()
        self.layout.addWidget(self.plotter)

        self.buttons = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.buttons)

        self.add_button("Import Mesh", self.import_mesh)
        self.add_button("Clean Mesh", self.clean_mesh)
        self.add_button("Align Mesh", self.align_mesh,
                        "Select three points in order: right tragus, nasion, left tragus.")
        self.add_button("Save Mesh", self.save_mesh)
        self.add_button("Slice Mesh", self.slice_mesh)

    def add_button(self, name: str, cb: typing.Callable[..., None], tool_tip: str = ""):
        button = QtWidgets.QPushButton(name)
        button.clicked.connect(cb)
        button.setToolTip(tool_tip)
        self.buttons.addWidget(button)

    @QtCore.Slot()
    def import_mesh(self):
        self.input_path = self.open_file_dialog()
        self.mesh = pyvista.read(self.input_path)
        self.show_mesh()

    def show_mesh(self, wireframe=False):
        self.plotter.clear()
        self.plotter.add_light(pyvista.Light(light_type="headlight"))
        self.mesh.compute_normals(inplace=True)
        actor = self.plotter.add_mesh(self.mesh, culling="back")
        if wireframe:
            prop = actor.GetProperty()
            prop.SetRepresentationToWireframe()
        self.plotter.view_xz()
        self.plotter.reset_camera()

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
        self.landmarks = []
        self.plotter.enable_point_picking(
            callback=self.point_pick_cb)

    def point_pick_cb(self, picked_point):
        self.landmarks.append(picked_point)
        self.plotter.add_point_labels(
            points=[picked_point], labels=[picked_point], point_size=20, point_color="#FF0000")
        if len(self.landmarks) < 3:
            return
        self.plotter.disable_picking()
        tmp = tempfile.NamedTemporaryFile(suffix=".stl")
        self.mesh.save(tmp.name)
        helmet.align_mesh(tmp.name, tmp.name, self.landmarks)
        self.mesh = pyvista.read(tmp.name)
        tmp.close()
        self.show_mesh()

    @QtCore.Slot()
    def save_mesh(self):
        if not self.output_path:
            self.output_path = self.save_file_dialog()
        self.mesh.save(self.output_path)

    @QtCore.Slot()
    def clean_mesh(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".stl")
        self.mesh.save(tmp.name)
        helmet.clean_mesh(tmp.name, tmp.name)
        self.mesh = pyvista.read(tmp.name)
        tmp.close()
        self.show_mesh()

    @QtCore.Slot()
    def slice_mesh(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".stl")
        self.mesh.save(tmp.name)
        helmet.generate_helmet(tmp.name, tmp.name)
        self.mesh = pyvista.read(tmp.name)
        tmp.close()
        self.show_mesh()
