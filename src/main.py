import sys
import gui
from PySide6 import QtWidgets
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",  help="3D model file path, *.stl")
    parser.add_argument(
        "-o", "--output", help="3D model file path, *.stl")

    args = parser.parse_args(sys.argv[1:])

    app = QtWidgets.QApplication()
    gui = gui.HelmetGui(args)
    gui.show()
    sys.exit(app.exec())
