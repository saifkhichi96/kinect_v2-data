import sys

from PyQt5 import QtWidgets
from qt_material import apply_stylesheet

from views.RecordWindow import RecordWindow


def main():
    # create the application and the main window
    app = QtWidgets.QApplication(sys.argv)

    # device, listener, registration = KinectV2.start_device()
    window = RecordWindow()

    # setup stylesheet
    apply_stylesheet(app, theme='dark_teal.xml')

    # run
    window.show()
    app.exit(app.exec_())


if __name__ == '__main__':
    main()
