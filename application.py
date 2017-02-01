#! /usr/bin/python3
# -*- coding: utf-8 -*-

import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QWidget, QApplication, QMainWindow, QFileDialog, QAction)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle('Sample Window')
        self.setMinimumHeight(400)
        self.setMinimumWidth(600)
        self.statusBar()

        openFile = QAction(QIcon('open.png'), '開く', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('新しいファイルを開く')
        openFile.triggered.connect(self.show_dialog)

        menu = self.menuBar()
        fileMenu = menu.addMenu('&ファイル')
        fileMenu.addAction(openFile)

    def show_dialog(self):
        filter = "Image Files (*.csv *.txt *.html *.xml *.py *.pyw)"
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/', filter=filter)

        if fname[0]:
            f = open(fname[0], 'r')

            with f:
                data = f.read()
                self.textEdit.setText(data)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()

    mainWindow.show()
    sys.exit(app.exec_())
