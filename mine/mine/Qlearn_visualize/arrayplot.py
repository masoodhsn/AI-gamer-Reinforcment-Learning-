import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

class MatrixViewer:
    def __init__(self, data):

        self.data = data
        self.rows, self.cols = data.shape

        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        self.win = pg.GraphicsLayoutWidget()
        self.view = self.win.addViewBox()
        self.view.setAspectLocked(True)
        self.win.show()

        # picture
        self.img_item = pg.ImageItem(self.data)
        self.view.addItem(self.img_item)

        lut = pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256)
        self.img_item.setLookupTable(lut)
        self.img_item.setLevels([np.min(self.data), np.max(self.data)])

        
        self.highlight = pg.RectROI([0, 0], [1, 1], pen=pg.mkPen('red', width=2))
        self.highlight.setZValue(10)
        self.highlight.setVisible(False)
        self.view.addItem(self.highlight)

        self.label = pg.TextItem(anchor=(0, 1), color='w')
        self.label.setZValue(11)
        self.view.addItem(self.label)

        # mouse connection
        self.proxy = pg.SignalProxy(self.view.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

    def mouseMoved(self, evt):
        pos = evt[0]  
        mouse_point = self.img_item.mapFromScene(pos)
        x, y = int(mouse_point.x()), int(mouse_point.y())

        if 0 <= x < self.cols and 0 <= y < self.rows:
            val = self.data[y, x]
            self.label.setText(f"x={x}, y={y}, val={val:.2f}")
            self.label.setPos(x + 1, y)
            self.highlight.setPos([x, y])
            self.highlight.setSize([1, 1])
            self.highlight.setVisible(True)
        else:
            self.highlight.setVisible(False)
            self.label.setText("")

    def update_data(self, new_data):

        if new_data.shape != (self.rows, self.cols):
            raise ValueError("اdiffrent shape")
        self.data = new_data
        self.img_item.setImage(self.data, autoLevels=False)
        # update
        self.img_item.setLevels([np.min(self.data), np.max(self.data)])

    def exec(self):
        self.app.exec()
