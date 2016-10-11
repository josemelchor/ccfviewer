import sys, os, traceback
sys.path.append(os.path.join(os.path.dirname(__file__)))

import json
from collections import OrderedDict
import numpy as np
import pyqtgraph as pg
import pyqtgraph.functions as fn
import pyqtgraph.metaarray as metaarray
from pyqtgraph.Qt import QtGui, QtCore
import math
from numpy import cross, eye, dot
from scipy.linalg import expm3, norm


class AtlasBuilder(QtGui.QMainWindow):
    def __init__(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


class AtlasViewer(QtGui.QWidget):
    def __init__(self, parent=None):
        self.atlas = None
        self.label = None

        QtGui.QWidget.__init__(self, parent)
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0,0,0,0)

        self.splitter = QtGui.QSplitter()
        self.layout.addWidget(self.splitter, 0, 0)

        self.view = VolumeSliceView()
        self.view.mouseHovered.connect(self.mouseHovered)
        self.view.mouseClicked.connect(self.mouseClicked)
        self.splitter.addWidget(self.view)
        
        self.statusLabel = QtGui.QLabel()
        self.layout.addWidget(self.statusLabel, 1, 0, 1, 1)
        self.statusLabel.setFixedHeight(30)

        self.pointLabel = QtGui.QLabel()
        self.layout.addWidget(self.pointLabel, 2, 0, 1, 1)
        self.pointLabel.setFixedHeight(30)

        self.ctrl = QtGui.QWidget(parent=self)
        self.splitter.addWidget(self.ctrl)
        self.ctrlLayout = QtGui.QVBoxLayout()
        self.ctrl.setLayout(self.ctrlLayout)

        self.displayCtrl = LabelDisplayCtrl(parent=self)
        self.ctrlLayout.addWidget(self.displayCtrl)
        self.displayCtrl.params.sigTreeStateChanged.connect(self.displayCtrlChanged)

        self.labelTree = LabelTree(self)
        self.labelTree.labelsChanged.connect(self.labelsChanged)
        self.ctrlLayout.addWidget(self.labelTree)

    def setLabels(self, label):
        self.label = label
        with SignalBlock(self.labelTree.labelsChanged, self.labelsChanged):
            for rec in label._info[-1]['ontology']:
                self.labelTree.addLabel(*rec)
        self.updateImage()
        self.labelsChanged()

    def setAtlas(self, atlas):
        self.atlas = atlas
        self.updateImage()

    def updateImage(self):
        if self.atlas is None or self.label is None:
            return
        axis = self.displayCtrl.params['Orientation']
        axes = {
            'right': ('right', 'anterior', 'dorsal'),
            'dorsal': ('dorsal', 'right', 'anterior'),
            'anterior': ('anterior', 'right', 'dorsal')
        }[axis]
        order = [self.atlas._interpretAxis(ax) for ax in axes]

        # transpose, flip, downsample images
        ds = self.displayCtrl.params['Downsample']
        self.displayAtlas = self.atlas.view(np.ndarray).transpose(order)
        with pg.BusyCursor():
            for ax in (0, 1, 2):
                self.displayAtlas = pg.downsample(self.displayAtlas, ds, axis=ax)
        self.displayLabel = self.label.view(np.ndarray).transpose(order)[::ds, ::ds, ::ds]

        # make sure atlas/label have the same size after downsampling

        self.view.setData(self.displayAtlas, self.displayLabel, scale=self.atlas._info[-1]['vxsize']*ds)

    def labelsChanged(self):
        lut = self.labelTree.lookupTable()
        self.view.setLabelLUT(lut)        
        
    def displayCtrlChanged(self, param, changes):
        update = False
        for param, change, value in changes:
            if param.name() == 'Composition':
                self.view.setOverlay(value)
            elif param.name() == 'Opacity':
                self.view.setLabelOpacity(value)
            else:
                update = True
        if update:
            self.updateImage()

    def mouseHovered(self, id):
        self.statusLabel.setText(self.labelTree.describe(id))
        
    def renderVolume(self):
        import pyqtgraph.opengl as pgl
        import scipy.ndimage as ndi
        self.glView = pgl.GLViewWidget()
        img = np.ascontiguousarray(self.displayAtlas[::8,::8,::8])
        
        # render volume
        #vol = np.empty(img.shape + (4,), dtype='ubyte')
        #vol[:] = img[..., None]
        #vol = np.ascontiguousarray(vol.transpose(1, 2, 0, 3))
        #vi = pgl.GLVolumeItem(vol)
        #self.glView.addItem(vi)
        #vi.translate(-vol.shape[0]/2., -vol.shape[1]/2., -vol.shape[2]/2.)
        
        verts, faces = pg.isosurface(ndi.gaussian_filter(img.astype('float32'), (2, 2, 2)), 5.0)
        md = pgl.MeshData(vertexes=verts, faces=faces)
        mesh = pgl.GLMeshItem(meshdata=md, smooth=True, color=[0.5, 0.5, 0.5, 0.2], shader='balloon')
        mesh.setGLOptions('additive')
        mesh.translate(-img.shape[0]/2., -img.shape[1]/2., -img.shape[2]/2.)
        self.glView.addItem(mesh)

        self.glView.show()
     
    # mouse_point[0] contains the Point object.
    # mouse_point[1] contains the structure id at Point
    def mouseClicked(self, mouse_point):
        point, to_clipboard = self.getCcfPoint(mouse_point)
        self.pointLabel.setText(point)
        self.view.target.setVisible(True)
        self.view.target.setPos(self.view.view2.mapSceneToView(mouse_point[0].scenePos()))
        self.view.clipboard.setText(to_clipboard)

    # Get CCF point coordinate and Structure id
    # Returns two strings. One used for display in a label and the other to put in the clipboard
    # PIR orientation where x axis = Anterior-to-Posterior, y axis = Superior-to-Inferior and z axis = Left-to-Right
    def getCcfPoint(self, mouse_point):

        axis = self.displayCtrl.params['Orientation']
        vxsize = self.atlas._info[-1]['vxsize'] * 1e6
        z_axis_rotated = self.view.slider.value() != 0

        if z_axis_rotated:
            p1 = self.view.mappedCoords[0][int(mouse_point[0].pos().x())][int(mouse_point[0].pos().y())]
            p2 = self.view.mappedCoords[1][int(mouse_point[0].pos().x())][int(mouse_point[0].pos().y())]
            p3 = self.view.mappedCoords[2][int(mouse_point[0].pos().x())][int(mouse_point[0].pos().y())]
        else:
            p1 = self.view.mappedCoords[0][int(mouse_point[0].pos().y())]
            p2 = self.view.mappedCoords[1][int(mouse_point[0].pos().y())]
            p3 = mouse_point[0].pos().x()

        # check ccf bounds, change to PIR orientation if within bounds
        if p1 > self.view.atlas.shape[1] or p1 < 0:
            p1 = 'N/A'
        else:
            p1 = abs(p1 - self.view.atlas.shape[1]) * vxsize

        if p2 > self.view.atlas.shape[2] or p2 < 0:
            p2 = 'N/A'
        else:
            p2 = abs(p2 - self.view.atlas.shape[2]) * vxsize

        if p3 > self.view.atlas.shape[0] or p3 < 0:
            p3 = 'N/A'
        else:
            if z_axis_rotated:
                p3 = abs(self.view.atlas.shape[0] - p3) * vxsize
            else:
                p3 = abs(p3 - self.view.atlas.shape[0]) * vxsize

        if axis == 'right':
            point = "x: " + str(p1) + " y: " + str(p2) + " z: " + str(p3) + " StructureID: " + str(mouse_point[1])
            clipboard_text = str(p1) + ";" + str(p2) + ";" + str(p3) + ";" + str(mouse_point[1])
        elif axis == 'anterior':
            point = "x: " + str(p3) + " y: " + str(p2) + " z: " + str(p1) + " StructureID: " + str(mouse_point[1])
            clipboard_text = str(p3) + ";" + str(p2) + ";" + str(p1) + ";" + str(mouse_point[1])
        elif axis == 'dorsal':
            point = "x: " + str(p2) + " y: " + str(p3) + " z: " + str(p1) + " StructureID: " + str(mouse_point[1])
            clipboard_text = str(p2) + ";" + str(p3) + ";" + str(p1) + ";" + str(mouse_point[1])
        else:
            point = 'N/A'
            clipboard_text = 'NULL'

        return point, clipboard_text


class LabelDisplayCtrl(pg.parametertree.ParameterTree):
    def __init__(self, parent=None):
        pg.parametertree.ParameterTree.__init__(self, parent=parent)
        params = [
            {'name': 'Orientation', 'type': 'list', 'values': ['right', 'anterior', 'dorsal']},
            {'name': 'Opacity', 'type': 'float', 'limits': [0, 1], 'value': 0.5, 'step': 0.1},
            {'name': 'Composition', 'type': 'list', 'values': ['Multiply', 'Overlay', 'SourceOver']},
            {'name': 'Downsample', 'type': 'int', 'value': 1, 'limits': [1, None], 'step': 1},
        ]
        self.params = pg.parametertree.Parameter(name='params', type='group', children=params)
        self.setParameters(self.params, showTop=False)
        self.setHeaderHidden(True)


class LabelTree(QtGui.QWidget):
    labelsChanged = QtCore.Signal()

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0,0,0,0)

        self.tree = QtGui.QTreeWidget(self)
        self.layout.addWidget(self.tree, 0, 0)
        self.tree.header().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.tree.headerItem().setText(0, "id")
        self.tree.headerItem().setText(1, "name")
        self.tree.headerItem().setText(2, "color")
        self.labelsById = {}
        self.labelsByAcronym = {}
        self.checked = set()
        self.tree.itemChanged.connect(self.itemChange)

        self.layerBtn = QtGui.QPushButton('Color by cortical layer')
        self.layout.addWidget(self.layerBtn, 1, 0)
        self.layerBtn.clicked.connect(self.colorByLayer)

        self.resetBtn = QtGui.QPushButton('Reset colors')
        self.layout.addWidget(self.resetBtn, 2, 0)
        self.resetBtn.clicked.connect(self.resetColors)

    def addLabel(self, id, parent, name, acronym, color):
        item = QtGui.QTreeWidgetItem([acronym, name, ''])
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(0, QtCore.Qt.Unchecked)

        if parent in self.labelsById:
            root = self.labelsById[parent]['item']
        else:
            root = self.tree.invisibleRootItem()

        root.addChild(item)

        btn = pg.ColorButton(color=pg.mkColor(color))
        btn.defaultColor = color
        self.tree.setItemWidget(item, 2, btn)

        self.labelsById[id] = {'item': item, 'btn': btn}
        item.id = id
        self.labelsByAcronym[acronym] = self.labelsById[id]

        btn.sigColorChanged.connect(self.itemColorChanged)
        # btn.sigColorChanging.connect(self.imageChanged)

    def itemChange(self, item, col):
        checked = item.checkState(0) == QtCore.Qt.Checked
        with SignalBlock(self.tree.itemChanged, self.itemChange):
            self.checkRecursive(item, checked)
        self.labelsChanged.emit()

    def checkRecursive(self, item, checked):
        if checked:
            self.checked.add(item.id)
            item.setCheckState(0, QtCore.Qt.Checked)
        else:
            if item.id in self.checked:
                self.checked.remove(item.id)
            item.setCheckState(0, QtCore.Qt.Unchecked)

        for i in range(item.childCount()):
            self.checkRecursive(item.child(i), checked)

    def itemColorChanged(self, *args):
        self.labelsChanged.emit()

    def lookupTable(self):
        lut = np.zeros((2**16, 4), dtype=np.ubyte)
        for id in self.checked:
            if id >= lut.shape[0]:
                continue
            lut[id] = self.labelsById[id]['btn'].color(mode='byte')
        return lut

    def colorByLayer(self, root=None):
        try:
            unblock = False
            if not isinstance(root, pg.QtGui.QTreeWidgetItem):
                self.blockSignals(True)
                unblock = True
                root = self.labelsByAcronym['Isocortex']['item']

            name = str(root.text(1))
            if ', layer' in name.lower():
                layer = name.split(' ')[-1]
                layer = {'1': 0, '2': 1, '2/3': 2, '4': 3, '5': 4, '6a': 5, '6b': 6}[layer]
                btn = self.labelsById[root.id]['btn']
                btn.setColor(pg.intColor(layer, 10))
                #root.setCheckState(0, QtCore.Qt.Checked)

            for i in range(root.childCount()):
                self.colorByLayer(root.child(i))
        finally:
            if unblock:
                self.blockSignals(False)
                self.labelsChanged.emit()

    def resetColors(self):
        try:
            self.blockSignals(True)
            for k,v in self.labelsById.items():
                v['btn'].setColor(pg.mkColor(v['btn'].defaultColor))
                #v['item'].setCheckState(0, QtCore.Qt.Unchecked)
        finally:
            self.blockSignals(False)
            self.labelsChanged.emit()

    def describe(self, id):
        if id not in self.labelsById:
            return "Unknown label: %d" % id
        descr = []
        item = self.labelsById[id]['item']
        name = str(item.text(1))
        while item is not self.labelsByAcronym['root']['item']:
            descr.insert(0, str(item.text(0)))
            item = item.parent()
        return ' > '.join(descr) + "  :  " + name


class VolumeSliceView(QtGui.QWidget):
    mouseHovered = QtCore.Signal(object)
    mouseClicked = QtCore.Signal(object)

    def __init__(self, parent=None):
        self.atlas = None
        self.label = None

        QtGui.QWidget.__init__(self, parent)
        self.scale = None
        self.resize(800, 800)
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0,0,0,0)

        self.w1 = pg.GraphicsLayoutWidget()
        self.w2 = pg.GraphicsLayoutWidget()
        self.view1 = self.w1.addViewBox()
        self.view2 = self.w2.addViewBox()
        self.view1.setAspectLocked()
        self.view2.setAspectLocked()
        self.view1.invertY(False)
        self.view2.invertY(False)
        self.layout.addWidget(self.w1, 0, 0)
        self.layout.addWidget(self.w2, 1, 0)

        self.img1 = LabelImageItem()
        self.img2 = LabelImageItem()
        self.img1.mouseHovered.connect(self.mouseHovered)
        self.img2.mouseHovered.connect(self.mouseHovered)
        self.img2.mouseClicked.connect(self.mouseClicked)
        self.view1.addItem(self.img1)
        self.view2.addItem(self.img2)

        self.target = Target()
        self.target.setZValue(5000)
        self.view2.addItem(self.target)
        self.target.setVisible(False)

        self.line_roi = RotateROI([0, 0], [.0020, .0020], maxBounds=QtCore.QRectF(-10, -10, 230, 140), pen=(0, 9), movable=False)
        self.line_roi.setSize(pg.Point(.0025, 0))
        self.view1.addItem(self.line_roi, ignoreBounds=True)
        self.line_roi.sigRegionChanged.connect(self.updateSlice)

        # TODO: Remove RulerROI of merge RotateROI into it
        # self.roi = RulerROI([[10, 64], [120, 64]], pen='r')
        # self.view1.addItem(self.roi, ignoreBounds=True)
        # self.roi.sigRegionChanged.connect(self.updateSlice)

        self.zslider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.zslider.valueChanged.connect(self.updateImage)
        self.layout.addWidget(self.zslider, 2, 0)

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.sliderRotation)
        self.layout.addWidget(self.slider, 3, 0)

        self.lut = pg.HistogramLUTWidget()
        self.lut.setImageItem(self.img1.atlasImg)
        self.lut.sigLookupTableChanged.connect(self.histlutChanged)
        self.lut.sigLevelsChanged.connect(self.histlutChanged)
        self.layout.addWidget(self.lut, 0, 1, 3, 1)

        self.clipboard = QtGui.QApplication.clipboard()

    def setData(self, atlas, label, scale=None):
        if np.isscalar(scale):
            scale = (scale, scale)
        self.atlas = atlas
        self.label = label
        if self.scale != scale:
            self.scale = scale

            # # reset ROI position # TODO: Reset using new ROI coordinates
            # with SignalBlock(self.roi.sigRegionChanged, self.updateSlice):
            #     h1, h2 = self.roi.getHandles()
            #     p1 = self.view1.mapViewToScene(pg.Point(0, 0))
            #     if scale is None:
            #         scale = (1, 1)
            #     p2 = self.view1.mapViewToScene(pg.Point(100*scale[0], 100*scale[1]))
            #     h1.movePoint([p1.x(), p1.y()])
            #     h2.movePoint([p2.x(), p2.y()])

        self.zslider.setMaximum(atlas.shape[0])
        self.zslider.setValue(atlas.shape[0] // 2)
        self.slider.setRange(-90, 90)  # TODO: need to define a better range
        self.slider.setValue(0)
        self.updateImage(autoRange=True)
        self.updateSlice()
        self.lut.setLevels(atlas.min(), atlas.max())

    def updateImage(self, autoRange=False):
        z = self.zslider.value()
        self.img1.setData(self.atlas[z], self.label[z], scale=self.scale)
        if autoRange:
            self.view1.autoRange(items=[self.img1.atlasImg])

    def setLabelLUT(self, lut):
        self.img1.setLUT(lut)
        self.img2.setLUT(lut)

    def updateSlice(self):
        rotation = self.slider.value()

        if self.atlas is None:
            return

        if rotation == 0:
            atlas, self.mappedCoords = self.line_roi.getArrayRegion(self.atlas, self.img1.atlasImg, axes=(1, 2), returnMappedCoords=True)
            label = self.line_roi.getArrayRegion(self.label, self.img1.atlasImg, axes=(1, 2), order=0)
        else:
            atlas, self.mappedCoords = self.line_roi.getArrayRegion(self.atlas, self.img1.atlasImg, rotation=rotation, axes=(1, 2, 0), returnMappedCoords=True)
            label = self.line_roi.getArrayRegion(self.label, self.img1.atlasImg, rotation=rotation, axes=(1, 2, 0), order=1)
            # label = self.line_roi.getArrayRegion(self.label, self.img1.atlasImg, rotation=rotation, axes=(1, 2, 0), order=0)

        if atlas.size == 0:
            return
        
        self.img2.setData(atlas, label, scale=self.scale)
        self.view2.autoRange(items=[self.img2.atlasImg])
        self.target.setVisible(False)
        self.w1.viewport().repaint()  # repaint immediately to avoid processing more mouse events before next repaint
        self.w2.viewport().repaint()
        
    def sliderRotation(self):
        rotation = self.slider.value()
        self.set_rotation_roi(self.img1.atlasImg, rotation)
        self.updateSlice()

    def set_rotation_roi(self, img, rotation):

        h1, h2, h3, h4, h5 = self.line_roi.getHandles()

        d_angle = pg.Point(h2.pos() - h1.pos())  # This gives the length in ccf coordinate size  # TODO: Document why using this.
        d = pg.Point(self.line_roi.mapToItem(img, h2.pos()) - self.line_roi.mapToItem(img, h1.pos()))

        origin_roi = self.line_roi.mapToItem(img, h1.pos())

        if rotation == 0:
            offset = 0
        else:
            offset = self.get_offset(rotation)
        
        # This calculates by how much the ROI needs to shift
        hyp = (offset * self.scale[0]) / (math.cos(math.radians(-(90 - d.angle(pg.Point(1, 0)))))) 
        opposite = (math.sin(math.radians(-(90 - d.angle(pg.Point(1, 0)))))) * hyp
        adjacent = opposite / (math.tan(math.radians(-(90 - d.angle(pg.Point(1, 0))))))
        
        # TODO: make sure offset is a correct depiction of the angle change
        self.line_roi.sigRegionChanged.disconnect(self.updateSlice)  # This is kind of a hack to avoid recursion error. Using update=False doesn't move the handles.
        self.line_roi.setSize(pg.Point(d_angle.length(), hyp * 2))
        self.line_roi.setPos(pg.Point((origin_roi.x() * self.scale[-1]) + adjacent, (origin_roi.y() * self.scale[-1]) + opposite))
        self.line_roi.sigRegionChanged.connect(self.updateSlice)

    # def rotatePoint(self, point, theta): # TODO: REmove
    #     theta = math.radians(theta)
    #     return (point[0] * math.cos(theta) - point[1] * math.sin(theta),
    #             point[0] * math.sin(theta) + point[1] * math.cos(theta))

    def get_offset(self, rotation):
        theta = math.radians(-rotation)

        # Figure out the unit vector with theta angle
        x, z = 0, 1
        dc, ds = math.cos(theta), math.sin(theta)
        xv = dc * x - ds * z
        zv = ds * x + dc * z

        # Figure out the slope of the unit vector
        m = zv / xv

        # y = mx + b
        # Calculate the x-intercept. using half the distance in the z-dimension as b. Since we want the axis of rotation in the middle
        offset = (-self.atlas.shape[0] / 2) / m

        return abs(offset)

    def closeEvent(self, ev):
        self.imv1.close()
        self.imv2.close()
        self.data = None

    def histlutChanged(self):
        # note: img1 is updated automatically; only bneed to update img2 to match
        self.img2.atlasImg.setLookupTable(self.lut.getLookupTable(n=256))
        self.img2.atlasImg.setLevels(self.lut.getLevels())

    def setOverlay(self, o):
        self.img1.setOverlay(o)
        self.img2.setOverlay(o)

    def setLabelOpacity(self, o):
        self.img1.setLabelOpacity(o)
        self.img2.setLabelOpacity(o)


class LabelImageItem(QtGui.QGraphicsItemGroup):
    class SignalProxy(QtCore.QObject):
        mouseHovered = QtCore.Signal(object)  # id
        mouseClicked = QtCore.Signal(object)  # id

    def __init__(self):
        self._sigprox = LabelImageItem.SignalProxy()
        self.mouseHovered = self._sigprox.mouseHovered
        self.mouseClicked = self._sigprox.mouseClicked

        QtGui.QGraphicsItemGroup.__init__(self)
        self.atlasImg = pg.ImageItem(levels=[0,1])
        self.labelImg = pg.ImageItem()
        self.atlasImg.setParentItem(self)
        self.labelImg.setParentItem(self)
        self.labelImg.setZValue(10)
        self.labelImg.setOpacity(0.5)
        self.setOverlay('Multiply')

        self.labelColors = {}
        self.setAcceptHoverEvents(True)

    def setData(self, atlas, label, scale=None):
        self.labelData = label
        self.atlasData = atlas
        if scale is not None:
            self.resetTransform()
            self.scale(*scale)
        self.atlasImg.setImage(self.atlasData, autoLevels=False)
        # self.labelImg.setImage(self.labelData, autoLevels=False)  
        self.labelImg.setImage(self.labelData, autoLevels=True)  # TODO: Changed this, make sure it is fine.

    def setLUT(self, lut):
        self.labelImg.setLookupTable(lut)

    def setOverlay(self, overlay):
        mode = getattr(QtGui.QPainter, 'CompositionMode_' + overlay)
        self.labelImg.setCompositionMode(mode)

    def setLabelOpacity(self, o):
        self.labelImg.setOpacity(o)

    def setLabelColors(self, colors):
        self.labelColors = colors

    def hoverEvent(self, event):
        if event.isExit():
            return

        try:
            id = self.labelData[int(event.pos().x()), int(event.pos().y())]
        except IndexError, AttributeError:
            return
        self.mouseHovered.emit(id)

    def mouseClickEvent(self, event):
        id = self.labelData[int(event.pos().x()), int(event.pos().y())]
        self.mouseClicked.emit([event, id])

    def boundingRect(self):
        return self.labelImg.boundingRect()

    def shape(self):
        return self.labelImg.shape()


class RotateROI(pg.ROI):
    def __init__(self, pos, size, **args):
        # QtGui.QGraphicsRectItem.__init__(self, pos[0], pos[1], size[0], size[1])
        pg.ROI.__init__(self, pos, size, **args)
        self.addRotateHandle([0, 0.5], [1, 1])
        self.addScaleRotateHandle([1, 0.5], [0.5, 0.5])
        self.addTranslateHandle([0.5, 0.5])
        self.addFreeHandle([0, 1], [0, 0])  
        self.addFreeHandle([0, 0], [0, 0])

    def paint(self, p, *args):
        pg.ROI.paint(self, p, *args)
        h1 = self.handles[0]['item'].pos()
        h2 = self.handles[1]['item'].pos()
        h4 = self.handles[3]['item'] 
        h5 = self.handles[4]['item'] 
        h4.setVisible(False)
        h5.setVisible(False)
        p1 = p.transform().map(h1)
        p2 = p.transform().map(h2)

        vec = pg.Point(h2) - pg.Point(h1)
        length = vec.length()
        angle = vec.angle(pg.Point(1, 0))

        pvec = p2 - p1
        pvecT = pg.Point(pvec.y(), -pvec.x())
        pos = 0.5 * (p1 + p2) + pvecT * 40 / pvecT.length()

        angle_two = pg.Point(pvec).angle(pg.Point(1, 0))  # TODO: this is wrong because pvec is pointing down, why?
        
        p.resetTransform()

        txt = pg.siFormat(length, suffix='m') + '\n%0.1f deg' % angle_two
        p.drawText(QtCore.QRectF(pos.x() - 50, pos.y() - 50, 100, 100), QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter, txt)

    def boundingRect(self):
        r = pg.ROI.boundingRect(self)
        pxw = 50 * self.pixelLength(pg.Point([1, 0]))
        return r.adjusted(-50, -50, 50, 50)

    def getArrayRegion(self, data, img, axes=(0, 1), order=1, returnMappedCoords=False, rotation=0, **kwds):
        """
        Use the position of this ROI relative to an imageItem to pull a slice
        from an array.

        Since this pulls 1D data from a 2D coordinate system, the return value
        will have ndim = data.ndim-1

        See ROI.getArrayRegion() for a description of the arguments.
        """

        imgPts = [self.mapToItem(img, h.pos()) for h in self.getHandles()]

        d = pg.Point(imgPts[1] - imgPts[0])
        o = pg.Point(imgPts[0])
       
        if rotation != 0:
            new_vector, new_v_length, vector2, origin_x, origin_y, origin_z = self.get_affine_slice_params(data, img, o, d, rotation)
            rgn = fn.affineSlice(data, shape=(int(new_v_length), int(d.length())),
                                 vectors=[new_vector, vector2],
                                 origin=(origin_x, origin_y, origin_z), axes=axes, order=order,
                                 returnCoords=returnMappedCoords, **kwds)

        else:
            rgn = fn.affineSlice(data, shape=(int(d.length()),), vectors=[pg.Point(d.norm())], origin=o, axes=axes,
                                 order=order, returnCoords=returnMappedCoords, **kwds)
        return rgn

    # Gets the vector and origin along the x,z axis  
    # TODO: Document the use of vector arithmetic on x-y plane when in reality is for x-z plane
    # TODO: Need to use shape dynamically to work when changing orientations.
    def get_affine_slice_params(self, data, img, start_pos, d, rotation):
        counter_clockwise = rotation < 0
        
        h1, h2, h3, h4, h5 = self.getHandles()
        origin_roi = self.mapToItem(img, h5.pos())
        left_corner = self.mapToItem(img, h4.pos())
        
        if counter_clockwise:
            origin = pg.Point(origin_roi.x(), origin_roi.y())
            end_point = pg.Point(left_corner.x(), data.shape[0])
            new_vector = end_point - origin
            diff_y = left_corner.y() - origin_roi.y() 
        else:
            origin = pg.Point(left_corner.x(), left_corner.y())  
            end_point = pg.Point(origin_roi.x(), data.shape[0])
            new_vector = end_point - left_corner
            diff_y = origin_roi.y() - origin.y()
 
        new_vector_length = math.sqrt((new_vector.x() * new_vector.x()) + (new_vector.y() * new_vector.y()) + (diff_y * diff_y))
        new_3d_vector = (new_vector.x() / new_vector_length, diff_y/new_vector_length, new_vector.y()/new_vector_length)
        
        return new_3d_vector, data.shape[0], (d.norm().x(), d.norm().y(), 0), origin.x(), origin.y(), 0

    # Gets the vector and origin along the x,z axis
    # This is no longer used. # TODO: Remove this
    # def get_affine_slice_params_V1(self, data, img, start_pos, d, rotation):
    #     counter_clockwise = rotation < 0
    # 
    #     # Figuring out Vector to use along the (x,z) axis
    #     # Figure out the angle of rotation in radians. Initial vector starts at (0,1)
    #     theta = math.radians(-rotation)
    # 
    #     # Figure out the unit vector with theta angle
    #     x, z = 0, 1
    #     dc, ds = math.cos(theta), math.sin(theta)
    #     xv = dc * x - ds * z
    #     zv = ds * x + dc * z
    # 
    #     # Figure out the slope of the unit vector
    #     m = zv / xv
    # 
    #     # y = mx + b
    #     # Calculate the x-intercept. using half the distance in the z-dimension as b. Since we want the axis of rotation in the middle
    #     offset = (-data.shape[0] / 2) / m
    # 
    #     b = (data.shape[0] / 2) - (m * (start_pos.x()))
    # 
    #     x_min = start_pos.x() - abs(offset)
    #     x_max = start_pos.x() + abs(offset)
    # 
    #     # Figure out start and end points
    #     if x_min < 0:
    #         # When out of bounds in the x-axis, use the z-axis 
    #         p1x = 0
    #         p1z = b  # This should be z at x=0
    #     else:
    #         p1x = x_min
    #         if counter_clockwise:
    #             p1z = data.shape[0]
    #         else:
    #             p1z = 0
    # 
    #     if x_max > data.shape[1]:
    #         # When out of bounds in the x-axis, use the z-axis
    #         p2x = data.shape[1]
    #         p2z = (m * data.shape[1]) + b
    #     else:
    #         p2x = x_max
    # 
    #         if counter_clockwise:
    #             p2z = 0
    #         else:
    #             p2z = data.shape[0]
    # 
    #     if counter_clockwise:
    #         new_vector = pg.Point(pg.Point(p1x, p1z) - pg.Point(p2x, p2z))
    #         origin_x = p2x
    #         origin_z = p2z
    #     else:
    #         new_vector = pg.Point(pg.Point(p2x, p2z) - pg.Point(p1x, p1z))
    #         origin_x = p1x
    #         origin_z = p1z
    # 
    #     axis = [0, 1, 0]
    #     v = [d.norm().x(), d.norm().y(), 0]
    # 
    #     # Option 2
    #     M0 = self.test(axis, theta)
    #     vector2 = dot(M0, v)
    # 
    #     return new_vector, vector2, origin_x, start_pos.y(), origin_z
    # 
    # def test(self, axis, theta):
    #     return expm3(cross(eye(3), axis / norm(axis) * theta))
    # 
    # def rotatePolygon(self, polygon, theta):
    #     theta = math.radians(theta)
    #     rotatedPolygon = []
    #     for corner in polygon:
    #         rotatedPolygon.append((corner[0] * math.cos(theta) - corner[1] * math.sin(theta),
    #                                corner[0] * math.sin(theta) + corner[1] * math.cos(theta)))
    #     return rotatedPolygon


class Target(pg.GraphicsObject):
    def __init__(self, movable=True):
        pg.GraphicsObject.__init__(self)
        self._bounds = None
        self.color = (255, 255, 0)

    def boundingRect(self):
        if self._bounds is None:
            # too slow!
            w = self.pixelLength(pg.Point(1, 0))
            if w is None:
                return QtCore.QRectF()
            h = self.pixelLength(pg.Point(0, 1))
            # o = self.mapToScene(QtCore.QPointF(0, 0))
            # w = abs(1.0 / (self.mapToScene(QtCore.QPointF(1, 0)) - o).x())
            # h = abs(1.0 / (self.mapToScene(QtCore.QPointF(0, 1)) - o).y())
            self._px = (w, h)
            w *= 21
            h *= 21
            self._bounds = QtCore.QRectF(-w, -h, w*2, h*2)
        return self._bounds

    def viewTransformChanged(self):
        self._bounds = None
        self.prepareGeometryChange()

    def paint(self, p, *args):
        p.setRenderHint(p.Antialiasing)
        px, py = self._px
        w = 4 * px
        h = 4 * py
        r = QtCore.QRectF(-w, -h, w*2, h*2)
        p.setPen(pg.mkPen(self.color))
        p.setBrush(pg.mkBrush(0, 0, 255, 100))
        p.drawEllipse(r)
        p.drawLine(pg.Point(-w*2, 0), pg.Point(w*2, 0))
        p.drawLine(pg.Point(0, -h*2), pg.Point(0, h*2))


class RulerROI(pg.LineSegmentROI):
    def paint(self, p, *args):
        pg.LineSegmentROI.paint(self, p, *args)
        h1 = self.handles[0]['item'].pos()
        h2 = self.handles[1]['item'].pos()
        p1 = p.transform().map(h1)
        p2 = p.transform().map(h2)

        vec = pg.Point(h2) - pg.Point(h1)
        length = vec.length()
        angle = vec.angle(pg.Point(1, 0))

        pvec = p2 - p1
        pvecT = pg.Point(pvec.y(), -pvec.x())
        pos = 0.5 * (p1 + p2) + pvecT * 40 / pvecT.length()

        p.resetTransform()

        txt = pg.siFormat(length, suffix='m') + '\n%0.1f deg' % angle
        p.drawText(QtCore.QRectF(pos.x()-50, pos.y()-50, 100, 100), QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter, txt)

    # # Gets the vector and origin along the x,z axis  # TODO: Remove functions below after confirming they're not needed
    # def get_affine_slice_params(self, data, start_pos, d, rotation):
    #     counter_clockwise = rotation < 0
    # 
    #     # Figuring out Vector to use along the (x,z) axis
    #     # Figure out the angle of rotation in radians. Initial vector starts at (0,1)
    #     theta = math.radians(-rotation)
    # 
    #     # Figure out the unit vector with theta angle
    #     x, z = 0, 1
    #     dc, ds = math.cos(theta), math.sin(theta)
    #     xv = dc * x - ds * z
    #     zv = ds * x + dc * z
    # 
    #     # Figure out the slope of the unit vector
    #     m = zv / xv
    # 
    #     # y = mx + b
    #     # Calculate the x-intercept. using half the distance in the z-dimension as b. Since we want the axis of rotation in the middle
    #     offset = (-data.shape[0] / 2) / m
    # 
    #     b = (data.shape[0] / 2) - (m * (start_pos.x()))
    # 
    #     x_min = start_pos.x() - abs(offset)
    #     x_max = start_pos.x() + abs(offset)
    # 
    #     # Figure out start and end points
    #     if x_min < 0:
    #         # When out of bounds in the x-axis, use the z-axis 
    #         p1x = 0
    #         p1z = b  # This should be z at x=0
    #     else:
    #         p1x = x_min
    #         if counter_clockwise:
    #             p1z = data.shape[0]
    #         else:
    #             p1z = 0
    # 
    #     if x_max > data.shape[1]:
    #         # When out of bounds in the x-axis, use the z-axis
    #         p2x = data.shape[1]
    #         p2z = (m * data.shape[1]) + b
    #     else:
    #         p2x = x_max
    # 
    #         if counter_clockwise:
    #             p2z = 0
    #         else:
    #             p2z = data.shape[0]
    # 
    #     if counter_clockwise:
    #         new_vector = pg.Point(pg.Point(p1x, p1z) - pg.Point(p2x, p2z))
    #         origin_x = p2x
    #         origin_z = p2z
    #     else:
    #         new_vector = pg.Point(pg.Point(p2x, p2z) - pg.Point(p1x, p1z))
    #         origin_x = p1x
    #         origin_z = p1z
    # 
    #     axis = [0, 1, 0]
    #     v = [d.norm().x(), d.norm().y(), 0]
    # 
    #     M0 = self.test(axis, theta)
    #     vector2 = dot(M0, v)
    # 
    #     return new_vector, vector2, origin_x, origin_z
    # 
    # def test(self, axis, theta):
    #     return expm3(cross(eye(3), axis / norm(axis) * theta))
    # 
    # def rotatePolygon(self, polygon, theta):
    #     theta = math.radians(theta)
    #     rotatedPolygon = []
    #     for corner in polygon:
    #         rotatedPolygon.append((corner[0] * math.cos(theta) - corner[1] * math.sin(theta),
    #                                corner[0] * math.sin(theta) + corner[1] * math.cos(theta)))
    #     return rotatedPolygon


def readNRRDAtlas(nrrdFile=None):
    """
    Download atlas files from:
      http://help.brain-map.org/display/mouseconnectivity/API#API-DownloadAtlas
    """
    import nrrd
    if nrrdFile is None:
        nrrdFile = QtGui.QFileDialog.getOpenFileName(None, "Select NRRD atlas file")

    with pg.BusyCursor():
        data, header = nrrd.read(nrrdFile)

    # convert to ubyte to compress a bit
    np.multiply(data, 255./data.max(), out=data, casting='unsafe')
    data = data.astype('ubyte')

    # data must have axes (anterior, dorsal, right)
    # rearrange axes to fit -- CCF data comes in (posterior, inferior, right) order.
    data = data[::-1, ::-1, :]

    # voxel size in um
    vxsize = 1e-6 * float(header['space directions'][0][0])

    info = [
        {'name': 'anterior', 'values': np.arange(data.shape[0]) * vxsize, 'units': 'm'},
        {'name': 'dorsal', 'values': np.arange(data.shape[1]) * vxsize, 'units': 'm'},
        {'name': 'right', 'values': np.arange(data.shape[2]) * vxsize, 'units': 'm'},
        {'vxsize': vxsize}
    ]
    ma = metaarray.MetaArray(data, info=info)
    return ma


def readNRRDLabels(nrrdFile=None, ontologyFile=None):
    """
    Download label files from:
      http://help.brain-map.org/display/mouseconnectivity/API#API-DownloadAtlas

    Download ontology files from:
      http://api.brain-map.org/api/v2/structure_graph_download/1.json

      see:
      http://help.brain-map.org/display/api/Downloading+an+Ontology%27s+Structure+Graph
      http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies#AtlasDrawingsandOntologies-StructuresAndOntologies

    This method compresses the annotation data down to a 16-bit array by remapping
    the larger annotations to smaller, unused values.
    """
    global onto, ontology, data, mapping, inds, vxsize, info, ma

    import nrrd
    if nrrdFile is None:
        nrrdFile = QtGui.QFileDialog.getOpenFileName(None, "Select NRRD annotation file")

    if ontologyFile is None:
        ontoFile = QtGui.QFileDialog.getOpenFileName(None, "Select ontology file (json)")

    with pg.ProgressDialog("Loading annotation file...", 0, 5, wait=0) as dlg:
        print "Loading annotation file..."
        app.processEvents()
        # Read ontology and convert to flat table
        onto = json.load(open(ontoFile, 'rb'))
        onto = parseOntology(onto['msg'][0])
        l1 = max([len(row[2]) for row in onto])
        l2 = max([len(row[3]) for row in onto])
        ontology = np.array(onto, dtype=[('id', 'int32'), ('parent', 'int32'), ('name', 'S%d'%l1), ('acronym', 'S%d'%l2), ('color', 'S6')])    

        if dlg.wasCanceled():
            return
        dlg += 1

        # read annotation data
        data, header = nrrd.read(nrrdFile)

        if dlg.wasCanceled():
            return
        dlg += 1

        # data must have axes (anterior, dorsal, right)
        # rearrange axes to fit -- CCF data comes in (posterior, inferior, right) order.
        data = data[::-1, ::-1, :]

        if dlg.wasCanceled():
            return
        dlg += 1

    # compress down to uint16
    print "Compressing.."
    u = np.unique(data)
    
    # decide on a 32-to-64-bit label mapping
    mask = u <= 2**16-1
    next_id = 2**16-1
    mapping = OrderedDict()
    inds = set()
    for i in u[mask]:
        mapping[i] = i
        inds.add(i)
   
    with pg.ProgressDialog("Remapping annotations to 16-bit...", 0, (~mask).sum(), wait=0) as dlg:
        app.processEvents()
        for i in u[~mask]:
            while next_id in inds:
                next_id -= 1
            mapping[i] = next_id
            inds.add(next_id)
            data[data == i] = next_id
            ontology['id'][ontology['id'] == i] = next_id
            ontology['parent'][ontology['parent'] == i] = next_id
            if dlg.wasCanceled():
                return
            dlg += 1
        
    data = data.astype('uint16')
    mapping = np.array(list(mapping.items()))    
 
    # voxel size in um
    vxsize = 1e-6 * float(header['space directions'][0][0])

    info = [
        {'name': 'anterior', 'values': np.arange(data.shape[0]) * vxsize, 'units': 'm'},
        {'name': 'dorsal', 'values': np.arange(data.shape[1]) * vxsize, 'units': 'm'},
        {'name': 'right', 'values': np.arange(data.shape[2]) * vxsize, 'units': 'm'},
        {'vxsize': vxsize, 'ai_ontology_map': mapping, 'ontology': ontology}
    ]
    ma = metaarray.MetaArray(data, info=info)
    return ma


def parseOntology(root, parent=-1):
    ont = [(root['id'], parent, root['name'], root['acronym'], root['color_hex_triplet'])]
    for child in root['children']:
        ont += parseOntology(child, root['id'])
    return ont


def writeFile(data, file):
    dataDir = os.path.dirname(file)
    if dataDir != '' and not os.path.exists(dataDir):
        os.makedirs(dataDir)

    if max(data.shape) > 200 and min(data.shape) > 200:
        data.write(file, chunks=(200, 200, 200))
    else:
        data.write(file)


####### Stolen from ACQ4; not in mainline pyqtgraph yet #########

def disconnect(signal, slot):
    """Disconnect a Qt signal from a slot.

    This method augments Qt's Signal.disconnect():

    * Return bool indicating whether disconnection was successful, rather than
      raising an exception
    * Attempt to disconnect prior versions of the slot when using pg.reload    
    """
    while True:
        try:
            signal.disconnect(slot)
            return True
        except TypeError, RuntimeError:
            slot = getPreviousVersion(slot)
            if slot is None:
                return False

class SignalBlock(object):
    """Class used to temporarily block a Qt signal connection::

        with SignalBlock(signal, slot):
            # do something that emits a signal; it will
            # not be delivered to slot
    """
    def __init__(self, signal, slot):
        self.signal = signal
        self.slot = slot

    def __enter__(self):
        disconnect(self.signal, self.slot)
        return self

    def __exit__(self, *args):
        self.signal.connect(self.slot)


if __name__ == '__main__':

    app = pg.mkQApp()

    v = AtlasViewer()
    v.show()

    path = os.path.dirname(os.path.realpath(__file__))
    atlasFile = os.path.join(path, "ccf.ma")
    labelFile = os.path.join(path, "ccf_label.ma")

    if os.path.isfile(atlasFile):
        atlas = metaarray.MetaArray(file=atlasFile, readAllData=True)
    else:
        try:
            atlas = readNRRDAtlas()
            writeFile(atlas, atlasFile)
        except:
            try:
                print "Unexpected error when creating ccf.ma file with " + atlasFile
                print traceback.print_exc()
                raise
            finally:
                print "Removing ccf.ma"
                if os.path.isfile(atlasFile):
                    os.remove(atlasFile)

    if os.path.isfile(labelFile):
        label = metaarray.MetaArray(file=labelFile, readAllData=True)
    else:
        try:
            label = readNRRDLabels()
            writeFile(label, labelFile)
        except:
            try:
                print "Unexpected error when creating ccf_label.ma file with " + labelFile
                print traceback.print_exc()
                raise
            finally:
                print "Removing ccf.ma and ccf_label.ma files..."
                if os.path.isfile(atlasFile):
                    os.remove(atlasFile)
                if os.path.isfile(labelFile):
                    os.remove(labelFile)

    v.setAtlas(atlas)
    v.setLabels(label)

    if sys.flags.interactive == 0:
        app.exec_()
