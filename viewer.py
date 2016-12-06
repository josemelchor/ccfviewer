import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import json
from collections import OrderedDict
import numpy as np
import pyqtgraph as pg
import pyqtgraph.functions as fn
import pyqtgraph.metaarray as metaarray
from pyqtgraph.Qt import QtGui, QtCore
import math
import points_to_aff


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
        self.view.targetReleased.connect(self.targetReleased)
        self.view.copyToClipboard.connect(self.copySubmitted)
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
        
        self.coordinateCtrl = CoordinatesCtrl(self)
        self.coordinateCtrl.coordinateSubmitted.connect(self.coordinateSubmitted)
        self.coordinateCtrl.displayLabelsSubmitted.connect(self.displayLabelsSubmitted)
        self.coordinateCtrl.toggleSubmitted.connect(self.toggleSubmitted)
        self.coordinateCtrl.copySubmitted.connect(self.copySubmitted)
        self.coordinateCtrl.resetSubmitted.connect(self.resetSubmitted)
        self.ctrlLayout.addWidget(self.coordinateCtrl)

    def setLabels(self, label):
        self.label = label
        with SignalBlock(self.labelTree.labelsChanged, self.labelsChanged):
            for rec in label._info[-1]['ontology']:
                self.labelTree.addLabel(*rec)
        self.updateImage()
        self.labelsChanged()

    def setAtlas(self, atlas):
        self.atlas = atlas
        self.coordinateCtrl.atlas_shape = atlas.shape
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
     
    def targetReleased(self, target): 
        point_label = self.getCcfPoint(target)
        self.pointLabel.setText(point_label)
        self.copySubmitted()

    # Get CCF point coordinate and Structure id
    # Returns one string used for display a label and structure id
    # PIR orientation where x axis = Anterior-to-Posterior, y axis = Superior-to-Inferior and z axis = Left-to-Right
    def getCcfPoint(self, target): 

        str_id = self.view.img2.labelData[int(target.pos()[0]), int(target.pos()[1])]
        axis = self.displayCtrl.params['Orientation']

        # find real lims id
        lims_str_id = (key for key, value in self.label._info[-1]['ai_ontology_map'] if value == str_id).next()
        
        M0, M0i = self.get_affine_matrices()

        # Find what the mouse point position is relative to the coordinate
        target_pos = self.get_point_in_CCF(target, M0i)

        if axis == 'right':
            point = "x: " + str(target_pos[0]) + " y: " + str(target_pos[1]) + " z: " + str(target_pos[2]) + " StructureID: " + str(lims_str_id)
        elif axis == 'anterior':
            point = "x: " + str(target_pos[2]) + " y: " + str(target_pos[1]) + " z: " + str(target_pos[0]) + " StructureID: " + str(lims_str_id)
        elif axis == 'dorsal':
            point = "x: " + str(target_pos[1]) + " y: " + str(target_pos[2]) + " z: " + str(target_pos[0]) + " StructureID: " + str(lims_str_id)
        else:
            point = 'N/A'

        return point
    
    def get_affine_matrices(self):
        # compute the 4x4 transform matrix
        a = self.scale_point_to_CCF(self.view.line_roi.origin)
        ab = self.scale_vector_to_PIR(self.view.line_roi.ab_vector)
        ac = self.scale_vector_to_PIR(self.view.line_roi.ac_vector)
        
        return points_to_aff.points_to_aff(a, np.array(ab), np.array(ac))
    
    # Find what the mouse point position is relative to the coordinate
    def get_point_in_CCF(self, target_cell, M0i):
        ab_length = np.linalg.norm(self.view.line_roi.ab_vector)
        ac_length = np.linalg.norm(self.view.line_roi.ac_vector)        
        p = (target_cell.pos()[0]/ac_length, target_cell.pos()[1]/ab_length)
        
        ccf_location = np.dot(M0i, [p[1], p[0], 0, 1]) # use the inverse transform matrix and the mouse point
        
        # These should be x, y, z
        return float(ccf_location[0]), float(ccf_location[1]), float(ccf_location[2])
    
    def scale_point_to_CCF(self, point):
        """
        Returns a tuple (x, y, z) scaled from Item coordinates to CCF coordinates
        
        Point is a tuple with values x, y, z (ordered) 
        """
        vxsize = self.atlas._info[-1]['vxsize'] * 1e6
        p_to_ccf = ((self.view.atlas.shape[1] - point[0]) * vxsize,
                    (self.view.atlas.shape[2] - point[1]) * vxsize,
                    (self.view.atlas.shape[0] - point[2]) * vxsize)
        return p_to_ccf
    
    def scale_vector_to_PIR(self, vector):
        """
        Returns a list representing a vector. The new vector is scaled to CCF coordinate size. Also orients the vector to PIR orientaion.
        
        Vector must me specified as a list
        """
        p_to_ccf = []
        for p in vector:
            p_to_ccf.append(-(p * self.atlas._info[-1]['vxsize'] * 1e6))  # Need to use negative since using PIR orientation
        return p_to_ccf

    def ccf_point_to_view(self, pos):
        """
        This function translates a ccf's position to the view's coordinates.
                
        The pos is a tuple with values x, y, z
        """
        vxsize = self.atlas._info[-1]['vxsize'] * 1e6
        return ((self.view.atlas.shape[1] - (pos[0] / vxsize)) * self.view.scale[0],
                (self.view.atlas.shape[2] - (pos[1] / vxsize)) * self.view.scale[1],
                (self.view.atlas.shape[0] - (pos[2] / vxsize)) * self.view.scale[1])

    def vector_to_view(self, vector):
        """
        Scales vector to view coordinate size. vector is a tuple with x, y, z (in that order)   
        """
        vxsize = self.atlas._info[-1]['vxsize'] * 1e6
        new_point = ((vector[0] / vxsize) * self.view.scale[0],
                     (vector[1] / vxsize) * self.view.scale[1],
                     (vector[2] / vxsize) * self.view.scale[1])  
        return new_point
  
    def coordinateSubmitted(self):
        try:
            if self.displayCtrl.params['Orientation'] != "right":
                displayError('Set Coordinate function is only supported with Right orientation')
                return
            
            vxsize = self.atlas._info[-1]['vxsize'] * 1e6
            location = json.loads(str(self.coordinateCtrl.line.text()))
            
            self.coordinateCtrl.set_location(location)
            self.coordinateCtrl.set_cell_tree()
            
            if "transform" in location["slice_specimen"].keys() and location["slice_specimen"]["transform"]:
                transform = location["slice_specimen"]["transform"]
                # Get transform matrix
                M1, M1i = points_to_aff.lims_obj_to_aff(transform)
                # Get origin, and vectors in ccf coordinates
                ccf_origin, ccf_ab_vector, ccf_ac_vector = points_to_aff.aff_to_origin_and_vectors(M1i)
                # Get origin and vectors back in view coordinates
                roi_origin = np.array(self.ccf_point_to_view(ccf_origin))
                ab_vector = -np.array(self.vector_to_view(ccf_ab_vector))
                ac_vector = -np.array(self.vector_to_view(ccf_ac_vector))

                to_ac_angle = self.view.line_roi.get_ac_angle(ac_vector)

                # Where the origin of the ROI should be
                if to_ac_angle > 0:
                    roi_origin = ac_vector + roi_origin

                to_size = self.view.line_roi.get_roi_size(ab_vector, ac_vector)
                to_ab_angle = self.view.line_roi.get_ab_angle(ab_vector)
            else:
                transform = None
                if not location["slice_specimen"]["cells"]:
                    self.view.clear_previous_cells()
                    return

            if location["slice_specimen"]["cells"]:
                
                self.view.set_cell_cluster(location["slice_specimen"]["cells"])
                
                if transform:
                    self.view.set_cell_cluster_positions(transform=transform, vxsize=vxsize)
                else:
                    # No transform given, place cell ignoring lineROI angles
                    self.view.set_cell_cluster_positions(vxsize=vxsize)
                    roi_origin = ((self.view.atlas.shape[1]/2) * self.view.scale[1], 0.0)
                    to_size = (self.view.atlas.shape[2] * self.view.scale[1], 0.0) 
                    to_ab_angle = 90
                    to_ac_angle = 0

                self.setPlane(pg.Point(roi_origin[0], roi_origin[1]), pg.Point(to_size), to_ab_angle, to_ac_angle)
                return

            else:
                # Transform present but no cells defined, just set plane
                self.setPlane(pg.Point(roi_origin[0], roi_origin[1]), pg.Point(to_size), to_ab_angle, to_ac_angle)
    
        except:
            displayError('Error: %s Check value: %s' % (sys.exc_info()[0], sys.exc_info()[1]))

    def setPlane(self, pos, size, ab_angle, ac_angle):
        self.view.slider.setValue(int(float(str(ac_angle))))  # NOTE: This is ridiculous, casting ac_angle to an int doesn't work unless I do this
        self.view.line_roi.setAngle(ab_angle)
        self.view.line_roi.setPos(pos)
        self.view.line_roi.setSize(size)

    def get_target_position(self, ccf_location, M, ab_vector, ac_vector, vxsize):
        """
        Use affine transform matrix M to map ccf coordinate back to original coordinates  
        """
        img_location = np.dot(M, ccf_location)
        
        p1 = (np.linalg.norm(ac_vector) / vxsize * img_location[1]) * self.view.scale[0]
        p2 = (np.linalg.norm(ab_vector) / vxsize * img_location[0]) * self.view.scale[0]
        
        return p1, p2

    def displayLabelsSubmitted(self):
        """
        Display target labels  
        """
        self.view.display_Labels()
        self.view.w2.viewport().repaint()
        
    def copySubmitted(self):
        # Set coordinateCtrl location
        transform = self.get_plane_transform()   
        self.coordinateCtrl.set_transform(transform)
        
        # Set cell locations
        self.set_cells()
        
        # Copy location to clipboard
        self.view.clipboard.setText(json.dumps(self.coordinateCtrl.location))
     
    # Saves current state of the targets (aka cells) in self.coordinateCtrl.location
    # PIR orientation where x axis = Anterior-to-Posterior, y axis = Superior-to-Inferior and z axis = Left-to-Right   
    def set_cells(self):
        if self.view.cell_cluster:
                
            M0, M0i = self.get_affine_matrices()
            
            for cell in self.view.cell_cluster:
        
                str_id = self.view.img2.labelData[int(cell.pos()[0]), int(cell.pos()[1])]
                axis = self.displayCtrl.params['Orientation']
        
                # find real lims id
                lims_str_id = (key for key, value in self.label._info[-1]['ai_ontology_map'] if value == str_id).next()
        
                # Find what the mouse point position is relative to the coordinate
                target_pos = self.get_point_in_CCF(cell, M0i)
        
                if axis == 'right':
                    self.coordinateCtrl.location["slice_specimen"]['cells'][cell.lims_id] = {"name": str(cell.label), "ccf_coordinate": [target_pos[0], target_pos[1], target_pos[2]], "structure_id": str(lims_str_id)} 
                elif axis == 'anterior':
                    self.coordinateCtrl.location["slice_specimen"]['cells'][cell.lims_id] = {"name": str(cell.label), "ccf_coordinate": [target_pos[2], target_pos[1], target_pos[0]], "structure_id": str(lims_str_id)} 
                elif axis == 'dorsal':
                    self.coordinateCtrl.location["slice_specimen"]['cells'][cell.lims_id] = {"name": str(cell.label), "ccf_coordinate": [target_pos[1], target_pos[2], target_pos[0]], "structure_id": str(lims_str_id)} 
                else:
                    print 'Error: unknown orientation' 
        
    def resetSubmitted(self):
        self.view.clear_previous_cells()
    
    def get_plane_transform(self):
        M0, M0i = self.get_affine_matrices()
        # Convert matrix transform to a LIMS dictionary
        ob = points_to_aff.aff_to_lims_obj(M0, M0i)
        return ob
    
    def toggleSubmitted(self, item):
        if item.type() == 5:
            self.view.hide_all_cells(item)
        else:
            self.view.hide_cell(item)


class CoordinatesCtrl(QtGui.QWidget):
    coordinateSubmitted = QtCore.Signal()
    displayLabelsSubmitted = QtCore.Signal()
    copySubmitted = QtCore.Signal()
    resetSubmitted = QtCore.Signal()
    toggleSubmitted = QtCore.Signal(object)
    
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.cell_tree = QtGui.QTreeWidget(self)
        self.layout.addWidget(self.cell_tree, 0, 0, 1, 4)
        self.cell_tree.header().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.cell_tree.headerItem().setText(0, "Specimen Name")
        self.checked = set()
        self.cell_tree.itemChanged.connect(self.itemChange)
        
        self.line = QtGui.QLineEdit(self)
        self.line.returnPressed.connect(self.set_coordinate)
        self.layout.addWidget(self.line, 1, 0, 1, 4)

        self.btn = QtGui.QPushButton('Set', self)
        self.layout.addWidget(self.btn, 2, 0)
        self.btn.clicked.connect(self.set_coordinate)
        
        self.labels_btn = QtGui.QPushButton('Labels', self)
        self.layout.addWidget(self.labels_btn, 2, 1)
        self.labels_btn.clicked.connect(self.display_labels)
        
        self.copy_btn = QtGui.QPushButton('Copy', self)
        self.layout.addWidget(self.copy_btn, 2, 2)
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        
        self.reset_btn = QtGui.QPushButton('Reset', self)
        self.layout.addWidget(self.reset_btn, 2, 3)
        self.reset_btn.clicked.connect(self.reset_cell_panel)
        
        self.location = {"slice_specimen": {"lims_id": None,
                                            "name": None,
                                            "transform": {},
                                            "cells": {}}}

    def itemChange(self, item):
        self.toggleSubmitted.emit(item)
        
    def reset_cell_panel(self):
        self.cell_tree.clear()
        self.location = {"slice_specimen": {"lims_id": None,
                                            "name": None,
                                            "transform": {},
                                            "cells": {}}}
        self.line.clear()
        self.resetSubmitted.emit()

    def set_cell_tree(self):
        if not self.location['slice_specimen']['name']:  # This would occur when setting plane only...
            return

        sp_slice = QtGui.QTreeWidgetItem([self.location['slice_specimen']['name']], 5) # Setting type=5 to identify root specimen (aka 'slice specimen')
        sp_slice.setFlags(sp_slice.flags() | QtCore.Qt.ItemIsUserCheckable)
        sp_slice.setCheckState(0, QtCore.Qt.Checked)
        self.cell_tree.clear()
        self.cell_tree.addTopLevelItem(sp_slice)
        cell_list = []
        
        for key, cell in self.location['slice_specimen']['cells'].items():
            cell_list.append(cell['name'])
            
        for cell in sorted(cell_list):
            c = QtGui.QTreeWidgetItem([cell])
            c.setFlags(c.flags() | QtCore.Qt.ItemIsUserCheckable)
            c.setCheckState(0, QtCore.Qt.Checked)
            sp_slice.addChild(c)

    def set_coordinate(self):
        try:
            errors = self.validate_location()
            if not errors:
                self.coordinateSubmitted.emit()
            else:
                displayError(errors)
        except ValueError:
            displayError("Error Setting coordinate")
    
    def set_location(self, place):
        self.location["slice_specimen"]["lims_id"] = place["slice_specimen"]["lims_id"]
        self.location["slice_specimen"]["name"] = place["slice_specimen"]["name"]
        self.set_transform(place["slice_specimen"]["transform"])    
        self.set_cells(place["slice_specimen"]["cells"])    
        
    def set_transform(self, transform):
        self.location["slice_specimen"]["transform"] = transform
        # print self.location
        
    def set_cells(self, cells):
        if cells:
            self.location["slice_specimen"]["cells"] = cells
        else:
            self.location["slice_specimen"]["cells"] = {} 
            
    def validate_location(self):
        # locations = self.line.text()
        json_string = str(self.line.text())
        if not json_string:
            return "No coordinate provided"
        
        location = json.loads(str(self.line.text()))
        
        for key, value in location['slice_specimen']['cells'].items():    
            coords = value['ccf_coordinate']
            
            if len(coords) < 3:
                return "Coordinate is malformed"
            else:
                errors = self.target_within_range(float(coords[0]), float(coords[1]), float(coords[2]))
                
            return errors
    
    def target_within_range(self, x, y, z):

        vxsize = atlas._info[-1]['vxsize'] * 1e6
        error = ""
        if z > (self.atlas_shape[2] * vxsize) or z < 0:
            error += "z coordinate {} is not within CCF range".format(z)
        if x > self.atlas_shape[0] * vxsize or x < 0:
            error += " x coordinate {} is not within CCF range".format(x)
        if y > self.atlas_shape[1] * vxsize or y < 0:
            error += " y coordinate {} is not within CCF range".format(y)
        
        return error
    
    def display_labels(self):
        """
        Display labels of targets (representing cells) 
        """
        self.displayLabelsSubmitted.emit()
        
    def copy_to_clipboard(self):
        self.copySubmitted.emit()
        
        
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
    sigDragged = QtCore.Signal(object)
    targetReleased = QtCore.Signal(object)
    copyToClipboard = QtCore.Signal()

    def __init__(self, parent=None):
        self.atlas = None
        self.label = None
        self.cell_cluster = None

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
        self.view1.addItem(self.img1)
        self.view2.addItem(self.img2)

        self.line_roi = RulerROI([.005, 0], [.008, 0], angle=90, pen=(0, 9), movable=False)
        self.view1.addItem(self.line_roi, ignoreBounds=True)
        self.line_roi.sigRegionChanged.connect(self.updateSlice)

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
        
        QtGui.QShortcut(QtGui.QKeySequence("Alt+Up"), self, self.slider_up)
        QtGui.QShortcut(QtGui.QKeySequence("Alt+Down"), self, self.slider_down)
        QtGui.QShortcut(QtGui.QKeySequence("Alt+Left"), self, self.tilt_left)
        QtGui.QShortcut(QtGui.QKeySequence("Alt+Right"), self, self.tilt_right)
        QtGui.QShortcut(QtGui.QKeySequence("Alt+1"), self, self.move_left)
        QtGui.QShortcut(QtGui.QKeySequence("Alt+2"), self, self.move_right)

    def slider_up(self):
        self.slider.triggerAction(QtGui.QAbstractSlider.SliderSingleStepAdd)
        
    def slider_down(self):
        self.slider.triggerAction(QtGui.QAbstractSlider.SliderSingleStepSub)
        
    def tilt_left(self):
        self.line_roi.rotate(1)
        
    def tilt_right(self):
        self.line_roi.rotate(-1)
        
    def move_right(self):
        # print '-- Pos'
        # print self.line_roi.pos()
        self.line_roi.setPos((self.line_roi.pos().x() + .0001, self.line_roi.pos().y()))
        
    def move_left(self):
        # print '-- Pos'
        # print self.line_roi.pos()
        self.line_roi.setPos((self.line_roi.pos().x() - .0001, self.line_roi.pos().y()))
        
    def set_cell_cluster(self, cells):
        self.clear_previous_cells()
        for key, value in cells.items():
            
            t = Target()
            t.lims_id = key
            t.setLabel(value['name'])
            t.setCoordinate(value['ccf_coordinate'])
            t.setZValue(5000)
            t.sigDragged.connect(self.targetDragged)
            self.img2.addToGroup(t)
            self.cell_cluster.append(t)
            
    def clear_previous_cells(self):
        if self.cell_cluster:
            # Remove any previous cells
            for item in self.cell_cluster:
                self.view2.removeItem(item)

            del self.cell_cluster[:]  
        
        self.cell_cluster = []

    def targetDragged(self, item):
        self.w2.viewport().repaint()
        self.targetReleased.emit(item)
    
    def set_cell_cluster_positions(self, transform=None, vxsize=None):
        if not self.cell_cluster:
            return
        
        if transform:
            # Get transform matrix
            M1, M1i = points_to_aff.lims_obj_to_aff(transform)
            # Get origin, and vectors in ccf coordinates
            ccf_origin, ccf_ab_vector, ccf_ac_vector = points_to_aff.aff_to_origin_and_vectors(M1i)
            
            for cell in self.cell_cluster:
                coords = cell.ccf_coordinate + [1]  # Need to add a 1 for matrix to work
                pos = self.get_target_position(coords, M1, ccf_ab_vector, ccf_ac_vector, vxsize)
                cell.setPos(pos[0], pos[1])  

        else:
            # No transform given
            for cell in self.cell_cluster:
                translated_x = (self.atlas.shape[1] - (cell.ccf_coordinate[0]/vxsize))# * self.scale[0] 
                translated_y = (self.atlas.shape[2] - (cell.ccf_coordinate[1]/vxsize))# * self.scale[0] 
                translated_z = (self.atlas.shape[0] - (cell.ccf_coordinate[2]/vxsize))# * self.scale[0]  
                cell.setPos(translated_z, translated_y)  # This will only work when using 'right' orientation

    def get_target_position(self, ccf_location, M, ab_vector, ac_vector, vxsize):
        """
        Use affine transform matrix M to map ccf coordinate back to original coordinates  
        """
        img_location = np.dot(M, ccf_location)
        
        p1 = (np.linalg.norm(ac_vector) / vxsize * img_location[1])
        p2 = (np.linalg.norm(ab_vector) / vxsize * img_location[0])
        
        return p1, p2
    
    def display_Labels(self):
        if not self.cell_cluster:
            return
        
        for cell in self.cell_cluster:
            cell.showLabel = not cell.showLabel
    
    def hide_all_cells(self, item): 
        count = item.childCount()

        for i in range(count):
            cell = item.child(i)
            cell.setCheckState(0, item.checkState(0))
            self.hide_cell(cell)

    def hide_cell(self, item):

        if not self.cell_cluster:
            return
        
        for cell in self.cell_cluster:
            if item.text(0) == cell.label:
                cell.setVisible(item.checkState(0) == QtCore.Qt.Checked)
                self.w2.viewport().repaint()

    def setData(self, atlas, label, scale=None):
        if np.isscalar(scale):
            scale = (scale, scale)
        self.atlas = atlas
        self.label = label
        if self.scale != scale:
            self.scale = scale

        self.zslider.setMaximum(atlas.shape[0])
        self.zslider.setValue(atlas.shape[0] // 2)
        self.slider.setRange(-45, 45)
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
            atlas = self.line_roi.getArrayRegion(self.atlas, self.img1.atlasImg, axes=(1, 2))
            label = self.line_roi.getArrayRegion(self.label, self.img1.atlasImg, axes=(1, 2), order=0)
        else:
            atlas = self.line_roi.getArrayRegion(self.atlas, self.img1.atlasImg, rotation=rotation, axes=(1, 2, 0))
            label = self.line_roi.getArrayRegion(self.label, self.img1.atlasImg, rotation=rotation, axes=(1, 2, 0), order=0)

        if atlas.size == 0:
            return
        
        self.img2.setData(atlas, label, scale=self.scale)
        self.view2.autoRange(items=[self.img2.atlasImg])
        self.w1.viewport().repaint() # repaint immediately to avoid processing more mouse events before next repaint
        self.w2.viewport().repaint()
        self.copyToClipboard.emit()
        
    def sliderRotation(self):
        rotation = self.slider.value()
        self.set_rotation_roi(self.img1.atlasImg, rotation)
        self.updateSlice()

    def set_rotation_roi(self, img, rotation):

        h1, h2, h3, h4, h5 = self.line_roi.getHandles()

        d_angle = pg.Point(h2.pos() - h1.pos())  # This gives the length in ccf coordinate size 
        d = pg.Point(self.line_roi.mapToItem(img, h2.pos()) - self.line_roi.mapToItem(img, h1.pos()))

        origin_roi = self.line_roi.mapToItem(img, h1.pos())

        if rotation == 0:
            offset = 0
        else:
            offset = self.get_offset(rotation)
        
        # This calculates by how much the ROI needs to shift
        if d.angle(pg.Point(1, 0)) == 90.0:
            # when ROI is on a 90 degree angle, can't really calculate using a right-angle triangle, ugh
            hyp, opposite, adjacent = offset * self.scale[0], 0, offset * self.scale[0]
        else:
            hyp = (offset * self.scale[0])  # / (math.cos(math.radians(-(90 - d.angle(pg.Point(1, 0)))))) 
            opposite = (math.sin(math.radians(-(90 - d.angle(pg.Point(1, 0)))))) * hyp
            adjacent = opposite / (math.tan(math.radians(-(90 - d.angle(pg.Point(1, 0))))))
        
        # This is kind of a hack to avoid recursion error. Using update=False doesn't move the handles.
        self.line_roi.sigRegionChanged.disconnect(self.updateSlice)  
        # increase size to denote rotation
        self.line_roi.setSize(pg.Point(d_angle.length(), hyp * 2))
        # Shift position in order to keep the cutting axis in the middle
        self.line_roi.setPos(pg.Point((origin_roi.x() * self.scale[-1]) + adjacent, (origin_roi.y() * self.scale[-1]) + opposite))
        self.line_roi.sigRegionChanged.connect(self.updateSlice)

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

    def __init__(self):
        self._sigprox = LabelImageItem.SignalProxy()
        self.mouseHovered = self._sigprox.mouseHovered

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
        self.labelImg.setImage(self.labelData, autoLevels=False)  

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

    def boundingRect(self):
        return self.labelImg.boundingRect()

    def shape(self):
        return self.labelImg.shape()


class RulerROI(pg.ROI):
    """
    ROI subclass with one rotate handle, one scale-rotate handle and one translate handle. Rotate handles handles define a line. 
    
    ============== =============================================================
    **Arguments**
    positions      (list of two length-2 sequences) 
    \**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    """
    
    def __init__(self, pos, size, **args):
        pg.ROI.__init__(self, pos, size, **args)
        self.ab_vector = (0, 0, 0)  # This is the vector pointing up/down from the origin
        self.ac_vector = (0, 0, 0)  # This is the vector pointing across form the orign
        self.origin = (0, 0, 0)     
        self.ab_angle = 90  # angle on the ab_vector
        self.ac_angle = 0   # angle of the ac_vector 
        self.addRotateHandle([0, 0.5], [1, 1])
        self.addScaleRotateHandle([1, 0.5], [0.5, 0.5])
        self.addTranslateHandle([0.5, 0.5])
        self.addFreeHandle([0, 1], [0, 0])  
        self.addFreeHandle([0, 0], [0, 0])
        self.newRoi = pg.ROI((0, 0), [1, 5], parent=self, pen=pg.mkPen('w', style=QtCore.Qt.DotLine))

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

        pvec = p2 - p1
        pvecT = pg.Point(pvec.y(), -pvec.x())
        pos = 0.5 * (p1 + p2) + pvecT * 40 / pvecT.length()

        angle = pg.Point(1, 0).angle(pg.Point(pvec)) 
        self.ab_angle = angle
        
        # Overlay a line to signal which side of the ROI is the back.
        if self.ac_angle > 0:
            self.newRoi.setVisible(True)
            self.newRoi.setPos(h5.pos())
        elif self.ac_angle < 0:
            self.newRoi.setVisible(True)
            self.newRoi.setPos(h4.pos())
        else:
            self.newRoi.setVisible(False)
            
        self.newRoi.setSize(pg.Point(self.size()[0], 0))
        
        p.resetTransform()

        txt = pg.siFormat(length, suffix='m') + '\n%0.1f deg' % angle + '\n%0.1f deg' % self.ac_angle
        p.drawText(QtCore.QRectF(pos.x() - 50, pos.y() - 50, 100, 100), QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter, txt)

    def boundingRect(self):
        r = pg.ROI.boundingRect(self)
        pxw = 50 * self.pixelLength(pg.Point([1, 0]))
        return r.adjusted(-50, -50, 50, 50)

    def getArrayRegion(self, data, img, axes=(0, 1), order=1, rotation=0, **kwds):

        imgPts = [self.mapToItem(img, h.pos()) for h in self.getHandles()]

        d = pg.Point(imgPts[1] - imgPts[0]) # This is the xy direction vector
        o = pg.Point(imgPts[0])
       
        if rotation != 0:
            ac_vector, ac_vector_length, origin = self.get_affine_slice_params(data, img, rotation)
            rgn = fn.affineSlice(data, shape=(int(ac_vector_length), int(d.length())), vectors=[ac_vector, (d.norm().x(), d.norm().y(), 0)],
                                 origin=origin, axes=axes, order=order, **kwds) 
            
            # Save vector and origin
            self.origin = origin
            self.ac_vector = ac_vector * ac_vector_length
        else:
            rgn = fn.affineSlice(data, shape=(int(d.length()),), vectors=[pg.Point(d.norm())], origin=o, axes=axes, order=order, **kwds)
            # Save vector and origin
            self.ac_vector = (0, 0, data.shape[0])
            self.origin = (o.x(), o.y(), 0.0) 
        
        # save this as well
        self.ab_vector = (d.x(), d.y(), 0)
        self.ac_angle = rotation
        
        return rgn

    def get_affine_slice_params(self, data, img, rotation):
        """
        Use the position of this ROI handles to get a new vector for the slice view's x-z direction.
        """
        counter_clockwise = rotation < 0
        
        h1, h2, h3, h4, h5 = self.getHandles()
        origin_roi = self.mapToItem(img, h5.pos())
        left_corner = self.mapToItem(img, h4.pos())
        
        if counter_clockwise:
            origin = np.array([origin_roi.x(), origin_roi.y(), 0])
            end_point = np.array([left_corner.x(), left_corner.y(), data.shape[0]])
        else:
            origin = np.array([left_corner.x(), left_corner.y(), 0])
            end_point = np.array([origin_roi.x(), origin_roi.y(), data.shape[0]])

        new_vector = end_point - origin
        ac_vector_length = np.sqrt(new_vector.dot(new_vector))
        
        return (new_vector[0], new_vector[1], new_vector[2]) / ac_vector_length, ac_vector_length, (origin[0], origin[1], origin[2])
    
    def get_roi_size(self, ab_vector, ac_vector):
        """
        Returns the size of the ROI expected from the given vectors.
        """
        # Find the width
        w = pg.Point(ab_vector[0], ab_vector[1]) 
    
        # Find the length
        l = pg.Point(ac_vector[0], ac_vector[1])
        
        return w.length(), l.length()
    
    def get_ab_angle(self, with_ab_vector=None):
        """
        Gets ROI.ab_angle. If with_ab_vector is given, then the angle returned is with respect to the given vector 
        """
        if with_ab_vector is not None:
            corner = pg.Point(with_ab_vector[0], with_ab_vector[1])
            return corner.angle(pg.Point(1, 0))
        else:
            return self.ab_angle
        
    def get_ac_angle(self, with_ac_vector=None):
        """
        Gets ROI.ac_angle. If with_ac_vector is given, then the angle returned is with respect to the given vector 
        """
        if with_ac_vector is not None:
            l = pg.Point(with_ac_vector[0], with_ac_vector[1])  # Explain this. 
            corner = pg.Point(l.length(), with_ac_vector[2])
            
            if with_ac_vector[0] < 0:  # Make sure this points to the correct direction 
                corner = pg.Point(-l.length(), with_ac_vector[2])
            
            return pg.Point(0, 1).angle(corner)
        else:
            return self.ac_angle


class Target(pg.GraphicsObject):
    sigDragged = QtCore.Signal(object)
    
    def __init__(self, movable=True):
        pg.GraphicsObject.__init__(self)
        self.movable = movable
        self._bounds = None
        self.color = (255, 255, 0)
        self.labelAngle = 0
        self.label = "N/A"
        self.showLabel = False
        self.ccf_coordinate = (0, 0, 0)
        self.lims_id = 0
        
    def setLabel(self, l):
        if not l:
            self.label = "N/A"
        else:
            self.label = l
            
    def setCoordinate(self, coords):
        if not coords:
            self.ccf_coordinate = (0, 0, 0)
        else:
            self.ccf_coordinate = coords

    def boundingRect(self):
        w = self.pixelLength(pg.Point(1, 0))
        if w is None:
            return QtCore.QRectF()
        w *= 5
        h = 5 * self.pixelLength(pg.Point(0, 1))
        r = QtCore.QRectF(-w*2, -h*2, w*4, h*4)
        return r

    def viewTransformChanged(self):
        self._bounds = None
        self.prepareGeometryChange()

    def paint(self, p, *args):
        p.setRenderHint(p.Antialiasing)
        px = self.pixelLength(pg.Point(1, 0))
        py = self.pixelLength(pg.Point(0, 1))
        w = 4 * px
        h = 4 * py
        r = QtCore.QRectF(-w, -h, w*2, h*2)
        p.setPen(pg.mkPen(self.color))
        p.setBrush(pg.mkBrush(0, 0, 255, 100))
        p.drawEllipse(r)
        p.drawLine(pg.Point(-w*2, 0), pg.Point(w*2, 0))
        p.drawLine(pg.Point(0, -h*2), pg.Point(0, h*2))
        
        if self.showLabel:
            angle = self.labelAngle * np.pi / 180.
            pos = p.transform().map(QtCore.QPointF(0, 0)) + 20 * QtCore.QPointF(np.cos(angle), -np.sin(angle))
            p.resetTransform()
            p.drawText(QtCore.QRectF(pos.x()-10, pos.y()-10, 250, 20), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self.label)
          
    def mouseDragEvent(self, ev):
        if not self.movable:
            return
        if ev.button() == QtCore.Qt.LeftButton:
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
            ev.accept()
            
            if not self.moving:
                return
                
            self.setPos(self.cursorOffset + self.mapToParent(ev.pos()))
            if ev.isFinish():
                self.moving = False
                self.sigDragged.emit(self)

    def hoverEvent(self, ev):
        if self.movable:
            ev.acceptDrags(QtCore.Qt.LeftButton)


def readNRRDAtlas(nrrdFile=None):
    """
    Download atlas files from:
      http://help.brain-map.org/display/mouseconnectivity/API#API-DownloadAtlas
    """
    import nrrd
    if nrrdFile is None:
        displayMessage('Please Select NRRD Atlas File')
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
        displayMessage('Select NRRD annotation file')
        nrrdFile = QtGui.QFileDialog.getOpenFileName(None, "Select NRRD annotation file")

    if ontologyFile is None:
        displayMessage('Select ontology file (json)')
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


def displayError(error):
    print error
    sys.excepthook(*sys.exc_info())
    err = QtGui.QErrorMessage()
    err.showMessage(error)
    err.exec_()


def displayMessage(message):
    box = QtGui.QMessageBox()
    box.setIcon(QtGui.QMessageBox.Information)
    box.setText(message)
    box.setStandardButtons(QtGui.QMessageBox.Ok)
    box.exec_()

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
    v.setWindowTitle('CCF Viewer')
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
            print "Unexpected error when creating ccf.ma file with " + atlasFile
            if os.path.isfile(atlasFile):
                try:
                    print "Removing ccf.ma"
                    os.remove(atlasFile)
                except:
                    print "Error removing ccf.ma:"
                    sys.excepthook(*sys.exc_info())
            raise   

    if os.path.isfile(labelFile):
        label = metaarray.MetaArray(file=labelFile, readAllData=True)
    else:
        try:
            label = readNRRDLabels()
            writeFile(label, labelFile)
        except:     
            print "Unexpected error when creating ccf_label.ma file with " + labelFile
            if os.path.isfile(labelFile):
                try:
                    print "Removing ccf.ma and ccf_label.ma files..."
                    if os.path.isfile(atlasFile):
                        os.remove(atlasFile)
                    if os.path.isfile(labelFile):
                        os.remove(labelFile)
                except:
                    print "Error removing ccf.ma and ccf_label.ma:"
                    sys.excepthook(*sys.exc_info())
            raise 

    v.setAtlas(atlas)
    v.setLabels(label)

    if sys.flags.interactive == 0:
        app.exec_()
