import sys
import os
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtWidgets, QtGui, Qt
import pyqtgraph.opengl as gl
import pyqtgraph as pg

from PIL import Image
from PIL import ImageEnhance

import json
import cv2
from pathlib import Path
from collections import defaultdict

from controls import TreeModel, LabeledSlider, EditViewBox, Transform, SliceImage, LabeledCircleWidget
from atlas import read_ontology, id_colors, color_atlas, get_atlas
import csv



class Viewer(QtWidgets.QMainWindow):

    def __init__(self):

        # Create and initialize window

        super(Viewer, self).__init__()  # Calling the parent class __init__ method

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Logging

        self._logger = logging.getLogger('Atlaslog')

        self._logger.setLevel(logging.DEBUG)

        self._log_handler = RotatingFileHandler('Atlaser.log', maxBytes=1e6, backupCount=1)

        formatter = logging.Formatter(

            '%(asctime)s :: %(filename)s :: %(funcName)s :: line %(lineno)d :: %(levelname)s :: %(message)s',

            datefmt='%Y-%m-%d:%H:%M:%S')

        self._log_handler.setFormatter(formatter)

        self._logger.addHandler(self._log_handler)

        sys.excepthook = handle_exception

        self._logger.info('Initializing the window')

        # Menu bar

        self.file_menu = QtWidgets.QMenu('&File', self)

        self.file_menu.addAction('&Open data', self.select_data_file, QtCore.Qt.CTRL + QtCore.Qt.Key_O)

        self.file_menu.addAction('&Export points', self.export_points, QtCore.Qt.CTRL + QtCore.Qt.Key_E)

        self.file_menu.addAction('&Quit', self.close, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)

        self.edit_menu = QtWidgets.QMenu('&Edit', self)

        self.edit_menu.addAction('&Undo', self.undo, QtCore.Qt.CTRL + QtCore.Qt.Key_Z)

        self.edit_menu.addAction('Clear cells', self.clear_cells)
        
        self.help_menu = QtWidgets.QMenu('&Help', self)

        self.help_menu.addAction('&Manuel', self.help, QtCore.Qt.CTRL + QtCore.Qt.Key_H)

        self.menuBar().addMenu(self.file_menu)

        self.menuBar().addMenu(self.edit_menu)

        self.menuBar().addMenu(self.help_menu)


        # MAIN LAYOUT

        self.main_layout = QtWidgets.QHBoxLayout()

        # create a main widget

        window_size = [1000, 1500]

        self.main_widget = QtWidgets.QWidget(self)

        self.main_widget.setLayout(self.main_layout)

        self.main_widget.setGeometry(0, 0, window_size[0], window_size[1])

        self.main_widget.setWindowTitle('Slice viewer - No Data')

        self.setCentralWidget(self.main_widget)

        # LAYOUTS

        self.h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.right_widget = QtWidgets.QWidget(self)

        self.g_layout = pg.GraphicsLayoutWidget(self.main_widget)

        self.v_layout_right = QtWidgets.QVBoxLayout()

        self.h_layout_right = QtWidgets.QHBoxLayout()

        self.h_layout_buttons = QtWidgets.QHBoxLayout()

        self.h_atlas = QtWidgets.QHBoxLayout()



        # WIDGETS
        
        # Slider for channel choice

        self.channel_sl = LabeledSlider('Channel')

        self.channel_sl.valueChanged.connect(self.channel_change)

        self.channel_sl.setSingleStep(1)

        self.channel_sl.setEnabled(False)

        # Brain slice slider

        self.brain_sl = LabeledSlider('Brain slice')

        self.brain_sl.valueChanged.connect(self.brain_change)

        self.brain_sl.setSingleStep(1)

        self.brain_sl.setRange(0, 4)

        self.brain_sl.setEnabled(False)

        # Slider for luminosity

        self.lum_sl = LabeledSlider('Brightness')

        self.lum_sl.setRange(0, 100)

        self.lum_sl.setValue(50)

        self.lum_sl.valueChanged.connect(self.luminosity_change)

        self.lum_sl.setEnabled(False)

        # Slider for contrast

        self.contrast_sl = LabeledSlider('Contrast')

        self.contrast_sl.setRange(0, 100)

        self.contrast_sl.setValue(50)

        self.contrast_sl.valueChanged.connect(self.contrast_change)

        self.contrast_sl.setEnabled(False)

        # Cell selection mode checkbox

        self.cell_select_cb = QtWidgets.QCheckBox('Cell selection mode')

        self.cell_select_cb.setChecked(False)

        self.cell_select_cb.clicked.connect(self.switch_cellmode)

        # Slider for atlas slice

        self.slice_sl = LabeledSlider('Atlas slice')

        self.slice_sl.setRange(0, 1)

        self.slice_sl.setSingleStep(1)

        self.slice_sl.valueChanged.connect(self.update_atlas)

        # Slider for atlas opacity

        self.alpha_sl = LabeledSlider('Atlas opacity',)

        self.alpha_sl.setRange(0, 100)

        self.alpha_sl.setSingleStep(1)

        self.alpha_sl.setValue(50)

        self.alpha_sl.valueChanged.connect(self.atlas_alpha)


        # Transformation sliders

        ### AJOUT PAR LE GROUPE ###
        # Image rotation widget

        self.rot_sl = LabeledCircleWidget('Image rotation', factor = 1)    # Création du bouton circulaire pour la rotation

        self.rot_sl.setRange(0, 360)   # les valeurs vont de 0 à 3600 divisé par le facteur de dessus (permet d'avoir plus de précision)

        self.rot_sl.setSingleStep(1)    # on définit de combien en combien on se déplace

        self.rot_sl.valueChanged.connect(self.sl_rot_changed)   # on lie le bouton à la fonction de rotation définit plus bas
        
        self.rot_sl.setValue(90)    # On fixe la valeur à 90° pour que l'image soit alignée avec l'atlas
        ###########################


        # Scale attribute (no widget and function used)

        self.scale_sl = LabeledSlider('Scale', factor = 1000)

        self.scale_sl.setRange(1, 2000)

        self.scale_sl.setSingleStep(1)

        self.scale_sl.valueChanged.connect(self.sl_scale_changed)

        # Show/Hide atlas check box

        self.show_atlas_cb = QtWidgets.QCheckBox('Show atlas')

        self.show_atlas_cb.setChecked(True)

        self.show_atlas_cb.stateChanged.connect(self.show_atlas)

        # Atlas orientation combo box

        self.orientation_cb = QtWidgets.QComboBox(self)

        self.orientation_cb.addItems(['Coronal', 'Horizontal', 'Sagittal'])

        self.orientation_cb.currentIndexChanged.connect(self.change_orientation)

        # Region tree

        self.tree = QtWidgets.QTreeView(self.right_widget)

        self.onto = read_ontology('mouse_ontology.json')

        self.tree_model = TreeModel(None, self.onto)

        self.tree.setModel(self.tree_model)

        self.tree.expanded.connect(self.expanded)

        self.tree.selectionModel().currentChanged.connect(self.select_region)

        # Buttons

        self.fliplr_pb = QtWidgets.QPushButton('Flip &left-right')

        self.fliplr_pb.setCheckable(True)

        self.flipud_pb = QtWidgets.QPushButton('Flip &up-down')

        self.flipud_pb.setCheckable(True)

        self.fliplr_pb.clicked.connect(self.flip_lr)

        self.flipud_pb.clicked.connect(self.flip_ud)

        # Graph widgets

        self.anat_image = pg.ImageItem()

        self.atlas_image = pg.ImageItem()

        self.template_image = pg.ImageItem()

        self.zoom_image = pg.ImageItem()

        self.atlas_image.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)

        # Mouse sensitivity on both windows

        self.atlas_image.setZValue(10)

        self.zoom_image.setZValue(10)

        # Create the selection tool (circles)

        self.cell_scatter = pg.ScatterPlotItem()

        ### MODIFICATION PAR LE GROUPE ###
        self.cell_pen = pg.mkPen(color=(242, 142, 85, 200), width=1.5)  # On change la couleur des ronds dessinés quand on fait une sélection ainsi
                                                                        # que la largeur pour qu'ils soient bien visibles
        self.cell_brush = pg.mkBrush(None)
        ############################

        self.cell_scatter.setBrush(self.cell_brush)

        self.cell_scatter.setPen(self.cell_pen)

        # Creation of ViewBox anat : adding the atlas and the image

        self.vb_anat = EditViewBox()

        self.vb_anat.rotation.connect(self.rotation)

        self.vb_anat.translation.connect(self.translation)

        self.vb_anat.scale.connect(self.scaling)

        self.vb_anat.cell_select.connect(self.cell_clicked)

        self.g_layout.addItem(self.vb_anat, row=0, col=0)

        self.vb_anat.setAspectLocked()

        self.vb_anat.invertY()

        # Adding the image, the atlas and the circle tool for selection into the left windows

        self.vb_anat.addItem(self.anat_image)

        self.vb_anat.addItem(self.atlas_image)

        self.vb_anat.addItem(self.cell_scatter)

        # Zoom inset on the high quality image

        self.vb_inset = self.g_layout.addViewBox(row=0, col=1)

        self.vb_inset.addItem(self.zoom_image)

        self.vb_inset.invertY()

        self.vb_inset.setAspectLocked(True)
	
	    # Cross pointer
        ### MODIFICATION PAR LE GROUPE ###
        cursor_pen = pg.mkPen(color=(242, 142, 85), width=2)    # Modification de la couleur et de la largueur de la croix (plus fine et couleur moins voyante)

        ### Modifications des coordonnées de dessins des lignes : les quatre traits de la croix se croisent désormais et la croix est fermé
        self._h_line_l = QtWidgets.QGraphicsLineItem(0, 2, 2, 2)

        self._h_line_r = QtWidgets.QGraphicsLineItem(1, 2, 4, 2)

        self._v_line_u = QtWidgets.QGraphicsLineItem(2, 3, 2, 4)

        self._v_line_d = QtWidgets.QGraphicsLineItem(2, 0, 2, 3)
        ###################################

        self._h_line_l.setParentItem(self.zoom_image)

        self._h_line_r.setParentItem(self.zoom_image)

        self._v_line_u.setParentItem(self.zoom_image)

        self._v_line_d.setParentItem(self.zoom_image)

        self._h_line_l.setPen(cursor_pen)

        self._h_line_r.setPen(cursor_pen)

        self._v_line_u.setPen(cursor_pen)

        self._v_line_d.setPen(cursor_pen)

        self.vb_inset.enableAutoRange()


        # LAYOUT setup

        self.right_widget.setLayout(self.v_layout_right)

        self.v_layout_right.addWidget(self.tree)

        self.v_layout_right.addWidget(self.lum_sl)

        self.h_atlas.addWidget(self.orientation_cb)

        self.h_atlas.addWidget(self.show_atlas_cb)

        self.v_layout_right.addLayout(self.h_atlas)

        self.v_layout_right.addWidget(self.slice_sl)

        self.v_layout_right.addWidget(self.alpha_sl)

        self.v_layout_right.addWidget(self.rot_sl)

        self.h_layout_buttons.addWidget(self.fliplr_pb)

        self.h_layout_buttons.addWidget(self.flipud_pb)

        self.v_layout_right.addLayout(self.h_layout_buttons)

        self.v_layout_right.addWidget(self.cell_select_cb)

        self.v_layout_right.addLayout(self.h_layout_right)

        self.h_splitter.addWidget(self.g_layout)

        self.h_splitter.setStretchFactor(0, 5)

        self.h_splitter.addWidget(self.right_widget)

        self.main_layout.addWidget(self.h_splitter)



        # Some shortcuts : moving slice by slice (atlas)
    
        atlas_fwd = QtWidgets.QShortcut(QtCore.Qt.Key_Right, self.main_widget)

        atlas_fwd.activated.connect(self.next_slice)

        atlas_bck = QtWidgets.QShortcut(QtCore.Qt.Key_Left, self.main_widget)

        atlas_bck.activated.connect(self.prev_slice)

        self._logger.info('End of window creation')



    def switch_cellmode(self, checked):

        pass



    def export_points(self):

        pass



    def zoom_change(self, value):

        pass



    def atlas_alpha(self, value):

        pass



    def undo(self):

        pass



    def show_atlas(self):

        pass



    def channel_change(self, value):

        pass



    def z_change(self, value):

        pass



    def brain_change(self, value):

        pass



    def luminosity_change(self, value):

        pass



    def contrast_change(self, value):

        pass



    def next_slice(self):

        pass



    def prev_slice(self):

        pass



    def flip_lr(self):

        pass



    def flip_ud(self):

        pass



    def select_region(self, current, previous):

        pass



    def expanded(self, index):

        self.tree.resizeColumnToContents(1)

        self.tree.resizeColumnToContents(0)



    def rotation(self, value):

        pass



    def translation(self, x_shift, y_shift):

        pass



    def scaling(self, scale):

        pass



    def update_atlas(self, value):

        pass



    def change_orientation(self, index):

        pass



    def select_data_file(self):

        pass



    def sl_rot_changed(self, value):

        pass



    def sl_scale_changed(self, value):

        pass



    def save_transf(self):

        pass



    def load_transf(self):

        pass



    def align(self):

        pass



    def cell_clicked(self, x, y, mx, my):

        pass



    def clear_transf(self):

        pass



    def clear_cells(self):

        pass


    def help_menu(self):

        pass





class AtlasExplorer(Viewer):

    def __init__(self):

        super().__init__()

        self.statusBar().showMessage('Loading...')

        self._logger.info('Loading...')

        Image.MAX_IMAGE_PIXELS = None   # Necessary to open super large tiff files

        # Load atlas and annotations

        self.atlas = np.load('contourify2.npy')

        self.template = get_atlas('average_template_25.nrrd')

        self.raw_atlas = get_atlas('annotation_25.nrrd')

        self.colors = id_colors(self.onto)

        self.c_atlas = np.load('color_atlas.npy')

        self._logger.info('Atlas opened')

        self.p_max = 2**16

        self._c_orient = 0

        self._sel_regions = {}

        self.orientation_cb.setCurrentIndex(2)

        self._data_path = None

        self.slice_image = SliceImage('')

        self.pic = np.zeros((5, 5), dtype=np.uint8)

        self._transf_raw = Image.new('L', (5, 5))   # Create a greyscale 8-bit image. Size of 5x5

        self.transf_raw = self._transf_raw

        self._raw_inset = np.zeros((5, 5), dtype=np.uint8)

        self.raw_inset = self._raw_inset

        self._transf = Transform()

        self.transf = self._transf

        self._inset_size = 1000

        self.cells = []

        self.cell_pos = []

        self.actions = []

        ### AJOUT PAR LE GROUPE ###
        self.rot_sl.setValue(90)
        ###########################

        self.scale_sl.setValue(int(self.transf.scale * self.scale_sl.factor))

        # Capture mouse movements to scroll the inset

        ### MODIFICATION ET AJOUTS PAR LE GROUPE ###
        self._proxy = pg.SignalProxy(self.vb_anat.scene().sigMouseMoved, rateLimit=0.1, delay=.1, slot=self.mouse_moved)

        # On modifie la fenêtre qui repère les mouvements de la souris : c'est la fenêtre de gauche et non plus la fenetre en bas à gauche (qui n'existe plus)

        self.setWindowTitle('Atlaser Sotfware') # On met un titre à la fenetre

        self.help_menu = None   # On initialise à None le manuel d'aide (qui n'est pas ouvert par défaut)

        self.myROIdata = [] # On initialise la liste stockant les coordonnées sauvegardées avec l'outil de sélection implémenté
        ###################################




    def switch_cellmode(self, checked):

        if checked:

            self.raw_inset = np.zeros((5, 5), dtype=np.uint8)

            self.transf_raw = self.apply_transf(self.slice_image.raw_img, False, False)

            self._proxy.rateLimit = 30

            self.statusBar().showMessage('Ready for cell selection', 1500)



    def draw_cross(self):

        """
        Draw the cross cursor on the inset zoom picture
        Place it at the center of the inset, which depends on the exact size of the self._raw_inset array
        """

        # inset size

        w, h = self._raw_inset.shape

        w2, h2 = w//2, h//2

        # Cross on the inset picture

        self._h_line_l.setLine(0, h2, w2, h2)

        self._h_line_r.setLine(w2, h2, w, h2)

        self._v_line_u.setLine(w2, h2, w2, h)

        self._v_line_d.setLine(w2, h2, w2, 0)



    def clear_cells(self):

        self.cells = []

        self.cell_pos = []

        self.cell_scatter.setData(self.cell_pos)



    def clear_transf(self):

        self.transf = Transform()


    def mouse_moved(self, pos):

        """
        Update the zoom inset from the raw image when the mouse cursor moves over the low resolution atlas aligned image
        Parameters
        ----------
        pos: list of SceneEvents
            Contains the mouse position (in scene coordinates, will need conversion)
        """

        # FIXME: When too close to the edges, not possible to get a precise position
        
        if not self.cell_select_cb.isChecked():

            return

        pos = pos[0]

        v_range = self.vb_anat.viewRange()

        x, y = self.convert_mouse_pos(pos.x(), pos.y(), v_range[0][0], v_range[1][0])

        x, y = x, y

        all_scales = self.slice_image.downfactor / self.transf.scale

        raw_x = int((x - self.transf.translation[1]) * all_scales)

        raw_y = int((y - self.transf.translation[0]) * all_scales)

        # If the mouse is out of the picture, don't bother

        if raw_x > self.transf_raw.height or raw_x < 0 or raw_y > self.transf_raw.width or raw_y < 0:

            return

        if self.transf_raw.width < 10:

            return

        raw_x, raw_y = raw_y, raw_x

        start_x = np.clip(raw_x - self._inset_size, 0, self.transf_raw.width - self._inset_size)

        start_y = np.clip(raw_y - self._inset_size, 0, self.transf_raw.height - self._inset_size)

        stop_x = np.clip(raw_x + self._inset_size, self._inset_size, self.transf_raw.width - 1)

        stop_y = np.clip(raw_y + self._inset_size, self._inset_size, self.transf_raw.height - 1)

        self.raw_inset = np.array(self.transf_raw.crop((start_x, start_y, stop_x, stop_y)))

        self.draw_cross()



    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:

        """
        closeEvent of Viewer.
        Perform some cleanup
        """

        a0.accept()

        super().closeEvent(a0)



    ### AJOUT PAR LE GROUPE ###
    def export_points(self):
        """
        Sauvegarde les données stockées dans le dictionnaire cells
        Save cells dictionnary data
        """
        try : 
            for i in range(0, len(self.cells)):     # On parcourt le dictionnaire

                if (self.cells[i]['Region'] == 'None'):     # Certaines régions stockées n'ont pas de parent

                    d = defaultdict(str)    # Initialisation du dictionnaire stockant les string (librairie collections)

                    d[1] = ""   # Le parent 1 (colonne 1) prend par défaut la valeur vide

                    d[2] = ""   # Idem pour la deuxième colonne

                    d = dict(d) # On convertit d en dictionnaire classique

                    self.cells[i].update(d)  # On ajoute à cells, pour chaque structure, seulement les deux derniers parents

                    self.cells[i]['structure'] = self.cells[i].pop('structure') # On retire la clé structure qui correspond à toute l'arborescence de la structure

                    del self.cells[i]['Region'] # On supprime la clé région car on n'a plus besoin de celle-ci pour la sauvegarde
                    

                else:

                    d = defaultdict(str)    # Initialisation du dictionnaire stockant les string (librairie collections)

                    d[1] = self.cells[i]['Region'][len(self.cells[i]['Region']) - 2]    # On récupère le parent le plus proche de la structure enregistrée dans cells

                    d[2] = self.cells[i]['Region'][len(self.cells[i]['Region']) - 3]    # On récupère le parent de la structure stockée dans d[1]

                    d = dict(d)     # idem qu'avant

                    self.cells[i].update(d)     # idem qu'avant

                    self.cells[i]['structure'] = self.cells[i].pop('structure')     # idem qu'avant

                    del self.cells[i]['Region']     # idem qu'avant
                    

            im_path = Path(self.data_path)  # On initialise le chemin de sauvegarde

            path = im_path.parent / f'cells_{im_path.stem}.csv'     # On définit le chemin de sauvegarde et le nom du fichier à la sortie
                                                                    # Le fichier aura le même nom que l'image 
                                                                    # et est sauvegardé dans le même dossier que celle-ci (souhait de l'utilisateur)

            keys = self.cells[i].keys()     # On récupère les clés du dictionnaire cells mis-à-jour au-dessus

            with open(path, 'a', newline='') as file:   # Pour chaque élément de cells, on écrit dans le fichier CSV. 
                                                        # Le mode a permet d'ajouter dans un fichier déjà existant

                dict_writer = csv.DictWriter(file, keys)

                dict_writer.writerows(self.cells)

            self.cells = []     # On vide le dictionnaire après une sauvegarde.
        
        
        except KeyError:    # On évite les erreurs de clés pouvant arriver avec des régions sans parent

            pass
        ##########################

            

    
    # Displays the image in the right windows when cell mode selection is activated
    
    def apply_transf(self, img, scale=True, trans=True):

        new_img = img.copy()

        if self.transf.flipped_lr:

            new_img = Image.fromarray(np.flipud(np.array(new_img)))

        if self.transf.flipped_ud:

            new_img = Image.fromarray(np.fliplr(np.array(new_img)))

        n_width = int(new_img.width*self.transf.scale)

        n_height = int(new_img.height * self.transf.scale)

        if scale:

            new_img = new_img.resize((n_width, n_height))

        if self.transf.rotation != 90:

            new_img = new_img.rotate(self.transf.rotation, expand=True)

        else:

            new_img = Image.fromarray(np.flipud(np.array(new_img).T))

        w, h = new_img.size

        if trans:

            x_shift, y_shift = self.transf.translation

        else:

            x_shift, y_shift = 0, 0

        buffer_img = Image.new(img.mode, (w+abs(x_shift), h+abs(y_shift)))

        buffer_img.paste(new_img, (x_shift, y_shift))

        return buffer_img



    def update_img(self):

        if self._data_path is None:

            return

        self.pic = np.array(self.apply_transf(self.slice_image.dw_img))

        # The old transformation is no longer valid for the inset display and cell selection

        self.cell_select_cb.setChecked(False)

        # Erase the previously transformed image to avoid confusion

        # Now the transformed picture needs to be recomputed by switching into cell selection mode

        self.raw_inset = np.zeros((5, 5), dtype=np.uint8)

        self.transf_raw = Image.new('L', (5, 5))

        # Useless to keep track of mouse movements too often since we have nothing to display

        self._proxy.rateLimit = 0.1



    def convert_mouse_pos(self, x, y, min_x, min_y):

        # Original position zoom corrected

        ex, ey = x, y

        # Correct for zoom

        px_width, px_height = self.anat_image.pixelSize()

        x /= px_width

        y /= px_height

        # Inverse transformation from display to data

        coords = np.intp(pg.transformCoordinates(self.anat_image.inverseDataTransform(), np.array([x, y])))

        dx = min_x + coords[0]

        dy = min_y + coords[1]

        # Divide dx and dy by the number of the atlas scale (here, 10)

        dx = dx

        dy = dy

        return dx, dy



    ### MODIFICATION PAR LE GROUPE ###
    def cell_clicked(self, x, y, mx, my):

        dx, dy = self.convert_mouse_pos(x, y, mx, my)

        dx, dy = int(dx), int(dy)

        # Get structure id if possible

        c_atlas = self.get_slice(self.raw_atlas, self.c_slice)

        if dx < 0 or dx >= c_atlas.shape[0] or dy < 0 or dy >= c_atlas.shape[1]:

            return

        reg_id = c_atlas[dx, dy]

        # Get region name

        l_reg = [(i, r) for i, r in enumerate(self.tree_model.regions) if r.id == reg_id]

        if len(l_reg) == 0:

            return

        else:

            l_reg = l_reg[0] # id de la structure récupérée
        
        reg = l_reg[1]  # nom de la structure récupérée

        ### AJOUT PAR LE GROUPE ###
        parent_reg = get_parent(self.onto, l_reg[0])    # On récupère la liste de tous les parents

        if parent_reg == None:  # S'il n'y a pas de parents, on convertit en string None pour le stockage dans le dictionnaire de string

            parent_reg = str(None)
        ###########################

        reg_ix = self.tree_model.match(self.tree_model.index(0, 0, QtCore.QModelIndex()), QtCore.Qt.DisplayRole, QtCore.QVariant(reg.abbr),

                                    1, QtCore.Qt.MatchExactly|QtCore.Qt.MatchRecursive)

        if reg_ix:

            reg_ix = reg_ix[0]

        self.tree.setCurrentIndex(reg_ix)

        ### AJOUT ET MODIFICATIONS PAR LE GROUPE ###
        if self.cell_select_cb.isChecked():     # Quand le mode de sélection est activé

            self.cells.append({'pos': (dx + 1, dy + 1), 'Region': parent_reg, 'structure': str(reg)})
            # On stocke les coordonnées (+1 pour pouvoir tracer à l'écran sur le pixel choisi), la région et son arborescence et la structure cliquée (nom)

            self.cell_pos.append((dx + 1, dy + 1))  # On ajoute les coordonnées à la liste des points dessinés à l'écran

            self.actions.append(((self.cells.pop, self.cell_pos.pop), (-1, -1)))

            self.cells = check_duplicate(self.cells)    # On contrôle les doublons dans cells
    
            self.cell_scatter.setData(pos = self.cell_pos)  # On dessine les ronds enregistrés

            # On contrôle qu'il n'y ait pas de doublons dessinés
            for i in range(0, len(self.cell_pos) - 1):

                if (dx + 1, dy + 1) == self.cell_pos[i]:

                    self.cell_pos.remove(self.cell_pos[i])

                    self.cell_scatter.setData(pos = self.cell_pos)

            # On contrôle qu'il n'y ait pas de doublons stockés dans le dictionnaire cells qui va être exporté à la fin          
            for i in range(0, len(self.cells) - 1):

                if (dx + 1, dy + 1) == self.cells[i]['pos']:
                    
                    self.cells.remove(self.cells[i])
        ##############################################




    def zoom_change(self, value):

        self.slice_image.c_zoom = value

        self.auto_scale()



    def atlas_alpha(self, value):

        self.update_atlas(self.slice_sl.value())


    ### AJOUT PAR LE GROUPE ###
    def undo(self):
        """
            Suppression des doublons affichés à l'écran. Se fait automatiquement quand on dessine un autre point que celui qui est en double.
            Accessible avec CTRL + Z. A utiliser en fin de manipulation pour avoir un dictionnaire de données propre et sans doublons

            Delete duplicates drawn circles on screen. It is automatic when the user clicks on another area that the one with duplicates.
            Can be used manually with CTRL + Z. Should be used when finishing the cells selection to clean data that will be save
        """
        actions, args = self.actions.pop(-1)

        for a, ix in zip(actions, args):

            a(ix)
        
        self.cell_pos = check_duplicate(self.cell_pos)  # On check les doublons dans la liste contenant les points dessinés à l'écran et on les supprime

        self.cell_scatter.setData(pos = self.cell_pos)  # On redessine tout le dictionnaire (update)
    ###################################



    def show_atlas(self):

        self.atlas_image.setVisible(self.show_atlas_cb.isChecked())
        


    def channel_change(self, value):

        self.slice_image.c_channel = value

        self.auto_scale()



    def z_change(self, value):

        self.slice_image.c_zslice = value

        self.auto_scale()



    def brain_change(self, value):

        self.slice_image.c_slice = value

        self._logger.debug('Changing brain slice')

        self.auto_scale()



    def luminosity_change(self, value):

        self.apply_brightness(value)



    def contrast_change(self, value):

        self.pic = self.apply_contrast(self.slice_image.img, value)

        self.apply_brightness(self.lum_sl.value())
        


    def apply_brightness(self, value):

        self.anat_image.setLevels((0, self.slice_image.p_max * (2 - value / 50)))



    ### AJOUT PAR LE GROUPE ### 
    # def update(self, roi):    
    #     roidata = []
    #     roidata.append(roi.pos())
    #     roidata.append(roi.size() * 8)
    #     self.myROIdata = roidata
    #     print (self.myROIdata)



    # def select_tool(self):

    #     # Création de l'outil de selection, sur l'image et d'une taille correcte
    #     myROI = pg.ROI([1000,1000], [350,500])

    #     # Ajout de l'outil sur la fenetre qui contient l'image
    #     self.vb_anat.addItem(myROI)

    #     # Gère le scaling horizontal
    #     myROI.addScaleHandle([1, 0.5], [0.5, 0.5])
    #     myROI.addScaleHandle([0, 0.5], [0.5, 0.5])

    #     # Gère le scaling vertical
    #     myROI.addScaleHandle([0.5, 0], [0.5, 1])
    #     myROI.addScaleHandle([0.5, 1], [0.5, 0])

    #     # Gère le scaling horizontal et vertical
    #     myROI.addScaleHandle([1, 1], [0, 0])
    #     myROI.addScaleHandle([0, 0], [1, 1])

    #     myROI.sigRegionChanged.connect(self.update)

    #     self.update(myROI)
    #############################



    @staticmethod

    def apply_contrast(img, value):
        
        def contrast_lut(v):

            m = 2**16   # Max in 16 bits

            v /= m      # Normalize to 1

            # 50 is the middle of the slider, so no extra contrast.

            # Over 50: exponent from 1 to 2

            # Under 50: exponent from 0 to 1

            pv = np.clip(np.power(v,  value / 50), 0, 1)

            pv *= m     # Go back to original scale

            return pv

        pic = np.array(img, dtype=np.float32)

        return contrast_lut(pic)


    ### AJOUT PAR LE GROUPE ###
    def next_slice(self):

        self.slice_sl.setValue(self.c_slice + 1)

        self.apply_brightness(self.lum_sl.value())  # Evite le problème de reset de la valeur en appliquant la valeur actuelle de luminosité




    def prev_slice(self):

        self.slice_sl.setValue(self.c_slice - 1)

        self.apply_brightness(self.lum_sl.value())  # Evite le problème de reset de la valeur en appliquant la valeur actuelle de luminosité


    def flip_lr(self):

        self.transf.flipped_lr = not self.transf.flipped_lr

        self.apply_brightness(self.lum_sl.value())  # Evite le problème de reset de la valeur en appliquant la valeur actuelle de luminosité



    def flip_ud(self):

        self.transf.flipped_ud = not self.transf.flipped_ud

        self.apply_brightness(self.lum_sl.value())  # Evite le problème de reset de la valeur en appliquant la valeur actuelle de luminosité



    def rotation(self, value):

        self.transf.add_rotation(value)

        self.update_transf_sliders()



    def translation(self, x_shift, y_shift):

        x_shift, y_shift = int(-x_shift), int(-y_shift)

        self.transf.add_translation((x_shift, y_shift))

        self.apply_brightness(self.lum_sl.value())  # Evite le problème de reset de la valeur en appliquant la valeur actuelle de luminosité



    def scaling(self, value):

        scale = value / self.slice_image.img.height / 5

        self.transf.add_scale(scale)

        self.update_transf_sliders()

        self.apply_brightness(self.lum_sl.value())  # Evite le problème de reset de la valeur en appliquant la valeur actuelle de luminosité



    def sl_rot_changed(self, value):

        self.transf.rotation = value / self.rot_sl.factor

        self.apply_brightness(self.lum_sl.value())  # Evite le problème de reset de la valeur en appliquant la valeur actuelle de luminosité
    ##################################



    def update_transf_sliders(self):

        self.rot_sl.setValue(int(self.transf.rotation))

        self.fliplr_pb.setChecked(bool(self.transf.flipped_lr))

        self.flipud_pb.setChecked(bool(self.transf.flipped_ud))



    @property

    def transf_raw(self):

        return self._transf_raw



    @transf_raw.setter

    def transf_raw(self, value):

        self._transf_raw = value

        self.raw_inset = np.zeros((5, 5), dtype=np.uint8)



    @property

    def raw_inset(self):

        return self._raw_inset



    @raw_inset.setter

    def raw_inset(self, value):

        self._raw_inset = value

        self.zoom_image.setImage(value)



    @property

    def transf(self):

        return self._transf



    @transf.setter

    def transf(self, value):

        self._transf = value

        self.transf.rotation_changed.connect(self.update_img)

        self.transf.translation_changed.connect(self.update_img)

        self.transf.scale_changed.connect(self.update_img)

        self.transf.fliplr_changed.connect(self.update_img)

        self.transf.flipud_changed.connect(self.update_img)

        self.update_img()

        self.update_transf_sliders()



    @property

    def sel_regions(self):

        return self._sel_regions



    @sel_regions.setter

    def sel_regions(self, value):

        self._sel_regions = value

        self.update_atlas(self.slice_sl.value())



    @property

    def pic(self):

        return self._pic



    @pic.setter

    def pic(self, value):

        self._pic = value

        self.anat_image.setImage(value)



    @property

    def data_path(self):

        return self._data_path



    @data_path.setter

    def data_path(self, value):
        """
        When the data path changes, open the corresponding data file
        Parameters
        ----------
        value: str
            Path to the data as selected from the dialog box
        """

        self._data_path = value

        self.load_image()



    @property

    def c_orient(self):

        return self._c_orient



    @c_orient.setter

    def c_orient(self, value):

        self._c_orient = value

        self.slice_sl.setRange(0, self.template.shape[value] - 1)



    def get_slice(self, volume, value):

        if self.c_orient == 0:

            s = volume[value, ...]

            d = (1, 0, 2) if len(s.shape) == 3 else (1, 0)

            s = s.transpose(d)

        elif self.c_orient == 1:

            s = volume[:, value, ...]

        else:

            if len(volume.shape) == 3:

                s = volume[:, :, value]

            else:

                s = volume[:, :, value, :]

        return s



    def update_atlas(self, value):

        self.c_slice = value

        c_atlas = self.get_slice(self.c_atlas, value)

        s_atlas = self.get_slice(self.atlas, value)

        c_template = self.get_slice(self.template, value)

        atlas_alpha = int(self.alpha_sl.value()/100*255)

        ids = np.unique(s_atlas)

        alpha = np.zeros(s_atlas.shape) + atlas_alpha

        alpha[np.isin(s_atlas, list(self.sel_regions))] = 0

        gi = np.logical_and(alpha == atlas_alpha, np.any(c_atlas > 0, 2))

        c_atlas[gi, ...] = 255

        c_atlas = np.dstack((c_atlas, alpha))

        self.template_image.setImage(c_template)

        self.atlas_image.setImage(c_atlas)



    def change_orientation(self, index):

        self.c_orient = index

        self.update_atlas(self.slice_sl.value())



    def select_data_file(self):

        cwd = os.getcwd()

        dpath, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Choose an image file",

                                                               cwd, 'Image (*.tiff *.tif *.vsi *.ndpis)')

        if dpath != '':

            # Set the new data path. The data_path setter will take care of the rest

            self.data_path = dpath



    ### AJOUT PAR LE GROUPE ###
    # Helping manuel to get shortcuts and tips

    def help(self):
        """
            Ouvre une image contenant les raccourcis du logiciel utilisables par l'utilisateur
            Open an image with all the shortcuts the user can use
        """
        
        img = cv2.imread("helpManuel.png")  # ouverture de l'image manuel
        
        cv2.imshow('Help manuel', img)  # affichage de l'image à l'écran
    ############################



    def auto_scale(self):

        atlas_shape = self.c_atlas.shape[:2]

        self.actions = []

        self.cells = []

        self.cell_pos = []

        self.transf.scale = 1

        self.update_img()



    def update_n_slices(self, n_zslices, n_channels, n_res_levels, n_brainslices):

        self.channel_sl.setRange(0, n_channels - 1)

        self.brain_sl.setRange(0, n_brainslices - 1)

        self._logger.debug(f'Number of z slices: {n_zslices}')



    def update_max_value(self, p_max):

        self.p_max = p_max

        self.apply_brightness(self.lum_sl.value())



    def img_updated_event(self):

        self.pic = self.apply_contrast(self.slice_image.img, self.contrast_sl.value())

        self.apply_brightness(self.lum_sl.value())

        self._logger.debug(f'Pic max: {self.pic.max()}')

        self._logger.debug(f'Display range: {self.anat_image.levels}')



    def load_image(self):

        """
        Load the image defined in self.data_path
        Loading procedure depends on format. Handles tif through PIL, vsi through python-bioformats
        """

        # Load the imge

        self.slice_image = SliceImage(self.data_path)

        self.slice_image.n_slices_known.connect(self.update_n_slices)

        self.slice_image.img_updated.connect(self.img_updated_event)

        self.slice_image.max_value_changed.connect(self.update_max_value)

        self.slice_image.load()

        # Reset contrast and brightness

        if self.slice_image.is_tiff:

            self.channel_sl.setEnabled(False)

            self.scale_sl.setValue(0.2)

        else:

            self.channel_sl.setEnabled(True)

            self.channel_sl.setValue(self.slice_image.c_channel)

        if self.slice_image.is_ndpis:

            self.brain_sl.setEnabled(True)

        else:

            self.brain_sl.setEnabled(False)

    # Finish initialization

        self.auto_scale()

        self.lum_sl.setEnabled(True)

        self.contrast_sl.setEnabled(True)

        self.transf_raw = self.slice_image.raw_img



    def select_region(self, current, previous):

        c_item = self.tree.model().data(current, QtCore.Qt.UserRole)

        sel_regions = set([c_item.region.id])

        children = set([ch.region.id for ch in self.get_all_children(c_item)])

        sel_regions.update(children)

        self.sel_regions = sel_regions

        self._logger.debug('Region selected')



    def get_all_children(self, region):

        for ch in region.child_items:

            yield ch

            if ch.child_items:

                yield from self.get_all_children(ch)



    def save_transf(self):

        if self.data_path is None:

            return

        self._logger.debug(self.cells)

        params = dict(manual=self.transf.params, image=self.data_path, cells=self.cells)

        params['manual']['atlas_orientation'] = self.c_orient

        params['manual']['atlas_slice'] = self.slice_sl.value()

        params['manual']['prefactor'] = self.slice_image.downfactor

        im_path = Path(self.data_path)

        path = im_path.parent / f'params_{im_path.stem}.json'

        with open(path, 'w') as f:

            json.dump(params, f, indent=4)



    def load_transf(self):

        cwd = os.getcwd()

        dpath, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a transformation file",

                                                               cwd, 'JSON (*.json)')

        if dpath != '':

            # Load the transformation

            with open(dpath, 'r') as f:

                workspace = json.load(f)

                image = workspace['image']

                d_transf = workspace['manual']

                l_cells = workspace['cells']

            transf = Transform((d_transf['x_shift'], d_transf['y_shift']), d_transf['rotation'],

                               d_transf['scale'], d_transf['flip_lr'], d_transf['flip_ud'])

            try:

                self.orientation_cb.setCurrentIndex(d_transf['atlas_orientation'])

                self.slice_sl.setValue(int(d_transf['atlas_slice']))

            except KeyError:

                pass

            self.data_path = image

            self.transf = transf

            self.cells = l_cells

            self.cell_pos = [tuple(c['pos']) for c in l_cells]

            self.cell_scatter.setData(pos=self.cell_pos)



def handle_exception(exc_type, exc_value, exc_traceback):

    """Handle uncaugt exceptions and print in logger."""

    logger = logging.getLogger('Atlaslog')

    if issubclass(exc_type, KeyboardInterrupt):

        sys.__excepthook__(exc_type, exc_value, exc_traceback)

        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))



### AJOUT PAR LE GROUPE ###
def check_duplicate(list):
    """
        Parcourt une liste et renvoie une liste sans doublons

        Takes a list and return a list without duplicate values
    """

    res = [] 

    for i in list: 

        if i not in res: 

            res.append(i)

    return res 
###########################


### AJOUT PAR LE GROUPE ###
def get_parent(json_tree, target_id):
    """
        Permet à partir d'un arbre stocké dans un fichier JSON et un id cible d'obtenir toute l'arborescence de la région ayant l'id cible

        From a JSON file tree data and a target ID, return the target region and its parent
    """

    for element in json_tree:

        if element['id'] == target_id:

            return [element['id']]

        else:

            if element['children']:

                check_child = get_parent(element['children'], target_id)

                if check_child:

                    return [element['name']] + check_child
###########################




if __name__ == '__main__':

    qApp = QtWidgets.QApplication(sys.argv)

    window = AtlasExplorer()

    window.setWindowIcon(QtGui.QIcon("logo.png"))   # Ajout d'un logo au logiciel

    window.show()

    window.setGeometry(40, 20, 1000, 800)

    sys.exit(qApp.exec_())

