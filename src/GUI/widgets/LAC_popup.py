import PyQt6.QtWidgets as qtw
import PyQt6.QtCore as qtc


class LACPopup(qtw.QDialog):
    inputs = qtc.pyqtSignal(str, str)  # Signal to emit the LAC and Image file paths    

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LAC Popup")
        self.setWindowFlags(
            qtc.Qt.WindowType.Window
            | qtc.Qt.WindowType.WindowStaysOnTopHint
            | qtc.Qt.WindowType.WindowCloseButtonHint
            | qtc.Qt.WindowType.WindowMinimizeButtonHint
        )
        self.setStyleSheet("""
            QWidget {
                font-size: 14px;
                font-weight: 500;
            }
            QToolButton {
                padding: 5px 10px;
                border: 1px solid #ACACAC;
                border-radius: 3px;
                background-color: #F0F0F0;
            }""")

        self._init_ui()
        self.connect_signals()
        self.resize(450,200)


    def _init_ui(self):
        layout = qtw.QVBoxLayout(self)
        layout.addWidget(qtw.QLabel("Lateral Artifact Correction .csv file"))
        sublayout = qtw.QHBoxLayout()
        self.LAC_input = qtw.QLineEdit()
        sublayout.addWidget(self.LAC_input)
        self.browse_LAC_btn = qtw.QPushButton("Browse")
        sublayout.addWidget(self.browse_LAC_btn)
        layout.addLayout(sublayout)

        layout.addWidget(qtw.QLabel("Image File"))
        sublayout2 = qtw.QHBoxLayout()
        self.Image_input = qtw.QLineEdit()
        sublayout2.addWidget(self.Image_input)
        self.browse_Image_btn = qtw.QPushButton("Browse")
        sublayout2.addWidget(self.browse_Image_btn)
        layout.addLayout(sublayout2)

        self.apply_btn = qtw.QPushButton("Apply")
        self.close_btn = qtw.QPushButton("Close")
        btn_layout = qtw.QHBoxLayout()
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.close_btn)
        layout.addLayout(btn_layout)
        
    def BrowseLACFile(self):
        file_dialog = qtw.QFileDialog(self)
        file_dialog.setNameFilter("CSV Files (*.csv)")
        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            self.LAC_input.setText(selected_file)

    def BrowseImageFile(self):
        file_dialog = qtw.QFileDialog(self)
        file_dialog.setNameFilter("Image Files (*.tif *.jpg *.jpeg *.bmp, *.tiff, *.png)")
        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            self.Image_input.setText(selected_file)

    def connect_signals(self):
        self.browse_LAC_btn.clicked.connect(self.BrowseLACFile)
        self.browse_Image_btn.clicked.connect(self.BrowseImageFile)
        self.close_btn.clicked.connect(self.close)
        self.apply_btn.clicked.connect(self.emit_inputs)

    def emit_inputs(self):
        lac_file = self.LAC_input.text()
        image_file = self.Image_input.text()
        self.inputs.emit(lac_file, image_file)