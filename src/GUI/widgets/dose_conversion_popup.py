import PyQt6.QtWidgets as qtw
import PyQt6.QtCore as qtc


class CalPopup(qtw.QDialog):
    inputs = qtc.pyqtSignal(str, str, str, bool, bool)  # Signal to emit the Cal and Image file paths    

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cal Popup")
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
        layout.addWidget(qtw.QLabel("Calibration.csv file"))
        sublayout = qtw.QHBoxLayout()
        self.Cal_input = qtw.QLineEdit()
        sublayout.addWidget(self.Cal_input)
        self.browse_Cal_btn = qtw.QPushButton("Browse")
        sublayout.addWidget(self.browse_Cal_btn)
        layout.addLayout(sublayout)

        layout.addWidget(qtw.QLabel("Dose File"))
        sublayout2 = qtw.QHBoxLayout()
        self.Image_input = qtw.QLineEdit()
        sublayout2.addWidget(self.Image_input)
        self.browse_Image_btn = qtw.QPushButton("Browse")
        sublayout2.addWidget(self.browse_Image_btn)
        layout.addLayout(sublayout2)

        layout.addWidget(qtw.QLabel('Select Channels To Use For Dose Conversion'))
        sublayout3 = qtw.QHBoxLayout()
        self.channel_radio_group = qtw.QButtonGroup(self)
        labels = ["Multichannel", "Red", "Green", "Blue"]
        
        for index, text in enumerate(labels):
            radio_button = qtw.QRadioButton(text)
            sublayout3.addWidget(radio_button)
            self.channel_radio_group.addButton(radio_button, index)
            
        self.channel_radio_group.button(0).setChecked(True)
        layout.addLayout(sublayout3)

        layout.addWidget(qtw.QLabel('Select Outputs'))
        sublayout4 = qtw.QHBoxLayout()
        self.dicom_checkbox = qtw.QCheckBox("Save as DICOM")
        sublayout4.addWidget(self.dicom_checkbox)
        self.csv_checkbox = qtw.QCheckBox("Save as CSV")
        sublayout4.addWidget(self.csv_checkbox)
        layout.addLayout(sublayout4)

        self.apply_btn = qtw.QPushButton("Apply")
        self.close_btn = qtw.QPushButton("Close")
        btn_layout = qtw.QHBoxLayout()
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.close_btn)
        layout.addLayout(btn_layout)
        
    def BrowseCalFile(self):
        file_dialog = qtw.QFileDialog(self)
        file_dialog.setNameFilter("CSV Files (*.csv)")
        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            self.Cal_input.setText(selected_file)

    def BrowseImageFile(self):
        file_dialog = qtw.QFileDialog(self)
        file_dialog.setNameFilter("Dose Files (*.tif *.jpg *.jpeg *.bmp, *.tiff, *.png)")
        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            self.Image_input.setText(selected_file)

    def connect_signals(self):
        self.browse_Cal_btn.clicked.connect(self.BrowseCalFile)
        self.browse_Image_btn.clicked.connect(self.BrowseImageFile)
        self.close_btn.clicked.connect(self.close)
        self.apply_btn.clicked.connect(self.emit_inputs)

    def emit_inputs(self):
        Cal_file = self.Cal_input.text()
        image_file = self.Image_input.text()
        channel_id = self.channel_radio_group.checkedId()
        channel = self.channel_radio_group.button(channel_id).text()[0].lower()
        self.inputs.emit(Cal_file, image_file, channel, self.dicom_checkbox.isChecked(), self.csv_checkbox.isChecked())