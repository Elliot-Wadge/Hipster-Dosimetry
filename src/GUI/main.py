import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QToolButton,
    QMenu,
    QMessageBox
)
import PyQt6.QtWidgets as qtw
from GUI.widgets.LAC_popup import LACPopup
from GUI.widgets.dose_conversion_popup import CalPopup
from HipsterDosimetry.lateral_artifact import apply_LA_correction as apply_LA
from HipsterDosimetry.util import combine_tif_images, convert_image, save_dose_to_rtdose
from pathlib import Path
import skimage as ski
import numpy as np

class FilmHDToolbar(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Film HD")
        
        # Keep floating on top and enable compact window frame
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )

        # 1. Apply global font scale and padding via Qt Style Sheet
        # Adjust 'font-size' and 'padding' here to scale the entire toolbar up or down
        self.setStyleSheet("""
            QWidget {
                font-size: 14px;
                font-weight: 500;
            }
            QToolButton {
                padding: 5px 10px;
                border: 1px solid #ACACAC;
                border-radius: 3px;
                background-color: #4a4444;
            }
            QToolButton:hover {
                background-color: #b5b5b3;
            }
            QToolButton:pressed {
                background-color: #D0D0D0;
            }
            QToolButton::menu-indicator {
                image: none; /* Removes the small dropdown arrow to save space */
            }
        """)

        self._init_ui()
        self.adjustSize()
        self.connect_signals()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Lock layout tightly around child widgets
        layout.setSizeConstraint(QHBoxLayout.SizeConstraint.SetFixedSize)

        # File Dropdown Button (Replaces top menu bar)
        self.file_btn = QToolButton(self)
        self.file_btn.setText("File")
        self.file_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        
        file_menu = QMenu(self.file_btn)
        exit_action = file_menu.addAction("&Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        combine_action = file_menu.addAction("&Combine Images")
        combine_action.triggered.connect(self._open_Combine_Images_popup)

        about_action = file_menu.addAction("&About")
        about_action.triggered.connect(self._show_about_dialog)
        
        self.file_btn.setMenu(file_menu)
        layout.addWidget(self.file_btn)

        # Action Tool Buttons
        self.LAC_btn = self._create_tool_button("Apply LAC", "")
        self.Convert2Dose_btn = self._create_tool_button("Convert to Dose", "")
        self.gamma_btn = self._create_tool_button("Gamma", "")

        layout.addWidget(self.LAC_btn)
        layout.addWidget(self.Convert2Dose_btn)
        layout.addWidget(self.gamma_btn)

    def _create_tool_button(self, text: str, tooltip: str) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(text)
        btn.setToolTip(tooltip)
        return btn

    def _show_about_dialog(self):
        QMessageBox.about(
            self,
            "About",
            "Film HD Application\nVersion 1.0"
        )

    def connect_signals(self):
        self.LAC_btn.clicked.connect(self._open_LAC_popup)
        self.Convert2Dose_btn.clicked.connect(self._open_Convert2Dose_popup)
        # self.gamma_btn.clicked.connect(self._open_Gamma_popup)

    def _open_LAC_popup(self):
        LAC_popup = LACPopup(self)
        LAC_popup.inputs.connect(self._handle_LAC_inputs)
        LAC_popup.show()

    def _handle_LAC_inputs(self, lac_file, image_file):
        try:
            corrected_image = apply_LA(image_file, lac_file)
            save_path = Path(image_file).parent / f"{Path(image_file).stem}_LAC_corrected.tif"
            ski.io.imsave(save_path, corrected_image.astype(np.uint16))
            QMessageBox.information(self, "Success", f"Lateral Artifact Correction applied successfully.\nSaved to: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply LAC: {str(e)}")

    def _open_Convert2Dose_popup(self):
        Convert2Dose_popup = CalPopup(self)
        Convert2Dose_popup.inputs.connect(self._handle_Convert2Dose_inputs)
        Convert2Dose_popup.show()

    def _handle_Convert2Dose_inputs(self, cal_file, image_file, save_as_dicom, save_as_csv):
        try:
            dose, delta, od = convert_image(image_file, cal_file)
            save_path0 = Path(image_file).parent / f"{Path(image_file).stem}_dose.tif"
            save_path1 = Path(image_file).parent / f"{Path(image_file).stem}_delta.tif"
            save_path2 = Path(image_file).parent / f"{Path(image_file).stem}_dose.csv"
            ski.io.imsave(save_path0, dose[:,:,0].astype(np.float32))
            ski.io.imsave(save_path1, delta.astype(np.float32))
            if save_as_csv:
                np.savetxt(save_path2, dose[:,:,0], delimiter=',')
            if save_as_dicom:
                save_dose_to_rtdose(dose[:,:,0], save_path0.with_suffix('.dcm'))
            QMessageBox.information(self, "Success", "Image converted to dose successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to convert image to dose: {str(e)}")

    def _open_Combine_Images_popup(self):
        # Placeholder for Combine Images functionality
        directory = qtw.QFileDialog.getExistingDirectory(self, "Select Directory Containing Images")
        if directory:
            try:
                combine_tif_images(directory)
                qtw.QMessageBox.information(self, "Success", "Images combined successfully.")
            except Exception as e:
                qtw.QMessageBox.critical(self, "Error", f"Failed to combine images: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = FilmHDToolbar()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()