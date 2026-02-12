import sys
import os
import subprocess
import glob
from datetime import datetime
from pathlib import Path

# Import PyQt6 components
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDateTimeEdit, QFrame, 
    QGroupBox, QGridLayout, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QDateTime, QThread, pyqtSignal
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')  # Set backend before importing pyplot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from data_viz_plot import main as plot_data_summary
from data_viz_plot import DATALAKE, STUDY_IDS

from prepare_files import sort_lfp, sort_cgx

# ============================================================================
# DATA PROCESSOR
# ============================================================================

class DataProcessor:
    """Handles all data processing and database operations"""
    
    def __init__(self, data_root_path=DATALAKE):
        self.data_root_path = Path(data_root_path)        
    
    def process_lfp_upload(self, patient_id, visit_start, visit_end):
        try:
            # Prompt user to select all LFP files, move to parser-ready directoy,
            lfp_dir = QFileDialog.getExistingDirectory(
                        None, "Select tablet Documents folder containing downloaded JSONs & PDFs")

            json_files = glob.glob(f'{lfp_dir}/*.json')
            pdf_files = glob.glob(f'{lfp_dir}/*.pdf')
            sort_lfp(json_files, pdf_files, patient_id)

            # Run LFP source parser
            result = subprocess.run(["/home/nbusleep/BCM/CODE/scripts/run_lfp_parser.sh"], capture_output=True, text=True)

            return {
                'records_uploaded': len(json_files) + len(pdf_files),
                'timestamp': datetime.now().isoformat(),
                'status': f'{result.returncode}'
            }
        except Exception as e:
            raise Exception(f"LFP upload failed: {str(e)}")
    
    def process_audio_video_upload(self, patient_id, visit_start, visit_end):
        try:
            # Run A/V source parser
            result = subprocess.run(["/home/nbusleep/BCM/CODE/scripts/run_av_parser.sh"], capture_output=True, text=True)

            return {
                'files_uploaded': 0, # Change this to read file log
                'timestamp': datetime.now().isoformat(),
                'status': f'{result.returncode}'
            }
        except Exception as e:
            raise Exception(f"Audio/Video upload failed: {str(e)}")
    
    def process_cgx_upload(self, patient_id, visit_start, visit_end):
        try:
            # Prompt user to select CGX drive, move to parser-ready directory
            cgx_dir = QFileDialog.getExistingDirectory(
                        None, "Select CGX SD card drive")
            cgx_files = glob.glob(f'{cgx_dir}/*.cgx')
            sort_cgx(cgx_files, patient_id)

            #  Run CGX source parser
            result = subprocess.run(["/home/nbusleep/BCM/CODE/scripts/run_cgx_parser.sh"], capture_output=True, text=True)

            return {
                'records_uploaded': len(cgx_files),
                'timestamp': datetime.now().isoformat(),
                'status': f'{result.returncode}'
            }
        except Exception as e:
            raise Exception(f"CGX upload failed: {str(e)}")


# ============================================================================
# WORKER THREAD
# ============================================================================

class UploadWorker(QThread):
    """Worker thread for handling data uploads"""
    finished = pyqtSignal(str, bool, dict)
    progress = pyqtSignal(str)
    
    def __init__(self, data_type, patient_id, visit_start, visit_end, processor):
        super().__init__()
        self.data_type = data_type
        self.patient_id = patient_id
        self.visit_start = visit_start
        self.visit_end = visit_end
        self.processor = processor
    
    def run(self):
        """Execute the upload"""
        try:
            self.progress.emit(f"Starting {self.data_type} upload...")
            
            # Call appropriate processor method
            method_map = {
                'lfp': self.processor.process_lfp_upload,
                'audio_video': self.processor.process_audio_video_upload,
                'cgx': self.processor.process_cgx_upload
            }
            
            result = method_map[self.data_type](
                self.patient_id, 
                self.visit_start, 
                self.visit_end
            )
            
            self.progress.emit(f"{self.data_type} upload complete!")
            self.finished.emit(self.data_type, True, result)
            
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            self.finished.emit(self.data_type, False, {'error': str(e)})


# ============================================================================
# VISUALIZATION WIDGET
# ============================================================================

class VisualizationWidget(QWidget):
    """Widget for data visualization"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        try:
            self.figure = Figure(figsize=(10, 7))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
            
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title('Data Visualization', fontsize=12, fontweight='bold')
            self.canvas.draw()
        except:
            label = QLabel("Matplotlib not installed.\nInstall with: pip install matplotlib")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #666; font-size: 14px;")
            layout.addWidget(label)
        
        self.setLayout(layout)
    
    def update_plot(self, upload_worker):
        """Update visualization with uploaded data"""
        self.ax.clear()
        
        self.ax = plot_data_summary(upload_worker.patient_id, upload_worker.visit_start, upload_worker.visit_end, self.ax)
        self.figure.tight_layout()
        self.canvas.draw()

        # Lab worlds folder TODO: Uncomment later
        #self.figure.savefig(Path('/mnt/projectworlds') / STUDY_IDS[upload_worker.patient_id[:-3]] / upload_worker.patient_id / 'NBU_visits' / f'{upload_worker.patient_id}_{upload_worker.visit_start}_visit_plot.pdf', bbox_inches='tight')


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class PatientDataUploadApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize data processor
        self.processor = DataProcessor(
            data_root_path=DATALAKE
        )
        
        self.uploaded_data = []
        self.upload_status = {
            'lfp': None,
            'audio_video': None,
            'cgx': None
        }
        self.workers = {}
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle('Patient Data Upload System')
        self.setMinimumSize(1000, 800)
        self.apply_stylesheet()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Add sections
        main_layout.addWidget(self.create_header())
        main_layout.addWidget(self.create_patient_info_section())
        main_layout.addWidget(self.create_upload_section())
        main_layout.addWidget(self.create_visualization_section(), stretch=1)
        
        self.statusBar().showMessage('Ready')
    
    def create_header(self):
        """Create header"""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        layout = QVBoxLayout(header)
        
        title = QLabel('NBU Data Upload Interface')
        title.setFont(QFont('Arial', 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #1E293B;")
        
        subtitle = QLabel('Upload NBU data streams to Datalake')
        subtitle.setFont(QFont('Arial', 12))
        subtitle.setStyleSheet("color: #64748B;")
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        return header
    
    def create_patient_info_section(self):
        """Create patient info section"""
        group = QGroupBox('Patient Information')
        group.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        group.setStyleSheet("""
            QGroupBox {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
                margin-top: 10px;
            }
            QGroupBox::title {
                color: #1E293B;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QGridLayout()
        layout.setSpacing(15)
        
        # Patient ID
        patient_id_label = QLabel('Patient ID:')
        self.patient_id_input = QLineEdit()
        self.patient_id_input.setPlaceholderText('Enter Patient ID')
        self.patient_id_input.setMinimumHeight(35)
        
        # Visit Start
        visit_start_label = QLabel('Visit Start:')
        self.visit_start_input = QDateTimeEdit()
        self.visit_start_input.setDateTime(QDateTime.currentDateTime())
        self.visit_start_input.setCalendarPopup(True)
        self.visit_start_input.setDisplayFormat('yyyy-MM-dd HH:mm:ss')
        self.visit_start_input.setMinimumHeight(35)
        
        # Visit End
        visit_end_label = QLabel('Visit End:')
        self.visit_end_input = QDateTimeEdit()
        self.visit_end_input.setDateTime(QDateTime.currentDateTime())
        self.visit_end_input.setCalendarPopup(True)
        self.visit_end_input.setDisplayFormat('yyyy-MM-dd HH:mm:ss')
        self.visit_end_input.setMinimumHeight(35)
        
        layout.addWidget(patient_id_label, 0, 0)
        layout.addWidget(self.patient_id_input, 1, 0)
        layout.addWidget(visit_start_label, 0, 1)
        layout.addWidget(self.visit_start_input, 1, 1)
        layout.addWidget(visit_end_label, 0, 2)
        layout.addWidget(self.visit_end_input, 1, 2)
        
        group.setLayout(layout)
        return group
    
    def create_upload_section(self):
        """Create upload buttons section"""
        group = QGroupBox('Data Stream Upload')
        group.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        group.setStyleSheet("""
            QGroupBox {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
                margin-top: 10px;
            }
            QGroupBox::title {
                color: #1E293B;
            }
        """)
        
        layout = QGridLayout()
        layout.setSpacing(10)
        
        self.upload_buttons = {
            'lfp': self.create_upload_button('Run LFP Source Parser', 'lfp'),
            'audio_video': self.create_upload_button('Run A/V Source Parser', 'audio_video'),
            'cgx': self.create_upload_button('Run CGX Source Parser', 'cgx')
        }
        
        layout.addWidget(self.upload_buttons['lfp'], 0, 0)
        layout.addWidget(self.upload_buttons['audio_video'], 0, 1)
        layout.addWidget(self.upload_buttons['cgx'], 0, 2)
        
        group.setLayout(layout)
        return group
    
    def create_upload_button(self, text, data_type):
        """Create upload button"""
        button = QPushButton(text)
        button.setMinimumHeight(50)
        button.setFont(QFont('Arial', 11))
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda: self.handle_upload(data_type))
        self.update_button_style(button, None)
        return button
    
    def create_visualization_section(self):
        """Create visualization section"""
        group = QGroupBox('Data Visualization')
        group.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        group.setStyleSheet("""
            QGroupBox {
                background-color: white;
                border-radius: 5px;
                padding: 5px;
                margin-top: 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Summary
        summary_layout = QHBoxLayout()
        self.summary_label = QLabel('Total uploads: 0')
        self.summary_label.setFont(QFont('Arial', 10))
        
        refresh_button = QPushButton('Refresh')
        refresh_button.setMaximumWidth(150)
        refresh_button.clicked.connect(self.refresh_visualization)
        refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2563EB; }
        """)
        
        summary_layout.addWidget(self.summary_label)
        summary_layout.addStretch()
        summary_layout.addWidget(refresh_button)
        
        layout.addLayout(summary_layout)
        
        self.viz_widget = VisualizationWidget()
        layout.addWidget(self.viz_widget)
        
        group.setLayout(layout)
        return group
    
    def update_button_style(self, button, status):
        """Update button appearance"""
        styles = {
            'uploading': "background-color: #60A5FA; color: white; border-radius: 8px; font-weight: bold; padding: 10px;",
            'success': "background-color: #10B981; color: white; border-radius: 8px; font-weight: bold; padding: 10px;",
            'error': "background-color: #EF4444; color: white; border-radius: 8px; font-weight: bold; padding: 10px;",
            None: "background-color: #3B82F6; color: white; border-radius: 8px; font-weight: bold; padding: 10px;"
        }
        button.setStyleSheet(f"QPushButton {{ {styles.get(status, styles[None])} }}")
        button.setEnabled(status != 'uploading')
    
    def handle_upload(self, data_type):
        """Handle upload"""
        patient_id = self.patient_id_input.text().strip()
        if not patient_id:
            QMessageBox.warning(self, 'Error', 'Please enter Patient ID')
            return
        
        visit_start = self.visit_start_input.dateTime().toString(Qt.DateFormat.ISODate)
        visit_end = self.visit_end_input.dateTime().toString(Qt.DateFormat.ISODate)
        
        button = self.upload_buttons[data_type]
        self.update_button_style(button, 'uploading')
        button.setText('Uploading...')
        
        worker = UploadWorker(data_type, patient_id, visit_start, visit_end, self.processor)
        worker.finished.connect(self.on_upload_finished)
        worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        
        self.workers[data_type] = worker
        worker.start()
    
    def on_upload_finished(self, data_type, success, result):
        """Handle upload completion"""
        button = self.upload_buttons[data_type]
        
        if success:
            self.update_button_style(button, 'success')
            button.setText(f'{data_type.replace("_", " ").title()} ✓')
            
            self.uploaded_data.append({
                'type': data_type,
                'timestamp': datetime.now().isoformat(),
                'patient_id': self.patient_id_input.text(),
                'details': result
            })
            
            self.summary_label.setText(f'Total uploads: {len(self.uploaded_data)}')
            self.viz_widget.update_plot(self.workers[data_type])
            self.statusBar().showMessage(f'{data_type} upload successful', 3000)
        else:
            self.update_button_style(button, 'error')
            button.setText(f'{data_type.replace("_", " ").title()} (Retry)')
            QMessageBox.critical(self, 'Error', f'Upload failed: {result.get("error")}')
    
    def refresh_visualization(self):
        """Refresh plot"""
        if len(self.uploaded_data) == 0:
            QMessageBox.warning(self, 'Error', 'Please upload data to refresh visualization')
            return
        
        self.viz_widget.update_plot(self.workers[list(self.workers.keys())[0]])
    
    def apply_stylesheet(self):
        """Apply global styles"""
        self.setStyleSheet("""
            QMainWindow { background-color: #F1F5F9; }
            QLineEdit, QDateTimeEdit {
                border: 1px solid #CBD5E1;
                border-radius: 5px;
                padding: 8px;
                background-color: white;
            }
            QLineEdit:focus, QDateTimeEdit:focus { border: 2px solid #3B82F6; }
        """)


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont('Arial', 10))
    
    window = PatientDataUploadApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
