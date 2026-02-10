import sys
import os
from datetime import datetime
from pathlib import Path

# Import PySide6 components
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDateTimeEdit, QFrame, 
    QGroupBox, QGridLayout, QMessageBox
)
from PySide6.QtCore import Qt, QDateTime, QThread, Signal
from PySide6.QtGui import QFont

# Plotting imports - handle compatibility
MATLOTPLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Qt5Agg')  # Set backend before importing pyplot
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"Warning: Matplotlib not available. {str(e)}")

from data_viz_plot import main as plot_data_summary
from data_viz_plot import DATALAKE, ROOT

# ============================================================================
# DATA PROCESSOR
# ============================================================================

class DataProcessor:
    """Handles all data processing and database operations"""
    
    def __init__(self, data_root_path=DATALAKE):
        self.data_root_path = Path(data_root_path)        
    
    def process_lfp_upload(self, patient_id, visit_start, visit_end):
        """Upload LFP data - IMPLEMENT YOUR LOGIC HERE"""
        try:
            # TODO: Run LFP source parser
            
            return {
                'records_uploaded': 1000,
                'timestamp': datetime.now().isoformat(),
                'file_path': f'{self.data_root_path}/{patient_id}/lfp/',
                'status': 'success'
            }
        except Exception as e:
            raise Exception(f"LFP upload failed: {str(e)}")
    
    def process_audio_video_upload(self, patient_id, visit_start, visit_end):
        """Upload audio/video data - IMPLEMENT YOUR LOGIC HERE"""
        try:
            # TODO: Run A/V source parser
            return {
                'files_uploaded': 5,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        except Exception as e:
            raise Exception(f"Audio/Video upload failed: {str(e)}")
    
    def process_task_upload(self, patient_id, visit_start, visit_end):
        """Upload task data - IMPLEMENT YOUR LOGIC HERE"""
        try:
            # TODO: Run Task source parser
            return {
                'tasks_uploaded': 10,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        except Exception as e:
            raise Exception(f"Task upload failed: {str(e)}")
    
    def process_logger_upload(self, patient_id, visit_start, visit_end):
        """Upload logger data - IMPLEMENT YOUR LOGIC HERE"""
        try:
            # TODO: Run Logger source parser
            return {
                'logs_uploaded': 50,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        except Exception as e:
            raise Exception(f"Logger upload failed: {str(e)}")
    
    def process_cgx_upload(self, patient_id, visit_start, visit_end):
        """Upload CGX data - IMPLEMENT YOUR LOGIC HERE"""
        try:
            # TODO: Run CGX source parser
            return {
                'records_uploaded': 200,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        except Exception as e:
            raise Exception(f"CGX upload failed: {str(e)}")


# ============================================================================
# WORKER THREAD
# ============================================================================

class UploadWorker(QThread):
    """Worker thread for handling data uploads"""
    finished = Signal(str, bool, dict)
    progress = Signal(str)
    
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
                'task': self.processor.process_task_upload,
                'logger': self.processor.process_logger_upload,
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
        
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(15, 8))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
            
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title('Data Visualization', fontsize=12, fontweight='bold')
            self.canvas.draw()
        else:
            label = QLabel("Matplotlib not installed.\nInstall with: pip install matplotlib")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color: #666; font-size: 14px;")
            layout.addWidget(label)
        
        self.setLayout(layout)
    
    def update_plot(self, upload_worker):
        """Update visualization with uploaded data"""
        if not MATPLOTLIB_AVAILABLE or not upload_worker:
            return
        
        self.ax.clear()
        
        self.ax = plot_data_summary(upload_worker.patient_id, upload_worker.visit_start, upload_worker.visit_end, self.ax)
        self.figure.tight_layout()
        self.canvas.draw()


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
            'task': None,
            'logger': None,
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
                border-radius: 10px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout(header)
        
        title = QLabel('NBU Data Upload Interface')
        title.setFont(QFont('Arial', 24, QFont.Bold))
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
        group.setFont(QFont('Arial', 12, QFont.Bold))
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
        group.setFont(QFont('Arial', 12, QFont.Bold))
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
            'task': self.create_upload_button('Run Task Source Parser', 'task'),
            'logger': self.create_upload_button('Run Logger Source Parser', 'logger'),
            'cgx': self.create_upload_button('Upload CGX Data', 'cgx')
        }
        
        layout.addWidget(self.upload_buttons['lfp'], 0, 0)
        layout.addWidget(self.upload_buttons['audio_video'], 0, 1)
        layout.addWidget(self.upload_buttons['task'], 0, 2)
        layout.addWidget(self.upload_buttons['logger'], 1, 0)
        layout.addWidget(self.upload_buttons['cgx'], 1, 1)
        
        group.setLayout(layout)
        return group
    
    def create_upload_button(self, text, data_type):
        """Create upload button"""
        button = QPushButton(text)
        button.setMinimumHeight(50)
        button.setFont(QFont('Arial', 11))
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: self.handle_upload(data_type))
        self.update_button_style(button, None)
        return button
    
    def create_visualization_section(self):
        """Create visualization section"""
        group = QGroupBox('Data Visualization')
        group.setFont(QFont('Arial', 12, QFont.Bold))
        group.setStyleSheet("""
            QGroupBox {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
                margin-top: 10px;
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
        
        visit_start = self.visit_start_input.dateTime().toString(Qt.ISODate)
        visit_end = self.visit_end_input.dateTime().toString(Qt.ISODate)
        
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
