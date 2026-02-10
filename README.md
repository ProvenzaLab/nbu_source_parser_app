# Patient Data Upload Application - PySide6 Desktop Version

A complete Python desktop application for uploading clinical patient data streams to a database, built with PySide6.

## Project Files

```
patient_upload_complete.py         # Complete integrated application
patient_data_upload_pyside.py      # UI-only version (framework)
data_processing_examples.py        # Example data processing implementations
requirements_pyside.txt            # Python dependencies
```

## Prerequisites

- **Python 3.12+**

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_pyside.txt
```

Or install individually:

```bash
pip install PySide6 matplotlib pandas numpy sqlalchemy
```

### 2. Optional Database Drivers

Depending on your database:

```bash
# PostgreSQL
pip install psycopg2-binary

# MySQL
pip install pymysql

# MongoDB
pip install pymongo
```

## Quick Start

### Run the Complete Application

```bash
python patient_upload_complete.py
```

This runs the fully integrated application with UI and processing logic.

### Run the Framework Only

```bash
python patient_data_upload_pyside.py
```

This runs the UI framework where you can implement your own processing logic.

## Configuration

### Database Configuration

Edit the database URL in `patient_upload_complete.py`:

```python
self.processor = DataProcessor(
    data_root_path='/data/clinical',  # Your data directory
    database_url='sqlite:///patient_data.db'  # Your database
)
```

**Database URL Examples:**

```python
# SQLite (default - good for testing)
'sqlite:///patient_data.db'

# PostgreSQL
'postgresql://username:password@localhost:5432/patient_data'

# MySQL
'mysql+pymysql://username:password@localhost:3306/patient_data'

# SQL Server
'mssql+pyodbc://username:password@localhost/patient_data?driver=ODBC+Driver+17+for+SQL+Server'
```

### File Path Configuration

Update the `data_root_path` to match your data directory structure:

```python
DataProcessor(data_root_path='/path/to/your/data')
```

Expected directory structure:
```
/path/to/your/data/
├── P001/                    # Patient ID
│   ├── lfp/
│   │   └── 2024-01-15_lfp.csv
│   ├── audio_video/
│   │   └── 2024-01-15/
│   │       ├── video1.mp4
│   │       └── audio1.wav
│   ├── task/
│   ├── logger/
│   └── cgx/
├── P002/
...
```

## Implementation Guide

### Implementing Upload Logic

The main file to edit is `patient_upload_complete.py`. Look for `TODO` comments:

```python
def process_lfp_upload(self, patient_id, visit_start, visit_end):
    """Upload LFP data - IMPLEMENT YOUR LOGIC HERE"""
    try:
        # TODO: Replace with your actual implementation
        
        # 1. Construct file path
        visit_date = datetime.fromisoformat(visit_start).date()
        file_path = self.data_root_path / patient_id / 'lfp' / f'{visit_date}_lfp.csv'
        
        # 2. Read data
        df = pd.read_csv(file_path)
        
        # 3. Process data
        # ... your processing logic ...
        
        # 4. Upload to database
        df.to_sql('lfp_data', self.engine, if_exists='append', index=False)
        
        # 5. Return results
        return {
            'records_uploaded': len(df),
            'timestamp': datetime.now().isoformat(),
            'file_path': str(file_path)
        }
        
    except Exception as e:
        raise Exception(f"LFP upload failed: {str(e)}")
```

Repeat this pattern for each data type:
- `process_lfp_upload()`
- `process_audio_video_upload()`
- `process_task_upload()`
- `process_logger_upload()`
- `process_cgx_upload()`

### Example Implementations

The file `data_processing_examples.py` contains detailed examples of:
- Database setup with SQLAlchemy
- Reading various file formats
- Data validation and filtering
- Database uploads
- Visualization data generation

You can copy these implementations into `patient_upload_complete.py`.

## Usage

### 1. Launch Application

```bash
python patient_upload_complete.py
```

### 2. Enter Patient Information

- Enter the Patient ID
- Select visit start date/time
- Select visit end date/time

### 3. Upload Data

Click the upload buttons for each data stream you want to upload:
- **Upload LFP Data**
- **Upload Audio/Video Data**
- **Upload Task Data**
- **Upload Logger Data**
- **Upload CGX Data**

### 4. Monitor Progress

- Button colors indicate status:
  - **Blue**: Ready to upload
  - **Light Blue**: Uploading in progress
  - **Green**: Upload successful
  - **Red**: Upload failed (click to retry)

- Status bar shows current operation
- Total upload count updates automatically

### 5. View Visualization

The visualization plot updates automatically after each successful upload, showing:
- Upload counts by data type
- Bar chart visualization

Click "Refresh" to manually update the visualization.

## Advanced Features

### Multi-threaded Uploads

The application uses `QThread` to run uploads in background threads, keeping the UI responsive:

```python
class UploadWorker(QThread):
    """Worker thread for handling data uploads"""
    finished = Signal(str, bool, dict)
    progress = Signal(str)
    
    def run(self):
        # Upload logic runs here without blocking UI
        ...
```

### Custom Visualization

To customize the visualization, edit `VisualizationWidget.update_plot()`:

```python
def update_plot(self, uploaded_data):
    # Your custom plotting logic
    self.ax.clear()
    
    # Example: Time series plot
    timestamps = [item['timestamp'] for item in uploaded_data]
    values = [item['details'].get('records_uploaded', 0) for item in uploaded_data]
    self.ax.plot(timestamps, values, marker='o')
    
    self.canvas.draw()
```

### Database Models

Add custom database tables by extending the `Base` class:

```python
class CustomData(Base):
    __tablename__ = 'custom_data'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(String(50))
    # ... your columns ...
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'PySide6'"

```bash
pip install PySide6
```

### "ModuleNotFoundError: No module named 'matplotlib'"

```bash
pip install matplotlib
```

### Database Connection Errors

- Verify your database is running
- Check database URL credentials
- Ensure database driver is installed (psycopg2, pymysql, etc.)

### File Not Found Errors

- Verify `data_root_path` is correct
- Check file/directory permissions
- Ensure data files exist in expected locations

### UI Not Responding

- This shouldn't happen due to multi-threading
- If it does, check for infinite loops in processing functions
- Ensure exceptions are properly caught and raised

## Packaging as Standalone Application

### Using PyInstaller

```bash
pip install pyinstaller

pyinstaller --onefile --windowed patient_upload_complete.py
```

The executable will be in the `dist/` directory.

### Using cx_Freeze

```bash
pip install cx_Freeze

python setup.py build
```

## Development Tips

### Testing Upload Functions

Test individual upload functions before integrating:

```python
if __name__ == '__main__':
    processor = DataProcessor(
        data_root_path='/data/clinical',
        database_url='sqlite:///test.db'
    )
    
    result = processor.process_lfp_upload(
        patient_id='TEST001',
        visit_start='2024-01-15T09:00:00',
        visit_end='2024-01-15T12:00:00'
    )
    
    print(result)
```

### Logging

Add logging for debugging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='upload.log'
)

# In your functions
logging.info(f"Processing upload for patient {patient_id}")
```

### Error Handling

Always use try-except blocks and provide meaningful error messages:

```python
try:
    # Upload logic
    ...
except FileNotFoundError as e:
    raise Exception(f"Data file not found: {str(e)}")
except pd.errors.EmptyDataError:
    raise Exception("Data file is empty")
except Exception as e:
    raise Exception(f"Unexpected error: {str(e)}")
```

## Security Considerations

1. **Database Credentials**: Use environment variables
   ```python
   import os
   DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///default.db')
   ```

2. **Input Validation**: Validate patient IDs and dates
   ```python
   if not patient_id.isalnum():
       raise ValueError("Invalid patient ID format")
   ```

3. **HIPAA Compliance**: Ensure PHI is encrypted at rest and in transit

4. **Access Control**: Add user authentication if needed

## Performance Optimization

### Large File Handling

For large files, process in chunks:

```python
chunk_size = 10000
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    chunk.to_sql('lfp_data', engine, if_exists='append', index=False)
```

### Database Optimization

Use batch inserts:

```python
from sqlalchemy import insert

records = [dict(zip(df.columns, row)) for row in df.values]
stmt = insert(LFPData).values(records)
session.execute(stmt)
session.commit()
```

## License

[Your License Here]

## Support

For issues or questions:
1. Check this README
2. Review `data_processing_examples.py` for implementation guidance
3. Check PySide6 documentation: https://doc.qt.io/qtforpython/
4. Check SQLAlchemy documentation: https://docs.sqlalchemy.org/
# nbu_source_parser_app
