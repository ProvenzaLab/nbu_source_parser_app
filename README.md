
## Prerequisites

- **Python 3.12+**

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_pyside.txt
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

