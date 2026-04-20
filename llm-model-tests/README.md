# LLM Model Tests

This directory serves as the dedicated sandbox for running standalone tests against **Alpamayo** inference pipelines, extracting specific route tests, and managing the associated route frame database.

## Database Management

An integrated SQLite database (`annotations.db`) is used for managing, filtering, and querying specifically extracted route dashcam frames prior to Alpamayo ingestion.

### Interacting with the CLI:
```bash
python import_annotations.py  # Bulk import route frame extractions into DB
```

### Programmatic Python Access:
```python
from dataset_manager import DatasetManager

# Initialize the manager
db = DatasetManager()

# Example: Push a new telemetry-bound frame into records
db.add_frame("frame.jpg", "frames/frame.jpg")
```

## Alpamayo Evaluation Utilities

```bash
./setup_alpamayo.sh  # Establish the virtual environment wrapper 

# Standardized unit testing and basic execution checks over the model tensor operations 
python test_alpamayo.py

# Graph visualization rendering wrapper for model trajectory confidence bounds
python visualize_alpamayo.py
```
