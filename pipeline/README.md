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
