# Logs Directory

This directory stores application logs.

## Log Files

### app.log
Main application log file containing:
- Application startup/shutdown events
- API request logs
- Error messages
- Model loading status
- Database operations

## Log Format

```
2026-08-31 10:30:45,123 - deepfake_interceptor - INFO - Application startup complete
2026-08-31 10:30:46,456 - deepfake_interceptor - INFO - Loading deepfake detector
2026-08-31 10:30:50,789 - deepfake_interceptor - INFO - Analyzing audio: sample.wav
2026-08-31 10:30:51,234 - deepfake_interceptor - ERROR - Error loading audio: File not found
```

## Log Levels

| Level | Severity | Example |
|-------|----------|---------|
| DEBUG | Low | Variable assignments |
| INFO | Medium | Application events |
| WARNING | High | Configuration issues |
| ERROR | Critical | Failed operations |

## Viewing Logs

### Real-time (while running)
The logs are printed to console while the application runs.

### From file
```bash
# View entire log file
cat logs/app.log

# View last 100 lines
tail -100 logs/app.log

# Search for errors
grep ERROR logs/app.log

# Follow log in real-time
tail -f logs/app.log
```

### With Python
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Read logs
with open('logs/app.log') as f:
    for line in f:
        print(line.strip())
```

## Troubleshooting with Logs

### No logs generated
- Check if logs/ directory exists (created automatically)
- Verify write permissions
- Check `app/utils/logging.py` configuration

### Too many logs
- Adjust LOG_LEVEL in `.env`:
  ```
  LOG_LEVEL=WARNING  # Only show warnings and errors
  ```

### Finding errors
```bash
grep "ERROR\|Exception" logs/app.log
```

## Log Rotation

By default, logs grow continuously. To manage size:

```python
# In app/utils/logging.py, add:
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5            # Keep 5 rotated files
)
```

## Debug Mode

Enable detailed logging:

```bash
# In .env
LOG_LEVEL=DEBUG
DEBUG=True
```

Then run application and check `logs/app.log` for detailed trace.

---

**Created**: 2026-08-31
