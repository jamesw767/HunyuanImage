#!/bin/bash
# Quick status check for HunyuanImage-3.0 UI

PORT=7860
PID_FILE="/media/james/DataDrive/jamesw767/Hun3d/.hunyuan_ui.pid"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "HunyuanImage-3.0 Status"
echo "======================="

# Check by port
if lsof -i :$PORT > /dev/null 2>&1; then
    PID=$(lsof -ti :$PORT 2>/dev/null | head -1)
    echo -e "Status: ${GREEN}RUNNING${NC}"
    echo -e "PID: $PID"
    echo -e "URL: ${GREEN}http://localhost:$PORT${NC}"
    echo ""
    echo "To open in browser: xdg-open http://localhost:$PORT"
    echo "To stop: kill $PID"
else
    echo -e "Status: ${RED}NOT RUNNING${NC}"
    echo ""
    echo "To start: ./start_ui.sh"
    echo "Or double-click the desktop icon"
fi

echo ""
