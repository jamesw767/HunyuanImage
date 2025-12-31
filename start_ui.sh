#!/bin/bash
# HunyuanImage-3.0 Web UI Launcher
# Handles detection of running instances and provides clear status

PROJECT_DIR="/media/james/DataDrive/jamesw767/Hun3d"
VENV_DIR="$PROJECT_DIR/hunyuan_env"
PID_FILE="$PROJECT_DIR/.hunyuan_ui.pid"
LOG_FILE="$PROJECT_DIR/hunyuan_ui.log"
PORT=7860

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   HunyuanImage-3.0 Image Generator${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Function to check if already running
check_running() {
    # Check by PID file
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0  # Running
        else
            rm -f "$PID_FILE"  # Stale PID file
        fi
    fi

    # Check by port
    if lsof -i :$PORT > /dev/null 2>&1; then
        return 0  # Port in use
    fi

    return 1  # Not running
}

# Function to get the PID using the port
get_pid_by_port() {
    lsof -ti :$PORT 2>/dev/null | head -1
}

# Check if already running
if check_running; then
    echo -e "${YELLOW}HunyuanImage-3.0 UI is already running!${NC}"
    echo ""
    echo "Options:"
    echo "  1) Open in browser (already running)"
    echo "  2) Stop the running instance"
    echo "  3) Cancel"
    echo ""
    read -p "Choose [1/2/3]: " choice

    case $choice in
        1)
            echo -e "${GREEN}Opening browser...${NC}"
            xdg-open "http://localhost:$PORT" 2>/dev/null &
            echo ""
            echo -e "${GREEN}UI is running at: http://localhost:$PORT${NC}"
            echo "Press Enter to close this window..."
            read
            exit 0
            ;;
        2)
            echo -e "${YELLOW}Stopping HunyuanImage-3.0 UI...${NC}"
            PID=$(get_pid_by_port)
            if [ -n "$PID" ]; then
                kill "$PID" 2>/dev/null
                sleep 2
                if ps -p "$PID" > /dev/null 2>&1; then
                    kill -9 "$PID" 2>/dev/null
                fi
            fi
            rm -f "$PID_FILE"
            echo -e "${GREEN}Stopped.${NC}"
            echo ""
            echo "Do you want to restart? [y/N]: "
            read restart
            if [[ ! "$restart" =~ ^[Yy]$ ]]; then
                exit 0
            fi
            ;;
        *)
            echo "Cancelled."
            exit 0
            ;;
    esac
fi

# Check if virtual environment exists
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo -e "${RED}ERROR: Virtual environment not found at:${NC}"
    echo "  $VENV_DIR"
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Set environment variables
# Use GPU 1 = RTX PRO 6000 Blackwell (96GB)
# Note: nvidia-smi shows Blackwell as GPU 0, but CUDA/PyTorch ordering is reversed
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PROJECT_DIR/HunyuanImage-3.0:$PYTHONPATH"

# Change to project directory
cd "$PROJECT_DIR"

# Check GPU
echo -e "${BLUE}Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        echo -e "  GPU: ${GREEN}$GPU_NAME ($GPU_MEM)${NC}"
    else
        echo -e "  ${YELLOW}Warning: Could not detect GPU${NC}"
    fi
else
    echo -e "  ${YELLOW}Warning: nvidia-smi not found${NC}"
fi

echo ""
echo -e "${GREEN}Starting HunyuanImage-3.0 UI...${NC}"
echo -e "URL: ${GREEN}http://localhost:$PORT${NC}"
echo ""
echo -e "${YELLOW}Note: First load takes 1-2 minutes to load the model.${NC}"
echo -e "Press Ctrl+C to stop the server."
echo ""
echo "----------------------------------------"

# Save PID and run
python hunyuan_ui.py &
UI_PID=$!
echo $UI_PID > "$PID_FILE"

# Wait for the process
wait $UI_PID
EXIT_CODE=$?

# Cleanup
rm -f "$PID_FILE"

echo ""
echo "----------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}UI stopped normally.${NC}"
else
    echo -e "${RED}UI exited with error code: $EXIT_CODE${NC}"
    echo ""
    echo "Check the error messages above for details."
fi

echo ""
echo "Press Enter to close this window..."
read
