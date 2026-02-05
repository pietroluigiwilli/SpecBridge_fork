#!/bin/bash
# Download SpecBridge files from Zenodo
# Downloads checkpoints to runs/ directories and data files to data/ directory

set -e

# Configuration
ZENODO_RECORD="18357418"
ZENODO_URL="https://zenodo.org/api/records/${ZENODO_RECORD}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================================"
echo "SpecBridge Zenodo Download Script"
echo "============================================================"
echo ""

# File mapping: "zenodo_filename|local_path"
# Keep Zenodo filenames, use simple run directory names
declare -A FILE_MAP=(
    # Checkpoints -> runs/ directories (simple names)
    ["SpecBridge_MSGYM_checkpoint.pt"]="runs/msgym/SpecBridge_MSGYM_checkpoint.pt"
    ["SpecBridge_MSnLib_checkpoint.pt"]="runs/msnlib/SpecBridge_MSnLib_checkpoint.pt"
    ["SpecBridge_Spectraverse_checkpoint.pt"]="runs/spectraverse/SpecBridge_Spectraverse_checkpoint.pt"
    
    # Datasets -> data/ directory (keep Zenodo names)
    ["SpecBridge_MassSpecGym_dataset.mgf"]="data/SpecBridge_MassSpecGym_dataset.mgf"
    ["SpecBridge_MSnLib_dataset.mgf"]="data/SpecBridge_MSnLib_dataset.mgf"
    ["SpecBridge_Spectraverse_dataset.mgf"]="data/SpecBridge_Spectraverse_dataset.mgf"
    
    # Candidates -> data/ directory (keep Zenodo names)
    ["SpecBridge_MSGYM_candidates.pkl"]="data/SpecBridge_MSGYM_candidates.pkl"
    ["SpecBridge_MSnLib_candidates.pkl"]="data/SpecBridge_MSnLib_candidates.pkl"
    ["SpecBridge_Spectraverse_candidates.pkl"]="data/SpecBridge_Spectraverse_candidates.pkl"
)

# Determine which Python to use
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=python
else
    echo "❌ Failed to find a Python interpreter (python3 or python)."
    exit 1
fi

# Get file list from Zenodo
echo "📦 Fetching file list from Zenodo..."
FILES_JSON=$(curl -s "${ZENODO_URL}")

if [ -z "$FILES_JSON" ]; then
    echo "❌ Failed to fetch Zenodo record. Check your internet connection."
    exit 1
fi

# Extract file URLs and names
echo "✓ Connected to Zenodo record: ${ZENODO_RECORD}"
echo ""

# Download each file
for zenodo_name in "${!FILE_MAP[@]}"; do
    local_path="${FILE_MAP[$zenodo_name]}"
    
    # Extract download URL for this file
    download_url=$(echo "$FILES_JSON" | "$PYTHON_CMD" -c "
import sys, json
data = json.load(sys.stdin)
for f in data.get('files', []):
    if f.get('key') == '${zenodo_name}':
        print(f['links']['self'])
        break
" 2>/dev/null)
    
    if [ -z "$download_url" ]; then
        echo -e "${YELLOW}⚠️  File not found on Zenodo: ${zenodo_name}${NC}"
        continue
    fi
    
    # Create directory if needed
    local_dir=$(dirname "$local_path")
    if [ ! -d "$local_dir" ]; then
        echo "📁 Creating directory: $local_dir"
        mkdir -p "$local_dir"
    fi
    
    # Check if file already exists
    if [ -f "$local_path" ]; then
        echo -e "${BLUE}⊘ Skipping: ${zenodo_name} (already exists at ${local_path})${NC}"
        echo ""
        continue
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}📥 Downloading: ${zenodo_name}${NC}"
    echo "   → ${local_path}"
    
    # Get file size for progress
    file_size=$(echo "$FILES_JSON" | "$PYTHON_CMD" -c "
import sys, json
data = json.load(sys.stdin)
for f in data.get('files', []):
    if f.get('key') == '${zenodo_name}':
        print(f.get('size', 0))
        break
" 2>/dev/null)
    
    if [ -n "$file_size" ] && [ "$file_size" != "0" ]; then
        size_mb=$((file_size / 1024 / 1024))
        echo "   Size: ${size_mb} MB"
    fi
    echo ""
    
    # Download file
    if curl -L -o "$local_path" --progress-bar "$download_url"; then
        echo -e "${GREEN}✓ Downloaded: ${zenodo_name} → ${local_path}${NC}"
    else
        echo -e "${YELLOW}❌ Failed to download: ${zenodo_name}${NC}"
        # Remove partial file
        [ -f "$local_path" ] && rm "$local_path"
    fi
    echo ""
done

echo "============================================================"
echo -e "${GREEN}✅ Download complete!${NC}"
echo ""
echo "Files downloaded to:"
echo "  • Checkpoints: runs/*/ckpt_*.pt"
echo "  • Datasets: data/*.mgf"
echo "  • Candidates: data/*.pkl"
echo "============================================================"
