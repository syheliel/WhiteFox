#!/bin/bash

# Check if directory parameter is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <directory_path>"
    echo "Example: $0 ./rag_exp"
    exit 1
fi

BASE_DIR="$1"

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory '$BASE_DIR' does not exist."
    exit 1
fi

# Create a temporary Python wrapper script
cat > /tmp/run_scripts.py << 'EOF'
import sys
import importlib.util
import os

def run_script(script_path):
    try:
        # Get the module name from the file path
        module_name = os.path.basename(script_path).replace('.py', '')
        
        # Load the module spec
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None:
            return False
            
        # Create the module
        module = importlib.util.module_from_spec(spec)
        
        # Execute the module
        spec.loader.exec_module(module)
        return True
    except Exception as e:
        print(f"Error running {script_path}: {e}", file=sys.stderr) # Print errors for debugging
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_scripts.py <script_paths...>")
        return
    
    success_count = 0
    total_count = len(sys.argv[1:])
    
    for script_path in sys.argv[1:]:
        if run_script(script_path):
            print(f"  Success: {os.path.basename(script_path)}")
            success_count += 1
        else:
            print(f"  Failed: {os.path.basename(script_path)}")
    
    print(f"RESULTS: {success_count}/{total_count}")
    
if __name__ == "__main__":
    main()
EOF

# Initialize result storage variables
declare -A results_total
declare -A results_success

# Process each subdirectory
for subdir in "with_rag" "without_rag"; do
    DIR_PATH="$BASE_DIR/$subdir"
    
    # Check if the subdirectory exists
    if [ ! -d "$DIR_PATH" ]; then
        echo "Warning: Subdirectory '$subdir' does not exist in '$BASE_DIR'. Skipping."
        results_total["$subdir"]=-1 # Mark as skipped
        results_success["$subdir"]=-1
        continue
    fi
    
    echo "Processing directory: $DIR_PATH"
    
    # Find all Python scripts
    PYTHON_SCRIPTS=()
    while IFS= read -r script; do
        PYTHON_SCRIPTS+=("$script")
    done < <(find "$DIR_PATH" -name "*.py" -type f)
    
    total_files=${#PYTHON_SCRIPTS[@]}
    results_total["$subdir"]=$total_files
    
    if [ $total_files -eq 0 ]; then
        echo "No Python scripts found in $DIR_PATH"
        results_success["$subdir"]=0
        continue
    fi
    
    echo "Found $total_files Python scripts"
    
    # Run all scripts through the wrapper
    echo "Running all Python scripts for $subdir..."
    OUTPUT=$(python /tmp/run_scripts.py "${PYTHON_SCRIPTS[@]}")
    
    # Parse results
    echo "$OUTPUT" # Show individual script results
    SUCCESS_COUNT=$(echo "$OUTPUT" | grep "RESULTS:" | cut -d':' -f2 | cut -d'/' -f1 | tr -d ' ' | grep '^[0-9]*$' || echo 0) # Ensure SUCCESS_COUNT is a number, default to 0 if parse fails
    results_success["$subdir"]=$SUCCESS_COUNT
    
    # Report results (moved to end)
    # echo "Results for $subdir:"
    # echo "  Total Python scripts: $total_files"
    # echo "  Successful runs: $SUCCESS_COUNT"
    # if [ $total_files -gt 0 ]; then
    #    success_rate=$(( (SUCCESS_COUNT * 100) / total_files ))
    # else
    #    success_rate=0
    # fi
    # echo "  Success rate: $success_rate%"
    # echo "----------------------------------------"
done

# Report final results
echo "--- Final Evaluation Results ---"
for subdir in "with_rag" "without_rag"; do
    total_files=${results_total["$subdir"]}
    success_count=${results_success["$subdir"]}

    echo "Results for $subdir:"
    if [ "$total_files" -eq -1 ]; then
        echo "  Skipped (directory not found)."
    elif [ "$total_files" -eq 0 ]; then
        echo "  Total Python scripts: 0"
        echo "  (No scripts found)"
    else
        echo "  Total Python scripts: $total_files"
        echo "  Successful runs: $success_count"
        if [ $total_files -gt 0 ]; then
            success_rate=$(( (success_count * 100) / total_files ))
        else
            success_rate=0
        fi
        echo "  Success rate: $success_rate%"
    fi
    echo "----------------------------------------"
done

# Clean up
rm -f /tmp/run_scripts.py
echo "Evaluation complete."
