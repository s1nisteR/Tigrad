#!/usr/bin/env bash
# Render a .dot file to a graphviz image format.
# Usage: ./render_dot.sh <input.dot> [format]
# Supported formats: png, svg, pdf, jpg

set -euo pipefail

FORMATS="png svg pdf jpg"

usage() {
    echo "Usage: $0 <input.dot> [format]"
    echo ""
    echo "Formats: $FORMATS"
    echo "Default: png"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

INPUT="$1"
FORMAT="${2:-png}"

if [ ! -f "$INPUT" ]; then
    echo "Error: file '$INPUT' not found."
    exit 1
fi

# Validate format
if ! echo "$FORMATS" | grep -qw "$FORMAT"; then
    echo "Error: unsupported format '$FORMAT'"
    echo "Choose from: $FORMATS"
    exit 1
fi

OUTPUT="${INPUT%.dot}.$FORMAT"

if ! command -v dot &>/dev/null; then
    echo "Error: 'dot' (graphviz) is not installed."
    echo "Install with: brew install graphviz"
    exit 1
fi

dot -T"$FORMAT" "$INPUT" -o "$OUTPUT"
echo "Rendered: $OUTPUT"
