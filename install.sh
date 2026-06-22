#!/bin/bash
set -e

BINDIR="$HOME/.local/bin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$BINDIR"

TARGET="$BINDIR/aether"
if [ -L "$TARGET" ] || [ -f "$TARGET" ]; then
    echo "Removing existing $TARGET"
    rm -f "$TARGET"
fi

ln -s "$SCRIPT_DIR/bin/aether" "$TARGET"
chmod +x "$SCRIPT_DIR/bin/aether"

echo "Installed aether → $TARGET"

if ! echo "$PATH" | tr ':' '\n' | grep -qxF "$BINDIR"; then
    echo ""
    echo "⚠️  $BINDIR is not in your PATH."
    echo "   Add this to your ~/.zshrc or ~/.bashrc:"
    echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo ""
echo "Usage: aether [task]"
