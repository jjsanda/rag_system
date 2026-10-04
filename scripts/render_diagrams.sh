#!/usr/bin/env bash
# Render every Mermaid diagram in docs/diagrams/ to SVG and PNG via mermaid-cli (mmdc).
# Uses npx so no global install is needed (Node.js required). The puppeteer config passes
# --no-sandbox so it works in CI/containers.
set -euo pipefail
cd "$(dirname "$0")/.."

DIR="docs/diagrams"
CONFIG="$DIR/mermaid-config.json"
PUPPETEER="$DIR/puppeteer-config.json"
MMDC=(npx -y -p @mermaid-js/mermaid-cli mmdc -c "$CONFIG" -p "$PUPPETEER")

shopt -s nullglob
for mmd in "$DIR"/*.mmd; do
    base="${mmd%.mmd}"
    echo "rendering $(basename "$mmd")"
    "${MMDC[@]}" -i "$mmd" -o "${base}.svg" -b transparent
    "${MMDC[@]}" -i "$mmd" -o "${base}.png" -b white -w 1400
done
echo "Done. SVG + PNG written to $DIR/"
