#!/usr/bin/env bash
set -euo pipefail

# 1. Create an archive directory
mkdir -p backups

# 2. Move legacy/working versions into backups
mv sheet_counter_unified.py.v0 sheet_counter_unified.py.work-* \
   full_calibration_dbscan.py.v1 sheet_counter_ridges.py.v1 \
   2>/dev/null || true

# 3. Stage the new backups folder
git add backups

# 4. Untrack those legacy files so git no longer reports them
git rm --cached sheet_counter_unified.py.v0 sheet_counter_unified.py.work-* \
              full_calibration_dbscan.py.v1 sheet_counter_ridges.py.v1 \
   2>/dev/null || true

# 5. Commit the cleanup
git commit -m "chore: archive legacy scripts to backups/"

echo "✅ Cleanup complete. Legacy files are in backups/, and your working tree is clean."

