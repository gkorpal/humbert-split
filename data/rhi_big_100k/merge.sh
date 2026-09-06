#!/usr/bin/env bash
#
# Rebuild the split .txt datasets in this directory from their pieces.
#
#   ./merge.sh            # merge everything (~1.6 GB of output)
#   ./merge.sh 1000bit    # only files whose name contains a substring
#
# Unlike polz_big_100k, only part of this directory is split: the 11 files from
# 500bit up run 106-196 MB, over GitHub's 100 MB per-blob limit, so each is
# stored as a sequence of line-aligned pieces:
#
#   RHI2_1000bit_faa1c19f_100000.txt.part00
#   RHI2_1000bit_faa1c19f_100000.txt.part01
#   RHI2_1000bit_faa1c19f_100000.txt.part02
#
# The nine smaller files (50bit-450bit, 24-97 MB) are committed whole and are
# already present after a clone -- this script does not touch them.
#
# The pieces are plain, uncompressed text, so `cat` is all it takes -- this
# script just does it for every split file, in order, and verifies the result:
#
#   cat RHI2_1000bit_faa1c19f_100000.txt.part?? > RHI2_1000bit_faa1c19f_100000.txt
#
# Needs ~1.6 GB free. Safe to re-run: files already merged are skipped. The
# merged .txt are gitignored, so they will not show up as local changes.
#
# The pieces were produced with:
#   for f in $(find . -maxdepth 1 -name '*.txt' -size +100000000c -printf '%f\n'); do
#     split -C 90M -d -a 2 "$f" "$f.part"
#   done
#   sha256sum *.txt > SHA256SUMS
#
set -euo pipefail
cd "$(dirname "$0")"

FILTER="${1:-}"

# sha256sum on Linux, shasum on macOS; verification is skipped if neither exists.
if command -v sha256sum >/dev/null 2>&1; then
  SUM() { sha256sum "$@"; }
elif command -v shasum >/dev/null 2>&1; then
  SUM() { shasum -a 256 "$@"; }
else
  SUM() { return 1; }
fi

shopt -s nullglob
firsts=(*.part00)
if [ ${#firsts[@]} -eq 0 ]; then
  echo "error: no .part00 files in $(pwd)" >&2
  exit 1
fi

merged=0
skipped=0
for first in "${firsts[@]}"; do
  base="${first%.part00}"

  if [ -n "$FILTER" ]; then
    case "$base" in
      *"$FILTER"*) ;;
      *) continue ;;
    esac
  fi

  if [ -f "$base" ]; then
    echo "skip $base (already merged)"
    skipped=$((skipped + 1))
    continue
  fi

  pieces=("$base".part??)
  printf 'merging %s (%d pieces)... ' "$base" "${#pieces[@]}"

  # Write to a temp name first, so an interrupted run never leaves a truncated
  # file behind that the skip check above would later mistake for a complete one.
  cat "${pieces[@]}" > "$base.tmp$$"
  mv "$base.tmp$$" "$base"

  if [ -f SHA256SUMS ] && grep -q "  $base\$" SHA256SUMS; then
    if grep "  $base\$" SHA256SUMS | SUM -c - >/dev/null 2>&1; then
      echo "OK ($(du -h "$base" | cut -f1))"
    else
      echo "CHECKSUM MISMATCH" >&2
      rm -f "$base"
      echo "error: $base did not match SHA256SUMS and was removed." >&2
      echo "Its pieces are incomplete or corrupt -- re-fetch them and try again." >&2
      exit 1
    fi
  else
    echo "done ($(du -h "$base" | cut -f1), unverified)"
  fi

  merged=$((merged + 1))
done

echo
if [ "$merged" -eq 0 ] && [ "$skipped" -eq 0 ]; then
  echo "nothing matched filter '$FILTER'."
else
  echo "done: $merged merged, $skipped already present."
fi
