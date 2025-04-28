#!/usr/bin/env python
import csv
import io
import sys
from collections import defaultdict

# Set of non-permissive licenses that should cause failure
NON_PERMISSIVE_LICENSES = {
    "GPL", "GPLv2", "GPLv3", "GNU General Public License", "AGPL", "AGPLv3", "Affero GPL", "SSPL"
}

def license_matches_any(license_str: str, license_set: set):
    """Checks if any part of a semicolon-separated or 'or'-separated license string matches a set."""
    if not license_str:
        return False
    # Handle licenses separated by ';' or ' or '
    # Split by ' or ' first, then by ';' for each part
    parts = [l.strip() for part in license_str.split(' or ') for l in part.split(';')]
    # Filter out empty strings that might result from splitting
    parts = [p for p in parts if p]
    return any(part in license_set for part in parts)

def check_and_print_licenses_from_stdin():
    """Reads pip-licenses CSV from stdin, prints grouped licenses, and checks for non-permissive ones."""
    print("Reading and checking licenses from stdin...")

    # Read license data from standard input
    stdin_content = sys.stdin.read()

    if not stdin_content.strip():
        print("Warning: Received empty input from stdin. No license data provided?", file=sys.stderr)
        all_packages = []
    else:
        try:
            # Use io.StringIO to treat the stdin string as a file for csv.DictReader
            reader = csv.DictReader(io.StringIO(stdin_content))
            all_packages = list(reader)
        except Exception as e:
            print(f"Error parsing CSV data from stdin: {e}", file=sys.stderr)
            print(f"Input data was:\n{stdin_content}", file=sys.stderr)
            sys.exit(1)

    # --- Group packages by license ---
    licenses_grouped = defaultdict(list)
    for pkg in all_packages:
        license_name = pkg.get("License", "Unknown License")
        pkg_name = pkg.get("Name", "Unknown Package")
        pkg_version = pkg.get("Version", "Unknown Version")
        licenses_grouped[license_name].append(f"{pkg_name} ({pkg_version})")

    # --- Print grouped licenses ---
    print("\n--- Dependencies by License ---")
    if not licenses_grouped:
        print("No dependencies found.")
    else:
        # Sort licenses alphabetically for consistent output
        for license_name in sorted(licenses_grouped.keys()):
            print(f"\n[{license_name}]:")
            # Sort packages within each license group
            for pkg_info in sorted(licenses_grouped[license_name]):
                print(f" - {pkg_info}")
    print("-----------------------------\n")


    # --- Check for non-permissive licenses ---
    problematic = [
        pkg for pkg in all_packages
        if license_matches_any(pkg.get("License", ""), NON_PERMISSIVE_LICENSES)
    ]

    if problematic:
        print("🚫 Error: Found non-permissive licenses in dependencies:", file=sys.stderr)
        for pkg in problematic:
            print(f" - {pkg.get('Name', 'Unknown')} ({pkg.get('License', 'Unknown')})", file=sys.stderr)
        sys.exit(1) # Fail the process if non-permissive licenses are found
    else:
        print("✅ All licenses are permissive.")
        sys.exit(0) # Success

if __name__ == "__main__":
    check_and_print_licenses_from_stdin()
