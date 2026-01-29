#!/usr/bin/env python3
"""
Calimerge CLI - unified multi-camera motion capture.

Usage:
    calimerge           - Launch the unified GUI (default)
    calimerge gui       - Launch the unified GUI
    calimerge record    - Launch video recording GUI (legacy)
    calimerge clock     - Show sync verification clock
    calimerge --help    - Show this help
"""

import sys


def main():
    if len(sys.argv) < 2:
        # Default to unified GUI
        from calimerge.gui.main import main as gui_main
        return gui_main()

    if sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("Commands:")
        print("  gui       Launch the unified GUI (cameras, record, calibrate, process)")
        print("  record    Launch the video recording GUI (legacy)")
        print("  clock     Display sync verification clock widget")
        print()
        return 0

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove subcommand from argv

    if command == "gui":
        from calimerge.gui.main import main as gui_main
        return gui_main()

    elif command == "record":
        from calimerge.recorder_gui import main as record_main
        return record_main()

    elif command == "clock":
        from calimerge.clock_widget import main as clock_main
        return clock_main()

    else:
        print(f"Unknown command: {command}")
        print("Run 'calimerge --help' for usage")
        return 1


if __name__ == "__main__":
    sys.exit(main())
