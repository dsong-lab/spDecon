"""Run spot-level spDecon from the repository root."""
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ('reference', 'histology', 'multisample', 'multiomics'):
        from spDecon.extensions_cli import main
    else:
        if len(sys.argv) > 1 and sys.argv[1] == 'core':
            del sys.argv[1]
        from spDecon.cli import main
    main()
