import argparse
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient


class LoggingNotebookClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg['msg_type'] == 'stream':
            text = msg['content']['text']
            if not text.strip().endswith('%'):
                print(text, end='', flush=True)
        return super().process_message(msg, cell, cell_index)


def main():
    parser = argparse.ArgumentParser(description='Execute HW1 in a fresh Python kernel.')
    parser.add_argument('--fast', action='store_true', help='Use the debugging budget.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    notebook = nbformat.read(root / 'hw1.ipynb', as_version=4)
    output = root / ('hw1-fast.ipynb' if args.fast else 'hw1.ipynb')
    if args.fast:
        config = next(cell for cell in notebook.cells if cell.cell_type == 'code' and 'FAST_MODE =' in cell.source)
        config.source = config.source.replace('FAST_MODE = False', 'FAST_MODE = True')
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    started = time.monotonic()

    def save_progress(cell, cell_index, **kwargs):
        nbformat.write(notebook, output)
        print(f'Cell {cell_index}: {time.monotonic() - started:.1f} s', flush=True)

    client = LoggingNotebookClient(notebook, timeout=None, kernel_name='python3',
                                   resources={'metadata': {'path': str(root)}},
                                   on_cell_executed=save_progress)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
    print(f'Saved {output.name}; elapsed {(time.monotonic() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
