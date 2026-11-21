import argparse
import os
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
    parser = argparse.ArgumentParser(description='Выполнение ноутбука в новом ядре')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--fast', action='store_true', help='Отладочный бюджет из задания')
    mode.add_argument('--cpu-smoke', action='store_true', help='Короткая проверка всего кода на процессоре')
    parser.add_argument('--without-bonuses', action='store_true', help='Пропустить дополнительные эксперименты')
    parser.add_argument('--prepare-only', action='store_true', help='Выполнить разделы до обучения')
    args = parser.parse_args()
    if args.cpu_smoke:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    root = Path(__file__).resolve().parent
    notebook = nbformat.read(root / 'hw2_nlp.ipynb', 4)
    config = notebook.cells[6]
    final = not args.fast and not args.cpu_smoke and not args.prepare_only
    config.source = config.source.replace('FAST_MODE = True', f'FAST_MODE = {not final}')
    config.source = config.source.replace('FAST_MODE = False', f'FAST_MODE = {not final}')
    if args.cpu_smoke:
        config.source += "\nSMOKE_MODE = True\nN_TRAIN = 64\nN_EVAL = 32\nEPOCHS = 1\nARTIFACTS = WORK_DIR / 'cache/cpu-smoke'\nARTIFACTS.mkdir(parents=True, exist_ok=True)"
        notebook.cells[40].source = notebook.cells[40].source.replace('BATCH_SIZE = 32', 'BATCH_SIZE = 8')
    if args.without_bonuses:
        config.source += '\nBONUS_TASKS = set()'
    if args.prepare_only:
        notebook.cells = notebook.cells[:39]
    suffix = '-cpu' if args.cpu_smoke or args.prepare_only else '-fast' if args.fast else ''
    output = root / f'hw2_nlp{suffix}.ipynb'
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    started = time.monotonic()

    def save():
        temporary = output.with_suffix('.tmp')
        nbformat.write(notebook, temporary)
        for attempt in range(5):
            try:
                temporary.replace(output)
                return
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.1 * (attempt + 1))
        nbformat.write(notebook, output)

    def completed(cell, cell_index, **kwargs):
        save()
        print(f'Ячейка {cell_index}: {time.monotonic() - started:.1f} с', flush=True)

    client = LoggingNotebookClient(notebook, timeout=None, kernel_name='python3',
                                   resources={'metadata': {'path': str(root)}},
                                   on_cell_executed=completed)
    try:
        client.execute()
    finally:
        save()
    print(f'Сохранён {output.name}; время {(time.monotonic() - started) / 60:.1f} мин', flush=True)


if __name__ == '__main__':
    main()
