from pathlib import Path


class ConsoleLogger(object):
    def __init__(self, output_path, stdout):
        self.output_path = Path(output_path)
        self.output_file = open(output_path / 'log.txt', 'a')
        self.terminal = stdout

    def write(self, message):
        self.terminal.write(message)
        self.output_file.write(message)
        self.terminal.flush()

    def flush(self):
        pass
