from Terminal_and_HTML_Code.Terminal_and_HTML import terminal_html
from settings import app_settings
import matplotlib.pyplot as plt

class ST_TerminalLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ST_TerminalLogger, cls).__new__(cls)
            cls._instance.initialize_logger()
        return cls._instance

    def initialize_logger(self):
        formatted_datetime, p = terminal_html(app_settings.output_folder)
        self._logger_instance = p
        p.print(f"[main.py] terminal_html folder = {app_settings.output_folder}")

    def get_logger(self):
        return self._logger_instance


def log_message(message):
    logger = ST_TerminalLogger().get_logger()
    logger.print(message)

def log_image(fig):
    logger = ST_TerminalLogger().get_logger()
    logger.show(fig)

def visualize_matdata(data, title):

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(data, cmap='jet')
    axs[0].set_title(title)
    axs[0].axis('off')
    axs[1].axis('off')
    log_image(fig)
