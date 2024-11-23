import sys
import os
from datetime import datetime
from Terminal_and_HTML_Code.myLogger import Logger
from Terminal_and_HTML_Code.html_merge_imagee_and_text import create_empty_html_file, append_text_to_html, append_figure_to_html, myPrint

def terminal_html(folder, formatted_datetime=None):

    # Format the date and time as a string
    formatted_datetime_was_none = False
    if formatted_datetime is None:
        formatted_datetime_was_none = True
        # Get the current date and time
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S_%f")

    formated_datetime_file = folder + "formated_datetime.txt"
    
    with open(formated_datetime_file, 'w') as file:
        file.writelines(formatted_datetime)

    my_terminal_filename = "terminal_" + formatted_datetime + ".txt"
    if formatted_datetime_was_none:
        sys.stdout = Logger(folder + my_terminal_filename, append=False)
    else:
        sys.stdout = Logger(folder + my_terminal_filename, append=True)

    OUTPUT_FILE = "output___" + formatted_datetime + ".html"
    if formatted_datetime_was_none:
        create_empty_html_file(folder + OUTPUT_FILE)

    my_p = myPrint(folder + OUTPUT_FILE)

    # Print the unique string
    my_p.print("")
    my_p.print("[Terminal_and_HTML.py] Terminal filename: \t{}" .format(folder + my_terminal_filename))
    my_p.print("[Terminal_and_HTML.py] HTML filename: \t\t{}" .format(folder + OUTPUT_FILE))
    my_p.print("")
    return formatted_datetime, my_p

# Example usage
#terminal_html(folder)  # Use default action
#terminal_html(folder, "custom_value")  # Assign custom value