import pandas as pd
import mpld3
from io import BytesIO
import base64
import matplotlib.pyplot as plt

def append_text_to_html(input_file, *args):
    # Read the existing content of the input file
    with open(input_file, 'r') as file:
        content = file.readlines()

    # Find the position to insert the text (before the closing </body> tag)
    insert_index = None
    for i in range(len(content) - 1, -1, -1):
        if '</body>' in content[i]:
            insert_index = i
            break

    if insert_index is not None:
        # Create the new line for the text
        new_line = '<br>\n'

        # Insert the new line and the text
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                html_table = arg.to_html(index=False)
                content.insert(insert_index, new_line)
                content.insert(insert_index + 1, html_table + '\n')
            else:
                content.insert(insert_index, new_line)
                content.insert(insert_index + 1, str(arg) + '\n')

        # Write the modified content back to the file
        with open(input_file, 'w') as file:
            file.writelines(content)
    else:
        print('Error: </body> tag not found in the input file.')

# def append_figure_to_html(input_file, figure):
#     # Convert the figure to HTML using mpld3
#     html = mpld3.fig_to_html(figure)

#     # Read the existing content of the input file
#     with open(input_file, 'r') as file:
#         content = file.readlines()

#     # Find the position to insert the figure (before the closing </body> tag)
#     insert_index = None
#     for i in range(len(content) - 1, -1, -1):
#         if '</body>' in content[i]:
#             insert_index = i
#             break

#     if insert_index is not None:
#         # Insert the figure HTML
#         content.insert(insert_index, html + '\n')

#         # Write the modified content back to the file
#         with open(input_file, 'w') as file:
#             file.writelines(content)
#     else:
#         print('Error: </body> tag not found in the input file.')

def append_figure_to_html(input_file, image_data):
    if image_data is None:
        print('Error: No image data found.')
        return

    image_html = f'<img src="data:image/png;base64,{image_data}" alt="Plot Image">\n'

    # Read the existing content of the input file
    with open(input_file, 'r') as file:
        content = file.readlines()

    # Find the position to insert the image (before the closing </body> tag)
    insert_index = None
    for i in range(len(content) - 1, -1, -1):
        if '</body>' in content[i]:
            insert_index = i
            break

    if insert_index is not None:
        # Insert the image HTML
        content.insert(insert_index, image_html + '\n')

        # Write the modified content back to the file
        with open(input_file, 'w') as file:
            file.writelines(content)
    else:
        print('Error: </body> tag not found in the input file.')

def create_empty_html_file(file_name):
    # Create the empty HTML file
    with open(file_name, 'w') as file:
        file.write('<html>\n')
        file.write('<body>\n')
        file.write('</body>\n')
        file.write('</html>\n')

class myPrint(object): 
    def __init__(self, filename):
        self.filename = filename
    
    def print(self, *args):
        append_text_to_html(self.filename, *args)
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                print(arg)
            else:
                print(arg)
        
    def save_plot_as_image(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)  # Close the figure after saving to release resources
        return image_data

    def show(self, fig):
        image_data = self.save_plot_as_image(fig)
        append_figure_to_html(self.filename, image_data)