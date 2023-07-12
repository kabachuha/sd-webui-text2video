import gradio as gr

class Text2VideoExtension(object):
    """
    A simple base class that sets a definitive way to process extensions
    """

    def __init__(self, extension_name: str = '', extension_title: str = ''):

        self.extension_name = extension_name
        self.extension_title = extension_title
        self.return_args_delimiter = f"extension_{extension_name}"

    def return_ui_inputs(self, return_args: list = [] ):
        """
        All extensions should use this method to return Gradio inputs.
        This allows for tracking the inputs using a delimiter.
        Arguments are automatically processed and returned.
        
        Output: <my_extension_name> + [arg1, arg2, arg3] + <my_extension_name>
        """
    
        delimiter = gr.State(self.return_args_delimiter)
        return [delimiter] + return_args + [delimiter]

    def process_extension_args(self, all_args: list = []):
        """
        Processes all extension arguments and appends them into a list.
        The filtered arguments are piped into the extension's process method.
        """

        can_append = False
        extension_args = []

        for value in all_args:
            if value == self.return_args_delimiter and not can_append:
                can_append = True
                continue

            if can_append:
                if value == self.return_args_delimiter:
                    break
                else:
                    extension_args.append(value)

        return extension_args

    def log(self, message: str = '', *args):
        """
        Choose to print a log specific to the extension.
        """
        OKGREEN = '\033[92m'
        ENDC = '\033[0m'

        title = self.extension_title
        message = f"Extension {title}: {message} " + ', '.join(args)
        print(OKGREEN + message + ENDC)
