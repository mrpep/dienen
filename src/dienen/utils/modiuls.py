import importlib
import sys
from pathlib import Path
import inspect

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def module_from_folder(module_path):
    """
    Arguments:
        module_path: path to a .py file
    Returns:
        the module in module_path
    """

    module_path = Path(module_path)
    if module_path.exists():
        sys.path.append(str((module_path.absolute()).parent))
        module = importlib.import_module(module_path.stem)
    else:
        raise Exception('Folder doesn\'t exist')

    return module

def get_modules(module_paths):
    """
    Get all the modules where task definitions are found
    """
    
    task_modules = []
    for module_path in module_paths:
        try:
            module = module_from_file(Path(module_path).stem,module_path)
        except:
            try:
                module = module_from_folder(module_path)    
            except:
                try:
                    module = importlib.import_module(module_path)
                except:
                    raise Exception('Could not import Module {}'.format(module_path))

        task_modules.append(module)

    return task_modules

def get_members_from_module(module,filters=[inspect.isclass]):
    clsmembers = []
    for filt in filters:
        clsmembers.extend(inspect.getmembers(module, filt))

    clsmembers_dict = {cls[0]:cls[1] for cls in clsmembers}

    return clsmembers_dict

def get_classes_in_module(module):
    """
    Returns a dictionary containing all the available classes in a module.
    Arguments:
        module: a module from which to extract available classes
    Outputs:
        returns a dictionary containing class names as keys and class objects as values. 
    """

    clsmembers_dict = get_members_from_module(module,filters=[inspect.isclass])
    return clsmembers_dict

