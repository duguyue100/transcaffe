"""IO functions, checking, helpers, etc.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os


def file_checker(filename):
    """Check if the file is existed.

    Parameters
    ----------
    filename : string
        an absolute path to a file.

    Returns
    -------
    abort if the file is not existed.
    """
    if not os.path.isfile(filename):
        raise ValueError("File %s is not existed!" % (filename))

    return True


def dir_checker(dir_name, make_dir=False):
    """Check if the directory is existed.

    Parameters
    ----------
    dir_name : string
        an absolute path to a directory
    make_dir : bool
        if create it when the directory is not existed.

    Returns
    -------
    abort if the directory is not existed
    """
    if not os.path.isdir(dir_name):
        if make_dir is True:
            # make sure you have right permession, otherwise will raise
            # OSError
            os.mkdirs(dir_name)
        else:
            raise ValueError("Directory %s is not existed!" % (dir_name))

    return True


def keras_model_writer(model, save_path, filename):
    """Save a valid keras model to given path.

    Parameters
    ----------
    model :
        A Keras model
    save_path : string
        A given path
    filename : string
        A given name of the model (without extension)
    """
    dir_checker(save_path, make_dir=True)

    print("[MESSAGE] The target Keras model is saving...")
    open(os.path.join(save_path, filename+".json"),
         mode="w").write(model.to_json())

    model.save_weights(os.path.join(save_path, filename+".h5"), overwrite=True)
    print("[MESSAGE] The model %s is saved at %s" % (filename, save_path))
