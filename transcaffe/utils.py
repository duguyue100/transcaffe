"""IO functions, checking, helpers, etc.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

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
