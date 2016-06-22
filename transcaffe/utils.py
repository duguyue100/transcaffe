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
