"""Test utils module.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from transcaffe import utils

from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises


def test_file_checker():
    """test file_checker function."""
    assert_raises(ValueError, utils.file_checker, "/a/b/c")


def test_dir_checker():
    """test directory checker function."""
    assert_raises(ValueError, utils.dir_checker, "/a/b/c", False)
