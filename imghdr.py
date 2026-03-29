"""
A local shim for the removed 'imghdr' library in Python 3.13.
This satisfies legacy imports in 3rd-party libraries.
"""

def what(file, h=None):
    """
    Mock implementation of imghdr.what
    """
    return None
