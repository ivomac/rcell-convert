"""Utility functions for the project."""

def singleton(cls):
    """Create a singleton instance of a class.

    Args:
        cls (type): Class to create a singleton instance of.

    Returns:
        function: Function to create a singleton instance of the class.

    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
