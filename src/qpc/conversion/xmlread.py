"""A module to read the XML metadata files and extract data from them."""

from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


class XmlReader:
    """A class to read XML files and extract data from them.

    Attributes:
        file (Path): The path to the XML file.
        tree (ElementTree): The XML tree.
        root (Element): The root element of the XML tree.

    """

    def __init__(self, file: Path):
        """Initialize the class with the XML file.

        Args:
            file (Path): The path to the XML file.

        """
        self.file = file
        if not self.file.is_file():
            raise FileNotFoundError(f"XML file not found: {self.file}")
        self.tree = ET.parse(self.file)
        self.root = self.tree.getroot()

    def tag(
        self,
        *tags: tuple[str, int] | str,
        parser: Callable | None = None,
    ) -> str:
        """Get the text contents of a tag in a nested structure.

        Args:
            *tags (tuple[str, int] or str):
                The tag name for each level of the nested structure. If a
                tuple is provided, the second element is the index of the tag.
                Otherwise, the index is assumed to be 0.
            parser (Callable, optional):
                A function to parse the text content.

        Returns:
            str: The text content of the tag.

        """
        target_depth = len(tags)

        def nav(el, depth):
            if depth == target_depth:
                return el.text
            if isinstance(tags[depth], str):
                tag = tags[depth]
                index = 0
            else:
                tag, index = tags[depth]
            child = el.findall(tag)[index]
            return nav(child, depth + 1)

        text = nav(self.root, 0)
        if text is None:
            raise ValueError(f"Tag not found: {tags}")

        if parser is None:
            return text
        return parser(text)

    def tags(
        self,
        *tags: str,
        parser: Callable | None = None,
    ) -> list:
        """Get the text contents of all tags in a nested structure.

        Args:
            tags (tuple[str]): The tag name for each level of the nested structure.
            parser (Callable, optional): A function to parse the text content.

        Returns:
            list: The contents of the tags.

        """
        text = []
        target_depth = len(tags)

        def nav(el, depth):
            if depth == target_depth:
                string = str(el.text)
                if parser is not None:
                    string = parser(string)
                text.append(string)
            else:
                children = el.findall(tags[depth])
                for child in children:
                    nav(child, depth + 1)

        nav(self.root, 0)

        return text


class XmlFolder:
    """A class to extract data from the XML metadata files.

    The zip file is extracted if the folder is not found.

    Attributes:
        folder (Path): The folder containing the XML metadata files.

    """

    def __init__(self, folder: Path):
        """Initialize the class with the folder containing the XML metadata files.

        Args:
            folder (Path): The folder containing the XML metadata files.

        """
        self.folder = folder

        if not self.folder.is_dir():
            extract_zip(self.folder.with_suffix(".zip"))
        return

    def is_file(self, file: str) -> bool:
        """Check if a file exists in the folder.

        Args:
            file (str): The name of the file.

        Returns:
            bool: True if the file exists, False otherwise.

        """
        return (self.folder / file).is_file()

    def read(self, file: str) -> XmlReader:
        """Open an XML metadata file.

        Args:
            file (str): The name of the XML metadata file.

        Returns:
            XmlReader: The XML reader object.

        """
        if not file.endswith(".xml"):
            file += ".xml"

        return XmlReader(self.folder / file)


def extract_zip(zip_file: Path) -> Path:
    """Extract the contents of a zip file.

    Args:
        zip_file (Path): The path to the zip file.

    Returns:
        Path: The path to the extracted folder.

    """
    if not zip_file.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_file}")

    folder = zip_file.with_suffix("")
    with ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(folder)

    return folder
