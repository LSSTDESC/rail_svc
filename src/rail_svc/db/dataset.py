"""Database model for Dataset table"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import tables_io
from pydantic import BaseModel
from sqlalchemy import String
from sqlalchemy.ext.asyncio import async_scoped_session
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .. import models
from ..config import config as global_config
from .base import Base

if TYPE_CHECKING:
    from .catalog_tag import CatalogTag

logger = structlog.get_logger(__name__)


class Dataset(Base):
    """Dataset model representing a collection of astronomical objects.

    A Dataset is associated with a CatalogTag and references a file
    containing the actual data.
    """

    __tablename__ = "dataset"

    #: primary key
    id: Mapped[int] = mapped_column(primary_key=True)

    # Unique name for this Dataset
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

    #: Number of objects in the dataset
    n_objects: Mapped[int] = mapped_column()

    #: Path to the relevant file
    path: Mapped[str] = mapped_column(unique=True)

    #: foreign key into catalog_tag table
    catalog_tag_id: Mapped[int] = mapped_column(
        ForeignKey("catalog_tag.id", ondelete="CASCADE"),
        index=True,
    )

    # Relationship - read-only access to associated catalog_tag
    catalog_tag: Mapped["CatalogTag"] = relationship(
        "CatalogTag",
        back_populates="datasets",
        viewonly=True,
    )

    # Pydantic integration
    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class for serialization/validation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class for Dataset
        """
        return models.Dataset

    @classmethod
    def class_string(cls) -> str:
        """Return the class identifier string.

        Returns
        -------
        str
            The string 'dataset' for use in help functions and descriptions
        """
        return cls.__tablename__

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, id={self.id}, "
            f"n_objects={self.n_objects}, catalog_tag_id={self.catalog_tag_id}, "
            f"path={self.path!r})"
        )

    def __str__(self) -> str:
        """Return a simple string representation of the Dataset.

        Returns
        -------
        str
            Just the dataset name
        """
        return self.name

    @classmethod
    async def get_create_kwargs(
        cls,
        session: async_scoped_session,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare kwargs for creating a Dataset instance.

        Parameters
        ----------
        session
            Database session
        **kwargs
            Must include 'name' and 'path'
            Should include 'catalog_tag_name' or 'catalog_tag_id'
            Optional: 'validate_file' (default: True), 'n_objects'

        Returns
        -------
        dict[str, Any]
            Validated kwargs ready for Dataset creation

        Raises
        ------
        KeyError
            If required parameters are missing
        FileNotFoundError
            If path is specified but file doesn't exist (when validate_file=True)
        ValueError
            If data validation fails or path traversal is detected
        """
        try:
            name = kwargs["name"]
            path = kwargs.get("path", None)
        except KeyError as e:
            logger.warning(
                "Missing input to create Dataset",
                table=cls.__name__,
                missing_key=str(e),
            )
            raise

        validate_file = kwargs.get("validate_file", True)

        # Get or validate catalog_tag_id
        catalog_tag_id = kwargs.get("catalog_tag_id", None)
        if catalog_tag_id is None:
            try:
                catalog_tag_name = kwargs["catalog_tag_name"]
            except KeyError as e:
                logger.warning(
                    "Missing input to create Dataset",
                    table=cls.__name__,
                    missing_key=str(e),
                )
                raise
            catalog_tag_ = await CatalogTag.get_row_by_name(session, catalog_tag_name)
            catalog_tag_id = catalog_tag_.id
        else:
            catalog_tag_ = await CatalogTag.get_row(session, catalog_tag_id)

        # Validate data file if path is provided
        if path is not None:
            if validate_file:
                # Prevent path traversal
                archive_path = Path(global_config.storage.archive).resolve()
                fullpath = (archive_path / path).resolve()

                if not str(fullpath).startswith(str(archive_path)):
                    logger.error(
                        "Path traversal attempt detected",
                        table=cls.__name__,
                        attempted_path=str(path),
                        archive_path=str(archive_path),
                    )
                    raise ValueError(
                        f"Path {path} would escape archive directory"
                    )

                n_objects = cls.validate_data_for_path(fullpath, catalog_tag_)
            else:
                n_objects = kwargs.get("n_objects", 1)
        else:
            # No path provided - use explicit n_objects
            n_objects = kwargs.get("n_objects", 1)

        return dict(
            name=name,
            path=path,
            n_objects=n_objects,
            catalog_tag_id=catalog_tag_id,
        )

    @classmethod
    def validate_data_for_path(
        cls,
        path: Path,
        catalog_tag: "CatalogTag | None" = None,
    ) -> int:
        """Validate that data file exists and can be read.

        Parameters
        ----------
        path
            File with the data
        catalog_tag
            CatalogTag in question (currently unused but reserved for future validation)

        Returns
        -------
        int
            Number of objects in the dataset

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the file cannot be read
        """
        # Future use: validate data matches catalog_tag schema
        _ = catalog_tag

        if not path.exists():
            logger.error(
                "Dataset file not found",
                table=cls.__name__,
                path=str(path),
            )
            raise FileNotFoundError(f"Input file {path} not found")

        try:
            n_objects = tables_io.io.getInputDataLength(str(path))
        except Exception as exc:
            logger.error(
                "Failed to read dataset file",
                table=cls.__name__,
                path=str(path),
                error=str(exc),
            )
            raise ValueError(
                f"Could not read data from {path}: {exc}"
            ) from exc

        return n_objects
