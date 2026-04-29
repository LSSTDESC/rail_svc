"""Database model for Estimates table"""

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
    from .dataset import Dataset
    from .estimator import Estimator

logger = structlog.get_logger(__name__)


class Estimates(Base):
    """Estimates model representing a collection of p(z) estimates

    An Estimates record is associated with a Dataset and an Estimator and
    references a file containing the actual probability distribution data.
    """

    __tablename__ = "estimates"

    #: primary key
    id: Mapped[int] = mapped_column(primary_key=True)

    # Unique name for this Estimates
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

    #: Number of objects in the estimates
    n_objects: Mapped[int] = mapped_column()

    #: Path to the relevant file
    qp_file_path: Mapped[str] = mapped_column(unique=True)

    #: foreign key into dataset table
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id", ondelete="CASCADE"),
        index=True,
    )

    #: foreign key into estimator table
    estimator_id: Mapped[int] = mapped_column(
        ForeignKey("estimator.id", ondelete="CASCADE"),
        index=True,
    )

    # Relationship - read-only access to associated dataset
    dataset: Mapped["Dataset"] = relationship(
        "Dataset",
        back_populates="estimates",
        viewonly=True,
    )

    # Relationship - read-only access to associated estimator
    estimator: Mapped["Estimator"] = relationship(
        "Estimator",
        back_populates="estimates",
        viewonly=True,
    )

    # Pydantic integration
    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class for serialization/validation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class for Estimates
        """
        return models.Estimates

    @classmethod
    def class_string(cls) -> str:
        """Return the class identifier string.

        Returns
        -------
        str
            The string 'estimates' for use in help functions and descriptions
        """
        return cls.__tablename__

    def __repr__(self) -> str:
        return (
            f"Estimates(name={self.name!r}, id={self.id}, "
            f"n_objects={self.n_objects}, dataset_id={self.dataset_id}, "
            f"estimator_id={self.estimator_id}, "
            f"path={self.qp_file_path!r})"
        )

    def __str__(self) -> str:
        """Return a simple string representation of the Estimates.

        Returns
        -------
        str
            Just the estimates name
        """
        return self.name

    @classmethod
    async def _resolve_foreign_key(
        cls,
        session: async_scoped_session,
        model_class: type,
        id_key: str,
        name_key: str,
        kwargs: dict[str, Any],
    ) -> int:
        """Helper to resolve foreign key from either ID or name.

        Parameters
        ----------
        session
            Database session
        model_class
            The model class to query (e.g., Estimator, Dataset)
        id_key
            Key name for the ID in kwargs (e.g., 'estimator_id')
        name_key
            Key name for the name in kwargs (e.g., 'estimator_name')
        kwargs
            Input kwargs dictionary

        Returns
        -------
        int
            The resolved foreign key ID

        Raises
        ------
        KeyError
            If neither ID nor name is provided
        """
        fk_id = kwargs.get(id_key, None)
        if fk_id is None:
            try:
                fk_name = kwargs[name_key]
            except KeyError as e:
                logger.warning(
                    "Missing input to create Estimates",
                    table=cls.__name__,
                    missing_key=str(e),
                )
                raise
            fk_obj = await model_class.get_row_by_name(session, fk_name)
            fk_id = fk_obj.id
        else:
            # Validate that the ID exists
            await model_class.get_row(session, fk_id)

        return fk_id

    @classmethod
    async def get_create_kwargs(
        cls,
        session: async_scoped_session,
        validate_file: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare kwargs for creating an Estimates instance.

        Parameters
        ----------
        session
            Database session
        validate_file
            Whether to validate the file exists and can be read (default: True)
        **kwargs
            Must include 'name' and 'qp_file_path'
            Should include 'estimator_name' or 'estimator_id'
            Should include 'dataset_name' or 'dataset_id'
            Optional: 'n_objects' (default: 0 if file not validated)

        Returns
        -------
        dict[str, Any]
            Validated kwargs ready for Estimates creation

        Raises
        ------
        KeyError
            If required parameters are missing
        FileNotFoundError
            If path is specified but file doesn't exist (when validate_file=True)
        ValueError
            If data validation fails or path traversal is detected
        """
        # Import here to avoid circular imports
        from .dataset import Dataset
        from .estimator import Estimator

        try:
            name = kwargs["name"]
            qp_file_path = kwargs["qp_file_path"]
        except KeyError as e:
            logger.warning(
                "Missing input to create Estimates",
                table=cls.__name__,
                missing_key=str(e),
            )
            raise

        # Resolve estimator_id
        estimator_id = await cls._resolve_foreign_key(
            session=session,
            model_class=Estimator,
            id_key="estimator_id",
            name_key="estimator_name",
            kwargs=kwargs,
        )

        # Resolve dataset_id
        dataset_id = await cls._resolve_foreign_key(
            session=session,
            model_class=Dataset,
            id_key="dataset_id",
            name_key="dataset_name",
            kwargs=kwargs,
        )

        # Validate data file
        if validate_file:
            # Prevent path traversal
            archive_path = Path(global_config.storage.archive).resolve()
            fullpath = (archive_path / qp_file_path).resolve()

            try:
                # Python 3.9+
                if not fullpath.is_relative_to(archive_path):
                    logger.error(
                        "Path traversal attempt detected",
                        table=cls.__name__,
                        attempted_path=str(qp_file_path),
                        archive_path=str(archive_path),
                    )
                    raise ValueError(
                        f"Path {qp_file_path} would escape archive directory"
                    )
            except AttributeError:
                # Fallback for Python < 3.9
                if not str(fullpath).startswith(str(archive_path)):
                    logger.error(
                        "Path traversal attempt detected",
                        table=cls.__name__,
                        attempted_path=str(qp_file_path),
                        archive_path=str(archive_path),
                    )
                    raise ValueError(
                        f"Path {qp_file_path} would escape archive directory"
                    )

            # Note: dataset could be passed to validate_data_for_path if needed
            # for schema validation in the future
            n_objects = cls.validate_data_for_path(fullpath)
        else:
            # Default to 0 to indicate unknown/unvalidated count
            n_objects = kwargs.get("n_objects", 0)

        return dict(
            name=name,
            qp_file_path=qp_file_path,
            n_objects=n_objects,
            dataset_id=dataset_id,
            estimator_id=estimator_id,
        )

    @classmethod
    def validate_data_for_path(
        cls,
        path: Path,
        dataset: "Dataset | None" = None,
    ) -> int:
        """Validate that data file exists and can be read.

        Parameters
        ----------
        path
            File with the data
        dataset
            Dataset in question (currently unused but reserved for future
            validation against dataset schema)

        Returns
        -------
        int
            Number of objects in the estimates

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the file cannot be read
        """
        # Future use: validate data matches dataset schema
        _ = dataset

        if not path.exists():
            logger.error(
                "Estimates file not found",
                table=cls.__name__,
                path=str(path),
            )
            raise FileNotFoundError(f"Input file {path} not found")

        try:
            n_objects = tables_io.io.getInputDataLength(str(path))
        except (OSError, ValueError) as exc:
            logger.error(
                "Failed to read estimates file",
                table=cls.__name__,
                path=str(path),
                error=str(exc),
            )
            raise ValueError(
                f"Could not read data from {path}: {exc}"
            ) from exc

        return n_objects
