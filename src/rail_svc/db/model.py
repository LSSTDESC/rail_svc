"""Database model for Model table"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel
from rail.core import Model as RailModel
from sqlalchemy import String
from sqlalchemy.ext.asyncio import async_scoped_session
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .. import models, db_funcs
from ..config import config as global_config
from .algorithm import Algorithm
from .base import Base
from .catalog_tag import CatalogTag

if TYPE_CHECKING:
    from .estimator import Estimator

logger = structlog.get_logger(__name__)


class Model(Base):
    """Model representing a trained machine learning model.

    A Model is associated with an Algorithm and CatalogTag, and references
    a file containing the serialized model data.
    """

    __tablename__ = "model"

    #: primary key
    id: Mapped[int] = mapped_column(primary_key=True)

    #: Name for this Model, unique
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

    #: Path to the relevant file
    path: Mapped[str] = mapped_column()

    #: foreign key into `Algorithm` table
    algo_id: Mapped[int] = mapped_column(
        ForeignKey("algorithm.id", ondelete="CASCADE"),
        index=True,
    )

    #: foreign key into catalog_tag table
    catalog_tag_id: Mapped[int] = mapped_column(
        ForeignKey("catalog_tag.id", ondelete="CASCADE"),
        index=True,
    )

    # Relationship - read-only access to associated algorithm
    algo: Mapped[Algorithm] = relationship(
        "Algorithm",
        back_populates="models",
        viewonly=True,
    )

    # Relationship - read-only access to associated catalog_tag
    catalog_tag: Mapped[CatalogTag] = relationship(
        "CatalogTag",
        back_populates="models",
        viewonly=True,
    )

    # Relationship - read-only access to associated Estimators
    estimators: Mapped[list[Estimator]] = relationship(
        "Estimator",
        back_populates="model",
        viewonly=True,
    )

    # Pydantic integration
    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class for serialization/validation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class for Model
        """
        return models.Model

    @classmethod
    def class_string(cls) -> str:
        """Return the class identifier string.

        Returns
        -------
        str
            The string 'model' for use in help functions and descriptions
        """
        return cls.__tablename__

    def __repr__(self) -> str:
        return (
            f"Model(name={self.name!r}, id={self.id}, "
            f"algo_id={self.algo_id}, catalog_tag_id={self.catalog_tag_id}, "
            f"path={self.path!r})"
        )

    def __str__(self) -> str:
        """Return a simple string representation of the Model.

        Returns
        -------
        str
            Just the model name
        """
        return self.name

    @classmethod
    async def get_create_kwargs(
        cls,
        session: async_scoped_session,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare kwargs for creating a Model instance.

        Parameters
        ----------
        session
            Database session
        **kwargs
            Must include 'name' and 'path'
            Should include 'algo_name' or 'algo_id'
            Should include 'catalog_tag_name' or 'catalog_tag_id'
            Optional: 'validate_file' (default: True)

        Returns
        -------
        dict[str, Any]
            Validated kwargs ready for Model creation

        Raises
        ------
        KeyError
            If required parameters are missing
        FileNotFoundError
            If path is specified but file doesn't exist (when validate_file=True)
        ValueError
            If model validation fails or path traversal is detected
        """
        try:
            name = kwargs["name"]
            path = kwargs["path"]
        except KeyError as e:
            logger.warning(
                "Missing input to create Model",
                table=cls.__name__,
                missing_key=str(e),
            )
            raise

        validate_file = kwargs.get("validate_file", True)

        # Get or validate algo_id
        algo_id = kwargs.get("algo_id", None)
        if algo_id is None:
            try:
                algo_name = kwargs["algo_name"]
            except KeyError as e:
                logger.warning(
                    "Missing input to create Model",
                    table=cls.__name__,
                    missing_key=str(e),
                )
                raise
            algo_ = await db_funcs.get_row_by_name(Algorithm, session, algo_name)
            algo_id = algo_.id
        else:
            algo_ = await db_funcs.get_row(Algorithm, session, algo_id)

        # Get or validate catalog_tag_id
        catalog_tag_id = kwargs.get("catalog_tag_id", None)
        if catalog_tag_id is None:
            try:
                catalog_tag_name = kwargs["catalog_tag_name"]
            except KeyError as e:
                logger.warning(
                    "Missing input to create Model",
                    table=cls.__name__,
                    missing_key=str(e),
                )
                raise
            catalog_tag_ = await db_funcs.get_row_by_name(CatalogTag, session, catalog_tag_name)
            catalog_tag_id = catalog_tag_.id
        else:
            catalog_tag_ = await db_funcs.get_row(CatalogTag, session, catalog_tag_id)

        # Validate model file if requested
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
                raise ValueError(f"Path {path} would escape archive directory")

            cls.validate_model(fullpath, algo_, catalog_tag_)

        return dict(
            name=name,
            path=path,
            algo_id=algo_id,
            catalog_tag_id=catalog_tag_id,
        )

    @classmethod
    def validate_model(
        cls,
        path: Path,
        algo: Algorithm,
        catalog_tag: CatalogTag,
    ) -> None:
        """Validate that the model is appropriate for the Algorithm and CatalogTag.

        Parameters
        ----------
        path
            Path to the model file
        algo
            Algorithm in question
        catalog_tag
            CatalogTag in question

        Raises
        ------
        FileNotFoundError
            If the model file doesn't exist
        ValueError
            If the model doesn't match the algorithm or catalog tag
        """
        if not path.exists():
            logger.error(
                "Model file not found",
                table=cls.__name__,
                path=str(path),
            )
            raise FileNotFoundError(f"Input file {path} not found")

        try:
            the_model = RailModel.read(str(path))
        except Exception as exc:
            logger.error(
                "Failed to read model file",
                table=cls.__name__,
                path=str(path),
                error=str(exc),
            )
            raise ValueError(f"Could not read model from {path}: {exc}") from exc

        # Validate catalog tag
        if the_model.catalog_tag:
            if the_model.catalog_tag != catalog_tag.name:
                logger.error(
                    "CatalogTag mismatch",
                    table=cls.__name__,
                    path=str(path),
                    model_catalog_tag=the_model.catalog_tag,
                    expected_catalog_tag=catalog_tag.name,
                )
                raise ValueError(
                    f"CatalogTag does not match: {the_model.catalog_tag} != {catalog_tag.name}"
                )

        # Validate algorithm
        if the_model.creation_class_name:
            expected_estimator_class = the_model.creation_class_name.replace(
                "Informer", "Estimator"
            )
            if algo.class_name != expected_estimator_class:
                logger.error(
                    "Algorithm class mismatch",
                    table=cls.__name__,
                    path=str(path),
                    expected_class=expected_estimator_class,
                    actual_class=algo.class_name,
                )
                raise ValueError(
                    f"Algorithm does not match: {expected_estimator_class} != {algo.class_name}"
                )

        logger.info(
            "Model validation successful",
            table=cls.__name__,
            path=str(path),
            algo=algo.name,
            catalog_tag=catalog_tag.name,
        )
