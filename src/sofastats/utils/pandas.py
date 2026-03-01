"""Utilties for pandas, mostly for managing support for both pandas 2 and 3 simultaneously."""

from contextlib import contextmanager
from typing import TypeVar, cast

import pandas as pd
from packaging.version import Version

IS_PANDAS_3PLUS = Version(pd.__version__) >= Version("3.0.0")

T = TypeVar("T", bound=pd.DataFrame | pd.Series)


@contextmanager
def no_silent_downcasting():
    ## https://stackoverflow.com/questions/77900971/pandas-futurewarning-downcasting-object-dtype-arrays-on-fillna-ffill-bfill
    ## https://medium.com/@felipecaballero/deciphering-the-cryptic-futurewarning-for-fillna-in-pandas-2-01deb4e411a1
    if not IS_PANDAS_3PLUS:
        with pd.option_context("future.no_silent_downcasting", True):
            yield
    else:
        yield


def infer_objects_no_copy(df: T) -> T:
    if not IS_PANDAS_3PLUS:
        return cast(T, df.infer_objects(copy=False))
    else:
        # In pandas 3+, the copy= argument to .infer_objects() is deprecated.
        # If we want copy=False, we can just use the default behaviour of .infer_objects() now.
        return cast(T, df.infer_objects())
