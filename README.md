# ryp: R inside Python

ryp is a minimalist, powerful Python library for:
- running R code inside Python
- quickly transferring huge datasets between Python (NumPy/pandas/polars) and R
  without writing to disk
- interactively working in both languages at the same time

ryp is an alternative to the widely used [rpy2](https://github.com/rpy2/rpy2) 
library. Compared to rpy2, ryp provides:
- increased stability
- a much simpler API, with less of a learning curve
- interactive printouts of R variables that match what you'd see in R
- a full-featured R terminal inside Python for interactive work
- inline plotting in Jupyter notebooks (requires the `svglite` R package)
- much faster data conversion with [Arrow](https://arrow.apache.org) (also
  provided by [rpy2-arrow](https://github.com/rpy2/rpy2-arrow))
- support for *every* NumPy, pandas and polars data type representable in base
  R, no matter how obscure
- support for sparse arrays/matrices
- recursive conversion of containers like R lists, Python tuples/lists/dicts, 
  and S3/S4/R6 objects
- full Windows support

ryp does the opposite of the 
[reticulate](https://rstudio.github.io/reticulate) R library, which runs Python
inside R.

## Installation

Install ryp via pip:

```bash
pip install ryp
```

conda:

```bash
conda install ryp
```

or mamba:

```bash
mamba install ryp
```

Or, install the development version via pip:

```bash
pip install git+https://github.com/Wainberg/ryp
```

ryp's only mandatory dependencies are:
- Python 3.7+
- R
- the [cffi](https://cffi.readthedocs.io/en/stable) Python package
- the [pyarrow](https://arrow.apache.org/docs/python) Python package, which 
  includes [NumPy](https://numpy.org) as a dependency
- the [arrow](https://arrow.apache.org/docs/r) R library

R and the arrow R library are automatically installed when installing ryp via 
conda or mamba, but not via pip. ryp uses the R installation pointed to by the
environment variable `R_HOME`, or if `R_HOME` is not defined or not a 
directory, by running `R RHOME` through `subprocess.run()`.

ryp also has several optional dependencies, which are not installed 
automatically with pip, conda or mamba. These are:
- [pandas](https://pandas.pydata.org), for `format='pandas'`
- [polars](https://pola.rs), for `format='polars'`
- [SciPy](https://scipy.org) and the
  [Matrix](https://cran.r-project.org/web/packages/Matrix) R library, for sparse
  matrices
- the [svglite](https://cran.r-project.org/web/packages/svglite) R library, for
  inline plotting in Jupyter notebooks

## Functionality

ryp consists of just three core functions:

1. `r(R_code)` runs a string of R code. `r()` with no arguments opens up an R 
   terminal inside your Python terminal for interactive work.
2. `to_r(python_object, R_variable_name)` converts a Python object into an R 
   object named `R_variable_name`. 
3. `to_py(R_statement)` converts the R object produced by evaluating 
   `R_statement` to Python. `R_statement` may be a single variable name, or a 
   more complex code snippet that evaluates to the R object you'd like to 
   convert.

and two more functions, `get_config()` and `set_config()`, for getting and 
setting ryp's global configuration options.

### `r()`

```python
r(R_code: str = ...) -> None
```

`r(R_code)` runs a string of R code inside ryp's R interpreter, which is 
embedded inside Python. It can contain multiple statements separated by
semicolons or newlines (e.g. within a triple-quoted Python string). It returns
`None`; use `to_py()` instead if you would like to convert the result back to 
Python.

`r()` with no arguments opens up an R terminal inside your Python terminal 
for interactive debugging. Press `Ctrl + D` to exit back to the Python 
terminal. R variables defined from Python will be available in the R terminal,
and variables defined in the R terminal will be available from Python once you
exit:

```python
>> > from ryp.ryp.ryp import r
>> > r('a = 1')
>> > r()
> a
[1]
1
> b < - 2
>
>> > r('b')
[1]
2
```

Note that the default value for `R_code` is the special sentinel value `...` 
(`Ellipsis`) rather than `None`. This stops users from inadvertently opening 
the terminal when passing a variable that is supposed to be a string but is 
unexpectedly `None`.

### `to_r()`

```python
to_r(python_object: object, R_variable_name: str, *, 
     format: Literal['keep', 'matrix', 'data.frame'] | None = None,
     rownames: object = None, colnames: object = None) -> None
```

`to_r(python_object, R_variable_name)` converts `python_object` to R, adding it
to R's global namespace (`globalenv`) as a variable named `R_variable_name`. 

If `python_object` is a container (`list`, `tuple`, or `dict`), `to_r()`
recursively converts each element and returns an R named list (if
`python_object` is a `dict`) or unnamed list (if `python_object` is a `list` or
`tuple`).

#### The `format` argument

By default (`format='keep'`), ryp converts polars and pandas DataFrames (and 
pandas MultiIndexes) into R data frames, and 2D NumPy arrays into R matrices. 
Specify `format='matrix'` to convert everything (even DataFrames) to R matrices
(in which case all DataFrame columns must have the same type), and 
`format='data.frame'` to convert everything (even 2D NumPy arrays) to R 
data frames.

`format` must be `None` unless `python_object` is a DataFrame, MultiIndex or 2D
NumPy array – or unless `python_object` is a `list`, `tuple`, or `dict`, in 
which case the `format` will apply recursively to any DataFrames, MultiIndexes
or 2D NumPy arrays it contains.

#### The `rownames` and `colnames` arguments

Since NumPy arrays, polars DataFrames and Series, and scipy sparse arrays and 
matrices lack row and column names, you can specify these separately via the 
`rownames` and/or `colnames` arguments, and they will be added to the converted
R object. `rownames` and `colnames` can be lists, tuples, string Series, or 
categorical Series with string categories, and will be automatically converted
to R character vectors. 

`rownames` and `colnames` must match the length or `shape[1]`, respectively, of
the object being converted. The one exception is that rownames of any length
may be added to a 0 &times; 0 polars DataFrame, since polars does not have the 
concept of an `N` &times; 0 DataFrame for nonzero `N`. (Dropping all the 
columns of a polars DataFrame always results in a 0 &times; 0 DataFrame, even 
if the original DataFrame had more than 0 rows.)

Because Python `bool`, `int`, `float`, and `str` convert to length-1 R vectors
that support names, you can pass length-1 `rownames` when converting objects of
these types. You can also pass `rownames` and/or `colnames` when 
`python_object` is a `list`, `tuple`, or `dict`, in which case row and column 
names will only be added to elements that support them. All elements that 
support `rownames` must have the same length as the `rownames`, and similarly 
for the `colnames`. 

`rownames` cannot be specified if `python_object` is a pandas Series or 
DataFrame (since they already have rownames, i.e. an index), or 
`bytes`/`bytearray` (since these convert to `raw` vectors, which lack 
rownames). `colnames` cannot be specified unless `python_object` is a 
multidimensional NumPy array or scipy sparse array or matrix, or something that
might contain one (`list`, `tuple`, or `dict`).

### `to_py()`

```python
to_py(R_statement: str, *,
      format: Literal['polars', 'pandas', 'pandas-pyarrow', 'numpy'] |
              dict[Literal['vector', 'matrix', 'data.frame'],
                   Literal['polars', 'pandas', 'pandas-pyarrow',
                           'numpy']] | None = None,
      index: str | Literal[False] | None = None,
      squeeze: bool | None = None) -> Any
```

`to_py(R_statement)` runs a single statement of R code (which can be as simple 
as a single variable name) and converts the resulting R object to Python. 

If the object is a list/S3 object, S4 object, or environment/R6 object, it
recursively converts each attribute/slot/field and returns a Python `dict` (or 
`list`, if the object is an unnamed list). For R6 objects, only public fields
will be converted.

#### The `format` argument

By default, or when `format='polars'`, R vectors will be converted to polars 
Series, and R data frames and matrices will be converted to polars DataFrames. 
You can change this by setting the `format` argument to `'pandas'`, 
`'pandas-pyarrow'` (like `'pandas'`, but converting to pyarrow dtypes wherever 
possible) or `'numpy'`. (You can also change the default format, e.g. with 
`set_config(to_py_format='pandas')`.)

For finer-grained control, you can set `format` for only certain R variable 
types by specifying a dictionary with `'vector'`, `'matrix'`, and/or
`'data.frame'` as keys and `'polars'`, `'pandas'`, `'pandas-pyarrow'` and/or 
`'numpy'` as values. 

`format` must be `None` when `R_statement` evaluates to `NULL`, when it 
evaluates to an array of 3 or more dimensions (these are always converted to 
NumPy arrays), or when the final result would be a Python scalar (see `squeeze`
below).

#### The `index` argument

By default, the R object's `names` or `rownames` will become the index (for 
pandas) or the first column (for polars) of the output Python object, named 
`'index'`. Set the `index` argument to a different string to change this name, 
or set `index=False` to not convert the `names`/`rownames`. 

Note that for polars, the output will be a two-column DataFrame (not a Series!)
when the input is an R vector, unless `index=False`. 

When the output is a NumPy array, `names` and `rownames` will always be 
discarded, since numeric NumPy arrays cannot store string indexes except with 
the inefficient `dtype=object`. 

`index` must be `None` when `format='numpy'`, or when the final result would be
a Python scalar (see `squeeze` below).

#### The `squeeze` argument

By default, length-1 R vectors, matrices and arrays will be converted to Python
scalars instead of Python arrays, Series or DataFrames. Set `squeeze=False` to
disable this special case. (R data frames are never converted to Python scalars
even if `squeeze=True`.) 

`squeeze` must be `None` unless the R object is a vector, matrix or array
(`raw` vectors don't count, because they always convert to Python scalars).

### `set_config()` and `get_config()`

```python
set_config(*, to_r_format=None, to_py_format=None, index=None, squeeze=None, 
           plot_width: int | float | None = None, 
           plot_height: int | float | None = None) -> None
```

`set_config` sets ryp's configuration settings. Arguments that are `None` are 
left unchanged.
    
For instance, to set pandas as the default format in `to_py()`, run 
`set_config(to_py_format='pandas')`.

- `to_r_format`: the default value for the `format` parameter in `to_r()`; 
  must be `'keep'` (the default), `'matrix'`, or `'data.frame'`.
- `to_py_format`: the default value for the `format` parameter in `to_py()`; 
  must be `'polars'` (the default), `'pandas'`, `'pandas-pyarrow'`, `'numpy'`,
  or a dictionary with one of those four Python formats and/or `None` as values
  and `'vector'`, `'matrix'` and/or `'data.frame'` as keys. If certain keys are 
  missing or have `None` as the format, leave their format unchanged.
- `index`: the default value for the `index` parameter in to_py(); must be a 
  string (default: `'index'`) or `False`. 
- `squeeze`: the default value for the `squeeze` parameter in `to_py()`; must  
  be `True` (the default) or `False`.
- `plot_width`: the width, in inches, of inline plots in Jupyter notebooks;
  must be a positive number. Defaults to 6.4 inches, to match Matplotlib's 
  default.
- `plot_height`: the height, in inches, of inline plots in Jupyter notebooks;
  must be a positive number. Defaults to 4.8 inches, to match Matplotlib's 
  default.

```python
get_config() -> dict[str, dict[str, str] | str | bool | int]
```

`get_config` returns the current configuration options as a dictionary, with 
keys `to_r_format`, `to_py_format`, `index`, `squeeze`, `plot_width`, and 
`plot_height`.

## Conversion rules

### Python to R (`to_r()`)

| Python                                                                  | R                                                                                                         |
|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `None`                                                                  | `NULL` (if scalar) or `NA` (if inside NumPy, pandas or polars)                                            |
| `nan`                                                                   | `NaN` (if scalar or inside polars) or `NA` (if inside NumPy or pandas)                                    |
| `pd.NA`                                                                 | `NA`                                                                                                      |
| `pd.NaT`, `np.datetime64('NaT')`, `np.timedelta64('NaT')`               | `NA`                                                                                                      |   
| `bool`                                                                  | length-1 `logical` vector                                                                                 |
| `int`                                                                   | length-1 `integer` (if `abs(x) <= 2_147_483_647`) or `bit64::integer64` vector                            |
| `float`                                                                 | length-1 `numeric` vector                                                                                 |
| `str`                                                                   | length-1 `character` vector                                                                               |
| `complex`                                                               | length-1 `complex` vector                                                                                 |
| `datetime.date`                                                         | length-1 `Date` vector                                                                                    |
| `datetime.datetime`                                                     | length-1 `POSIXct` vector                                                                                 |
| `datetime.timedelta`                                                    | length-1 `difftime(units='secs')` vector                                                                  |
| `datetime.time` (`tzinfo` must be `None`)                               | length-1 `hms::hms` vector                                                                                |
| `bytes`, `bytearray`                                                    | `raw` vector                                                                                              |
| `list`, `tuple`                                                         | unnamed list                                                                                              |
| `dict` (all keys must be strings)                                       | named list                                                                                                |
| polars Series, pandas Series<sup>&ast;</sup>, pandas `Index`            | vector                                                                                                    |
| polars DataFrame, pandas DataFrame<sup>&ast;</sup>, pandas `MultiIndex` | matrix<sup>&dagger;</sup> (if `format == 'matrix'`; all columns must have same data type) or `data.frame` |
| 1D NumPy array                                                          | vector                                                                                                    |
| 2D NumPy array                                                          | `data.frame` (if `format == 'data.frame'`) or matrix<sup>&dagger;</sup>                                   |
| &ge; 3D NumPy array                                                     | array<sup>&dagger;</sup>                                                                                  |
| 0D NumPy array (e.g. `np.array(1)`), NumPy generic (e.g. `np.int32(1)`) | length-1 vector                                                                                           |
| `csr_array`, `csr_matrix`                                               | `dgRMatrix` (if int or float), `lgRMatrix` (if boolean), -- (if complex)                                  | 
| `csc_array`, `csc_matrix`                                               | `dgCMatrix` (if int or float), `lgCMatrix` (if boolean), -- (if complex)                                  |
| `coo_array`, `coo_matrix`                                               | `dgTMatrix` (if int or float), `lgTMatrix` (if boolean), -- (if complex)                                  |

#### NumPy data types

| Python                                      | R                                                              |
|---------------------------------------------|----------------------------------------------------------------|
| `bool`                                      | `logical`                                                      |
| `int8`, `uint8`, `int16`, `uint16`, `int32` | `integer`                                                      |
| `uint32`, `uint64`                          | `integer` (if `x <= 2_147_483_647`) or `numeric`               |
| `int64`                                     | `integer` (if `abs(x) <= 2_147_483_647`) or `bit64::integer64` |
| `float16`, `float32`, `float64`, `float128` | `numeric` (note: `float128` loses precision)                   |
| `complex64`, `complex128`                   | `complex`                                                      |
| `bytes` (e.g. `'S1'`)                       | --                                                             |
| `str`/`unicode` (e.g. `'U1'`)               | `character`                                                    |
| `datetime64`                                | `POSIXct`                                                      | 
| `timedelta64`                               | `difftime(units='secs')`                                       |
| `void` (unstructured)                       | `raw`                                                          |
| `void` (structured)                         | --                                                             |
| `object`                                    | depends on the contents<sup>&Dagger;</sup>                     |

#### pandas-specific data types

| Python                                                               | R                                                              |
|----------------------------------------------------------------------|----------------------------------------------------------------|
| `BooleanDtype`                                                       | `logical`                                                      |
| `Int8Dtype`, `UInt8Dtype`, `Int16Dtype`, `UInt16Dtype`, `Int32Dtype` | `integer`                                                      |
| `UInt32Dtype`, `UInt64Dtype`                                         | `integer` (if `x <= 2_147_483_647`) or `numeric`               |
| `Int64Dtype`                                                         | `integer` (if `abs(x) <= 2_147_483_647`) or `bit64::integer64` |  
| `Float32Dtype`, `Float64Dtype`                                       | `numeric`                                                      |
| `StringDtype`                                                        | `character`                                                    |
| `CategoricalDtype(ordered=False)`                                    | unordered `factor`                                             |
| `CategoricalDtype(ordered=True)`                                     | ordered `factor`                                               |
| `DatetimeTZDtype`, `PeriodDtype`                                     | `POSIXct`                                                      |
| `IntervalDtype`, `SparseDtype`                                       | --                                                             |

#### pandas Arrow data types (`pd.ArrowDtype`)

| Python                                                     | R                                                              |
|------------------------------------------------------------|----------------------------------------------------------------|
| `pa.bool_`                                                 | `logical`                                                      |
| `pa.int8`, `pa.uint8`, `pa.int16`, `pa.uint16`, `pa.int32` | `integer`                                                      |
| `pa.uint32`, `pa.uint64`                                   | `integer` (if `x <= 2_147_483_647`) or `numeric`               |
| `pa.int64`                                                 | `integer` (if `abs(x) <= 2_147_483_647`) or `bit64::integer64` |
| `pa.float32`, `pa.float64`                                 | `numeric`                                                      |
| `pa.string`, `pa.large_string`                             | `character`                                                    |
| `pa.date32`                                                | `Date`                                                         |
| `pa.date64`, `pa.timestamp`                                | `POSIXct`                                                      |
| `pa.duration`                                              | `difftime(units='secs')`                                       |
| `pa.time32`, `pa.time64`                                   | `hms::hms`                                                     |
| `pa.dictionary(any integer type, pa.string(), ordered=0)`  | unordered `factor`                                             |
| `pa.dictionary(any integer type, pa.string(), ordered=1)`  | ordered `factor`                                               |
| `pa.null()`                                                | `vctrs::unspecified`                                           |

#### Polars data types

| Python                                      | R                                                              |
|---------------------------------------------|----------------------------------------------------------------|
| `Boolean`                                   | `logical`                                                      |
| `Int8`, `UInt8`, `Int16`, `UInt16`, `Int32` | `integer`                                                      |
| `UInt32`, `UInt64`                          | `integer` (if `x <= 2_147_483_647`) or `numeric`               |
| `Int64`                                     | `integer` (if `abs(x) <= 2_147_483_647`) or `bit64::integer64` |
| `Float32`, `Float64`                        | `numeric`                                                      |
| `Date`                                      | `Date`                                                         |
| `Datetime`                                  | `POSIXct`                                                      |
| `Duration`                                  | `difftime(units='secs')`                                       |
| `Time`                                      | `hms::hms`                                                     |
| `String`                                    | `character`                                                    |
| `Categorical`                               | unordered `factor`                                             |
| `Enum`                                      | ordered `factor`                                               |
| `Object`                                    | depends on the contents<sup>&Dagger;</sup>                     |
| `Null`                                      | `vctrs::unspecified`                                           | 
| `Binary`, `Decimal`, `List`, `Array`        | --                                                             |

#### Notes

<sup>&ast;</sup> For pandas Series and DataFrames, string indexes (and 
categorical indexes where the categories are strings) will be automatically
converted to `names`/`rownames`. The default index
(`pd.RangeIndex(len(python_object))`) will be ignored. All other indexes are
disallowed. 

<sup>&dagger;</sup> Because R does not support `POSIXct` and `Date` matrices or
arrays, dates and datetimes cannot be converted to R matrices or arrays.

<sup>&Dagger;</sup> For `dtype=object` and `dtype=pl.Object`, the output R type
depends on the contents, e.g. `'character'` if all elements are strings. Some
additional notes on ryp's handling of object data types:
- `None`, `np.nan`, `pd.NA`, `pd.NaT`, `np.datetime64('NaT')`, and 
  `np.timedelta64('NaT')` are all treated as missing values &ndash; even for 
  polars, where `np.nan` is ordinarily treated as a floating-point number 
  rather than a missing value. 
- Length-0 and all-missing data will be converted to the `vctrs::unspecified` R
  type (`vctrs` is part of the tidyverse). 
- If the elements are objects with a mix of types (or datetimes with a mix of
  time zones), Arrow will generally cause the conversion to fail, though mixes
  of related types (e.g. int and float) will be automatically cast to the
  common supertype and succeed. 
- Conversion will also fail if the contents are objects that are not 
  representable as R vector elements. This includes `bytes`/`bytearray` (which
  are only representable in R when scalar, as a `raw` vector) and Python
  containers (`list`, `tuple`, and  `dict`). 
- pandas `Timedelta` objects will be rounded down to the nearest microsecond,
  following the behavior of Arrow.

### R to Python (`to_py()`)

| R                                                                            | Python                                                                                                                                                                                        |
|------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `NULL`                                                                       | `None`                                                                                                                                                                                        |
| `NA`                                                                         | `None` (if scalar or `format='polars'`), `None`/`nan`/`pd.NA`/`pd.NaT`/`np.datetime64('NaT', 'us')`/`np.timedelta64('NaT', 'ns')`/etc. (if `format='numpy'` `'pandas'` or `'pandas-pyarrow'`) |
| `NaN`                                                                        | `nan`                                                                                                                                                                                         |
| length-1 vector, matrix or array, `squeeze == False`                         | scalar                                                                                                                                                                                        | 
| vector or 1D array, `format == 'numpy'`                                      | 1D NumPy array                                                                                                                                                                                |
| vector or 1D array, `format == 'pandas'` or `format == 'pandas-pyarrow'`     | pandas Series                                                                                                                                                                                 |
| vector or 1D array, `format == 'polars'`                                     | polars Series (if `index=False`) or two-column DataFrame                                                                                                                                      |
| matrix or `data.frame`, `format == 'numpy'`                                  | 2D NumPy array                                                                                                                                                                                |
| matrix or `data.frame`, `format == 'pandas'` or `format == 'pandas-pyarrow'` | pandas DataFrame                                                                                                                                                                              |
| matrix or `data.frame`, `format == 'polars'`                                 | polars DataFrame                                                                                                                                                                              |
| &ge; 3D array                                                                | NumPy array                                                                                                                                                                                   |  
| unnamed list                                                                 | `list`                                                                                                                                                                                        |
| named list, S3 object, S4 object, environment, S6 object                     | `dict`                                                                                                                                                                                        |
| `dgRMatrix`                                                                  | `csr_array(dtype='int32')`                                                                                                                                                                    |
| `dgCMatrix`                                                                  | `csc_array(dtype='int32')`                                                                                                                                                                    |
| `dgTMatrix`                                                                  | `coo_array(dtype='int32')`                                                                                                                                                                    |
| `lgRMatrix`, `ngRMatrix`                                                     | `csr_array(dtype=bool)`                                                                                                                                                                       |
| `lgCMatrix`, `ngCMatrix`                                                     | `csc_array(dtype=bool)`                                                                                                                                                                       |
| `lgTMatrix`, `ngTMatrix`                                                     | `coo_array(dtype=bool)`                                                                                                                                                                       |
| formula (`~`)                                                                | --                                                                                                                                                                                            |

#### Data types

| R                           | Python scalar                        | NumPy                                                    | pandas                                                   | pandas-pyarrow                                                 | polars                                     |
|-----------------------------|--------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------------|--------------------------------------------|
| `logical`                   | `bool`                               | `bool`                                                   | `bool`                                                   | `ArrowDtype(pa.bool_())`                                       | `Boolean`                                  |
| `integer`                   | `int`                                | `int32`                                                  | `int32`                                                  | `ArrowDtype(pa.int32())`                                       | `Int32`                                    |
| `bit64::integer64`          | `int`                                | `int64`                                                  | `int64`                                                  | `ArrowDtype(pa.int64())`                                       | `Int64`                                    |
| `numeric`                   | `float`                              | `float`                                                  | `float`                                                  | `ArrowDtype(pa.float64())`                                     | `Float64`                                  |
| `character`                 | `str`                                | `object` (with `str` elements)                           | `object` (with `str` elements)                           | `ArrowDtype(pa.string())`                                      | `String`                                   |
| `complex`                   | `complex`                            | `complex128`                                             | `complex128`                                             | `complex128`                                                   | --                                         |
| `raw`                       | `bytearray`                          | --                                                       | --                                                       | --                                                             | --                                         |
| unordered `factor`          | `str`                                | `object` (with `str` elements)                           | `CategoricalDtype(ordered=False)`                        | `ArrowDtype(pa.dictionary(pa.int8(), pa.string(), ordered=0))` | `Categorical`                              |
| ordered `factor`            | `str`                                | `object` (with `str` elements)                           | `CategoricalDtype(ordered=True)`                         | `ArrowDtype(pa.dictionary(pa.int8(), pa.string(), ordered=1))` | `Enum`                                     |
| `POSIXct` without time zone | `datetime.datetime`<sup>&ast;</sup>  | `datetime64[us]`<sup>&ast;</sup>                         | `datetime64[us]`<sup>&ast;</sup>                         | `ArrowDtype(pa.timestamp('us'))`<sup>&ast;</sup>               | `Datetime('us')`<sup>&ast;</sup>           |
| `POSIXct` with time zone    | `datetime.datetime`<sup>&ast;</sup>  | `datetime64[us]`<sup>&ast;</sup> (time zone discarded)   | `DatetimeTZDtype('us', time_zone)`<sup>&ast;</sup>       | `ArrowDtype(pa.timestamp('us', time_zone))`<sup>&ast;</sup>    | `Datetime('us, time_zone)`<sup>&ast;</sup> | 
| `POSIXlt`                   | `dict` of scalars                    | `dict` of NumPy arrays                                   | `dict` of pandas Series                                  | `dict` of pandas Series                                        | `dict` of polars Series                    |
| `Date`                      | `datetime.date`                      | `datetime64[D]`                                          | `datetime64[ms]`                                         | `ArrowDtype(pa.date32('day'))`                                 | `Date`                                     |
| `difftime`                  | `datetime.timedelta`<sup>&ast;</sup> | `timedelta64[ns]`                                        | `timedelta64[ns]`                                        | `ArrowDtype(pa.duration('ns'))`                                | `Duration(time_unit='ns')`                 |
| `hms::hms`                  | `datetime.time`<sup>&ast;</sup>      | `object` (with `datetime.time` elements)<sup>&ast;</sup> | `object` (with `datetime.time` elements)<sup>&ast;</sup> | `ArrowDtype(pa.time64('ns'))`<sup>&ast;</sup>                  | `Time`                                     |
| `vctrs::unspecified`        | `None`                               | `object` (with `None` elements)                          | `object` (with `None` elements)                          | `ArrowDtype(pa.null())`                                        | `Null`                                     |

<sup>&ast;</sup> Due to the limitations of conversion with Arrow, `POSIXct` and
`hms::hms` values are rounded down to the nearest microsecond when converting
to Python, except for `hms::hms` when converting to polars. `difftime` values
are also rounded down to the nearest microsecond, but only when converting to
scalar `datetime.timedelta` values (which cannot represent nanoseconds).

## Examples

1. Apply R's `scale()` function to a pandas DataFrame:

```python
>>> import pandas as pd
>>> from ryp import r, to_py, to_r, set_config
>>> set_config(to_py_format='pandas')
>>> data = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 3, 4]})
>>> to_r(data, 'data')
>>> r('data')
  a b
1 1 1
2 2 3
3 3 4
>>> r('data <- scale(data)')  # scale the R data.frame
>>> scaled_data = to_py('data')  # convert the R data.frame to Python
>>> scaled_data
     a         b
0 -1.0 -1.091089
1  0.0  0.218218
2  1.0  0.872872
```
Note: we could have just written `to_py('scale(data)')` instead of
`r('data <- scale(data)')` followed by `to_py('data')`.

2. Run a linear model on a polars DataFrame:

```python
>>> import polars as pl
>>> from ryp import r, to_py, to_r
>>> data = pl.DataFrame({'y': [7, 1, 2, 3, 6], 'x': [5, 2, 3, 2, 5]})
>>> to_r(data, 'data')
>>> r('model <- lm(y ~ x, data=data)')
>>> coef = to_py('summary(model)$coefficients', index='variable')
>>> p_value = coef.filter(variable='x').select('Pr(>|t|)')[0, 0]
>>> p_value
0.02831035772841884
```

3. Recursive conversion, showcasing all the keyword arguments to `to_r()` and
   `to_py()`:

```python
>>> import numpy as np
>>> from ryp import r, to_py, to_r
>>> arrays = {'ints': np.array([[1, 2], [3, 4]]),
...           'floats': np.array([[0.5, 1.5], [2.5, 3.5]])}
>>> to_r(arrays, 'arrays', format='data.frame',
...      rownames = ['row1', 'row2'], colnames = ['col1', 'col2'])
>>> r('arrays')
$ints
     col1 col2
row1    1    2
row2    3    4

$floats
     col1 col2
row1  0.5  1.5
row2  2.5  3.5
>>> arrays = to_py('arrays', format='pandas', index='foo')
>>> arrays['ints']
      col1  col2
foo
row1     1     2
row2     3     4
>>> arrays['floats']
      col1  col2
foo
row1   0.5   1.5
row2   2.5   3.5
```
