from __future__ import annotations
import cffi
import _cffi_backend
import datetime
import numpy as np
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import warnings
from typing import Any, Literal


__version__ = '0.2.0'


class ignore_sigint:
    """
    Ignore Ctrl + C when importing certain modules, to avoid errors due to
    incomplete imports.
    """
    def __enter__(self):
        if _main_thread:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    def __exit__(self, *_):
        if _main_thread:
            signal.signal(signal.SIGINT, signal.default_int_handler)


def _get_rlib_path(R_home: str) -> str:
    """
    Get the path to the R shared library.
    
    Based on the rpy2.situation.get_rlib_path() function.
    
    Args:
        R_home: the path to the R home directory
    
    Returns:
        The path to the R shared library.
    """
    system = platform.system()
    if system == 'Darwin':
        return os.path.join(R_home, 'lib', 'libR.dylib')
    elif system == 'Windows':
        return os.path.join(
            R_home, 'bin', 'x64' if sys.maxsize > 2 ** 32 else 'i386', 'R.dll')
    else:
        return os.path.join(R_home, 'lib', 'libR.so')


def _get_R_home_and_rlib_path() -> tuple[str, str]:
    """
    Get R's home directory, first by checking the environment variable R_HOME,
    then (if not defined or misconfigured) by running `R RHOME`. Also get the
    path to the R shared library.
    
    Based on the rpy2.situation.get_rlib_path() and rpy2.situation.get_r_home()
    functions.
    
    Returns:
        The R home directory and the path to the R shared library.
    """
    # If the environment variable R_HOME is defined, check that it exists and
    # that it is properly configured: it must exist, and the R shared library
    # must be found at the expected location under it (see _get_rlib_path()).
    R_home = os.environ.get('R_HOME')
    if R_home is not None:
        if os.path.isdir(R_home):
            rlib_path = _get_rlib_path(R_home)
            if os.path.exists(rlib_path):
                return R_home, rlib_path
            warning_message = (
                f'the environment variable R_HOME is set to {R_home}, which '
                f'is a valid directory, but the R shared library was not '
                f'found at the expected location {rlib_path} inside this '
                f'directory. The environment variable may be misconfigured.')
            warnings.warn(warning_message, RuntimeWarning)
        elif os.path.isfile(R_home):
            warning_message = (
                f'the environment variable R_HOME is set to a file, not a '
                f'directory: {R_home}. You may want to unset this environment '
                f'variable since it seems to be misconfigured.')
            warnings.warn(warning_message, RuntimeWarning)
        else:
            warning_message = (
                f'the environment variable R_HOME is set to a path that does '
                f'not exist: {R_home}. You may want to unset this environment '
                f'variable since it seems to be misconfigured.')
            warnings.warn(warning_message, RuntimeWarning)
    
    # If R_HOME is not defined or misconfigured, fall back to running `R RHOME`
    try:
        R_home = subprocess.run('R RHOME', shell=True, capture_output=True,
                                check=True, text=True).stdout.rstrip()
    except subprocess.CalledProcessError as e:
        error_message = (
            "the command 'R RHOME' did not run successfully, so the R "
            "home directory could not be determined. Is R installed?\nIf so, "
            "set the R_HOME environment variable or add the directory "
            "containing the R executable to your PATH.")
        raise RuntimeError(error_message) from e
    rlib_path = _get_rlib_path(R_home)
    if os.path.isfile(rlib_path):
        return R_home, rlib_path
    else:
        error_message = (
            f"internal ryp error: the command 'R RHOME' indicated that the R "
            f"home directory is {R_home}, but the R shared library was not "
            f"found at the expected location {rlib_path} inside this "
            f"directory")
        raise RuntimeError(error_message)


def _initialize_ryp() -> None:
    """
    Initialize ryp, if not already initialized.
    
    Based on the rpy2.rinterface_lib.embedded._initr() function.
    """
    global _ffi, _rlib, _ryp_thread, _ryp_PID, _main_thread, _jupyter_notebook
    # Disallow running ryp from a forked process if it has already been
    # initialized in the parent process
    current_PID = os.getpid()
    if _ryp_PID is not None and _ryp_PID != current_PID:
        error_message = (
            "r(), to_py() and to_r() cannot be called in a forked "
            "process if any of them were called in the parent "
            "process prior to forking; as discussed in the README, "
            "either switch to the 'forkserver' or 'spawn' "
            "multiprocessing backends, or avoid using ryp outside "
            "the parallel part of your code")
        raise RuntimeError(error_message)
    # Use double-checked locking when deciding whether to initialize
    if _rlib is None:
        with _init_lock:
            if _rlib is None:
                # Set _ryp_PID and _ryp_thread to the initializing process's
                # process ID and the initializing thread's thread ID
                _ryp_PID = current_PID
                _ryp_thread = threading.current_thread()
                # Define a foreign function interface (FFI) object
                _ffi = cffi.FFI()
                # Add functions from R's C API to the FFI
                FD_SETSIZE = 1024
                NFDBITS = 8 * _ffi.sizeof('unsigned long')
                R_header = '''
                typedef unsigned int SEXPTYPE;
                typedef struct SEXPREC *SEXP;
                typedef struct SEXPREC {
                    struct sxpinfo_struct sxpinfo;
                    struct SEXPREC *attrib;
                    struct SEXPREC *gengc_next_node, *gengc_prev_node;
                    union {
                        struct primsxp_struct primsxp;
                        struct symsxp_struct symsxp;
                        struct listsxp_struct listsxp;
                        struct envsxp_struct envsxp;
                        struct closxp_struct closxp;
                        struct promsxp_struct promsxp;
                    } u;
                } SEXPREC;
                struct sxpinfo_struct {
                    SEXPTYPE type : 5;
                    unsigned int scalar: 1;
                    unsigned int alt : 1;
                    unsigned int obj : 1;
                    unsigned int gp : 16;
                    unsigned int mark : 1;
                    unsigned int debug : 1;
                    unsigned int trace : 1;
                    unsigned int spare : 1;
                    unsigned int gcgen : 1;
                    unsigned int gccls : 3;
                    unsigned int named : 16;
                    unsigned int extra : 32;
                };
                struct primsxp_struct { int offset; };
                struct symsxp_struct { struct SEXPREC *pname;
                                       struct SEXPREC *value;
                                       struct SEXPREC *internal; };
                struct listsxp_struct { struct SEXPREC *carval;
                                        struct SEXPREC *cdrval;
                                        struct SEXPREC *tagval; };
                struct envsxp_struct { struct SEXPREC *frame;
                                       struct SEXPREC *enclos;
                                       struct SEXPREC *hashtab; };
                struct closxp_struct { struct SEXPREC *formals;
                                       struct SEXPREC *body;
                                       struct SEXPREC *env; };
                struct promsxp_struct { struct SEXPREC *value;
                                        struct SEXPREC *expr;
                                        struct SEXPREC *env; };

                typedef enum { FALSE = 0, TRUE } Rboolean;
                typedef unsigned char Rbyte;
                typedef struct { double r; double i; } Rcomplex;
                typedef ''' + ("ptrdiff_t" if _ffi.sizeof('size_t') > 4 else
                               "int") + ''' R_xlen_t;

                typedef void (*InputHandlerProc)(void *userData);
                typedef struct _InputHandler {
                  int activity;
                  int fileDescriptor;
                  InputHandlerProc handler;
                  struct _InputHandler *next;
                  int active;
                  void *userData;
                } InputHandler;

                typedef struct {
                    unsigned long fds_bits[''' + \
                    str(FD_SETSIZE // NFDBITS) + '''];
                } fd_set;
                struct timeval {
                    long tv_sec;
                    long tv_usec;
                };

                const SEXPTYPE NILSXP = 0;
                const SEXPTYPE SYMSXP = 1;
                const SEXPTYPE LISTSXP = 2;
                const SEXPTYPE CLOSXP = 3;
                const SEXPTYPE ENVSXP = 4;
                const SEXPTYPE PROMSXP = 5;
                const SEXPTYPE LANGSXP = 6;
                const SEXPTYPE SPECIALSXP = 7;
                const SEXPTYPE BUILTINSXP = 8;
                const SEXPTYPE CHARSXP = 9;
                const SEXPTYPE LGLSXP = 10;
                const SEXPTYPE INTSXP = 13;
                const SEXPTYPE REALSXP = 14;
                const SEXPTYPE CPLXSXP = 15;
                const SEXPTYPE STRSXP = 16;
                const SEXPTYPE DOTSXP = 17;
                const SEXPTYPE ANYSXP = 18;
                const SEXPTYPE VECSXP = 19;
                const SEXPTYPE EXPRSXP = 20;
                const SEXPTYPE BCODESXP = 21;
                const SEXPTYPE EXTPTRSXP = 22;
                const SEXPTYPE WEAKREFSXP = 23;
                const SEXPTYPE RAWSXP = 24;
                const SEXPTYPE S4SXP = 25;
                const SEXPTYPE NEWSXP = 30;
                const SEXPTYPE FREESXP = 31;
                const SEXPTYPE FUNSXP = 99;

                const int CE_UTF8 = 1;
                const int PARSE_OK = 1;

                extern SEXP R_BaseEnv;
                extern uintptr_t R_CStackLimit;
                extern SEXP R_ClassSymbol;
                extern SEXP R_DimNamesSymbol;
                extern SEXP R_DimSymbol;
                extern SEXP R_DoubleColonSymbol;
                extern SEXP R_GlobalEnv;
                extern InputHandler *R_InputHandlers;
                extern SEXP R_LevelsSymbol;
                extern SEXP R_NamesSymbol;
                extern int R_NaInt;
                extern SEXP R_NilValue;
                extern char R_ParseContext[256];
                extern int R_ParseContextLast;
                extern char R_ParseErrorMsg[256];
                extern SEXP R_RowNamesSymbol;

                SEXP CAR(SEXP);
                SEXP CDR(SEXP);
                SEXP CDDR(SEXP);
                Rcomplex *COMPLEX(SEXP);
                Rcomplex COMPLEX_ELT(SEXP, R_xlen_t);
                int *INTEGER(SEXP);
                int INTEGER_ELT(SEXP, R_xlen_t);
                Rboolean LOGICAL_ELT(SEXP, R_xlen_t);
                Rbyte *RAW(SEXP);
                double *REAL(SEXP);
                const char* R_CHAR(SEXP);
                SEXP R_FindNamespace(SEXP);
                SEXP R_ParseVector(SEXP, int, int *, SEXP);
                int R_ReplDLLdo1(void);
                void R_ReplDLLinit(void);
                extern int R_SignalHandlers;
                const char *R_curErrorBuf(void);
                SEXP R_do_new_object(SEXP);
                SEXP R_do_slot(SEXP, SEXP);
                SEXP R_do_slot_assign(SEXP, SEXP, SEXP);
                SEXP R_getClassDef(const char *);
                SEXP R_lsInternal(SEXP, Rboolean);
                void R_runHandlers(InputHandler *, fd_set *);
                SEXP R_tryCatchError(SEXP (*)(void *), void *,
                                     SEXP (*)(SEXP, void *), void *);
                SEXP R_tryEvalSilent(SEXP, SEXP, int *);
                SEXP Rf_ScalarInteger(int);
                SEXP Rf_ScalarLogical(int);
                SEXP Rf_ScalarReal(double);
                SEXP Rf_allocVector(SEXPTYPE, R_xlen_t);
                void Rf_defineVar(SEXP, SEXP, SEXP);
                SEXP Rf_eval(SEXP, SEXP);
                SEXP Rf_findFun(SEXP, SEXP);
                SEXP Rf_findVar(SEXP, SEXP);
                SEXP Rf_findVarInFrame(SEXP, SEXP);
                SEXP Rf_getAttrib(SEXP, SEXP);
                Rboolean Rf_inherits(SEXP, const char *);
                int Rf_initialize_R(int, char **);
                SEXP Rf_install(const char *);
                SEXP Rf_lang1(SEXP);
                SEXP Rf_lang2(SEXP, SEXP);
                SEXP Rf_lang3(SEXP, SEXP, SEXP);
                SEXP Rf_lang4(SEXP, SEXP, SEXP, SEXP);
                SEXP Rf_lcons(SEXP, SEXP);
                SEXP Rf_mkCharLenCE(const char *, int, int);
                SEXP Rf_protect(SEXP);
                SEXP Rf_setAttrib(SEXP, SEXP, SEXP);
                void Rf_unprotect(int);
                R_xlen_t Rf_xlength(SEXP);
                void SET_LOGICAL_ELT(SEXP, R_xlen_t, int);
                void SET_STRING_ELT(SEXP, R_xlen_t, SEXP);
                void SET_TAG(SEXP, SEXP);
                SEXP SET_VECTOR_ELT(SEXP, R_xlen_t, SEXP);
                SEXP STRING_ELT(SEXP, R_xlen_t);
                int TYPEOF(SEXP);
                SEXP VECTOR_ELT(SEXP, R_xlen_t);
                void setup_Rmainloop(void);

                int GA_doevent(void);
                int GA_initapp(int argc, char *argv[]);

                int select(int, fd_set *, fd_set *, fd_set *,
                           struct timeval *);
                '''
                _ffi.cdef(R_header)
                # Get the R home directory and shared object (.so/.dll) file
                R_home, rlib_path = _get_R_home_and_rlib_path()
                # Load the .so file
                if platform.system() == 'Windows':
                    # Add the directory containing the R DLL to the PATH, to
                    # work around github.com/python-cffi/cffi/issues/64
                    os.environ['PATH'] += f';{os.path.dirname(rlib_path)}'
                _rlib = _ffi.dlopen(rlib_path)
                # Error out if R has already been initialized (e.g. by rpy2)
                if _rlib.R_NilValue != _ffi.NULL:
                    error_message = (
                        "R has already been initialized; did you (or a "
                        "library you're using) import rpy2 earlier?")
                    raise RuntimeError(error_message)
                # Set R_HOME (required for Rf_initialize_R() to not crash)
                os.environ['R_HOME'] = R_home
                # Before initializing R, disable KeyboardInterrupts until the
                # end of initialization, to avoid a corrupted partial
                # initialization. If not on the main thread, KeyboardInterrupts
                # cannot be disabled due to the limitations of Python's signal
                # module, so skip this step.
                _main_thread = \
                    threading.current_thread() is threading.main_thread()
                if _main_thread:
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                # Also disable R's internal signal handlers; without this line,
                # if the user triggers a KeyboardInterrupt before the module is
                # fully imported, their first r() command will fail at
                # R_tryEvalSilent() with no error message
                _rlib.R_SignalHandlers = 0
                # Initialize R
                args = [_ffi.new('char[]', arg) for arg in
                        (b'R', b'--quiet', b'--no-save', b'--args', b'ryp')]
                _rlib.Rf_initialize_R(len(args), args)
                _rlib.R_CStackLimit = _ffi.cast('uintptr_t', -1)
                _rlib.setup_Rmainloop()
                # Do ryp initialization that requires R to be initialized
                r(f'q = quit = function(...) {{ cat("Press '
                  f'{_EOF_instructions} to exit the Python terminal, or run '
                  f'exit()\n") }}; options(arrow.int64_downcast = FALSE)')
                # If inside a Jupyter notebook, set up inline plotting
                try:
                    ipython = get_ipython()
                except NameError:
                    _jupyter_notebook = False
                else:
                    _jupyter_notebook = 'IPKernelApp' in ipython.config
                if _jupyter_notebook:
                    from IPython.display import display, SVG
                    with _RMemory(_rlib) as rmemory:
                        if not _require(b'svglite', rmemory):
                            error_message = (
                                'please install the svglite R package to use '
                                'inline plotting in Jupyter notebooks')
                            raise ImportError(error_message)
                    # Make a custom plotting device that saves each plot to a
                    # temp file as SVG
                    r(f'.tempfile = tempfile(fileext = "png"); '
                      f'options(device=function() {{ svglite(.tempfile, '
                      f'width={_config["plot_width"]}, '
                      f'height={_config["plot_height"]})}})')
                    temp_file = to_py('.tempfile')

                    def _plot_jupyter_inline(*_: Any) -> None:
                        """
                        If the user just ran a cell that created a plot in R,
                        display it. Executed after each cell is run in the
                        Jupyter notebook.
                        """
                        # If temp_file doesn't exist, nothing's been plotted in
                        # this cell
                        if not os.path.exists(temp_file):
                            return
                        # Save the plot to temp_file in R
                        r('dev.off()')
                        # Load the plot from temp_file into Python, as a
                        # bytestring
                        with open(temp_file, 'rb') as file:
                            data = file.read()
                        # Display the plot in the Jupyter notebook
                        display(SVG(data=data))
                        # Delete temp_file
                        os.unlink(temp_file)

                    ipython.events.register('post_run_cell',
                                            _plot_jupyter_inline)
                # Now that we're at the very end of initialization, reset the
                # KeyboardInterrupt signal handler to the default one. (This
                # would still be necessary even if we had not installed a
                # custom signal handler above, since Ctrl + C would not work
                # after setup_Rmainloop() otherwise.)
                if _main_thread:
                    signal.signal(signal.SIGINT, signal.default_int_handler)
    # To ensure thread-safety, keep track of which thread initialized ryp
    # and disallow execution by any other thread (within the same
    # process).
    if threading.current_thread() is not _ryp_thread:
        error_message = (
            "the R interpreter is not thread-safe, so ryp can't be used "
            "by multiple Python threads simultaneously; see the README "
            "for alternative parallelization strategies")
        raise RuntimeError(error_message)


class _RMemory:
    """
    A utility class for managing R's memory protection with Rf_protect() and
    Rf_unprotect(). Based on the rpy2.rinterface_lib.memorymanagement.rmemory
    context manager.
    """
    def __init__(self: _RMemory, rlib: _cffi_backend.Lib) -> None:
        self.rlib = rlib
        self.count = 0
    
    def __enter__(self: _RMemory) -> _RMemory:
        return self
    
    def protect(self: _RMemory,
                robject: _cffi_backend._CDataBase) -> _cffi_backend._CDataBase:
        robject = self.rlib.Rf_protect(robject)
        self.count += 1
        return robject
    
    def __exit__(self: _RMemory, *_) -> None:
        self.rlib.Rf_unprotect(self.count)


def _bytestring_to_character_vector(bytestring: bytes, rmemory: _RMemory) -> \
        _cffi_backend._CDataBase:
    """
    Convert a Python bytestring into a length-1 R character vector.
    
    Args:
        bytestring: a Python bytestring (bytes)
        rmemory: an instance of the _RMemory class

    Returns:
        A length-1 R character vector (STRSXP).
    """
    STRSXP = rmemory.protect(_rlib.Rf_allocVector(_rlib.STRSXP, 1))
    _rlib.SET_STRING_ELT(STRSXP, 0, _rlib.Rf_mkCharLenCE(
        bytestring, len(bytestring), _rlib.CE_UTF8))
    return STRSXP


def _string_to_character_vector(string: str,
                                rmemory: _RMemory) -> _cffi_backend._CDataBase:
    """
    Convert a Python string into a length-1 R character vector.
    
    Args:
        string: a Python string
        rmemory: an instance of the _RMemory class

    Returns:
        A length-1 R character vector (STRSXP).
    """
    return _bytestring_to_character_vector(string.encode('utf-8'), rmemory)


def _call(function_call: _cffi_backend._CDataBase, rmemory: _RMemory,
          error_description: str, *,
          environment: _cffi_backend._CDataBase = None) -> \
        _cffi_backend._CDataBase:
    """
    Call an R function.
    
    Args:
        function_call: an R pairlist representing a function call. The first
                       element is the function; the rest are the arguments.
        rmemory: an instance of the _RMemory class
        error_description: a description to print as part of the error message
                           if there is an error; will be prefixed by 'Internal
                           ryp error: ' and suffixed with the R error message
        environment: the R environment (or R6 object) to evaluate the function
                     call inside; by default, baseenv()
    
    Returns:
        The R object returned by the function.
    """
    if environment is None:
        environment = _rlib.R_BaseEnv
    status = _ffi.new('int[1]')
    result = rmemory.protect(
        _rlib.R_tryEvalSilent(function_call, environment, status))
    if status[0] != 0:
        error_message = \
            _ffi.string(_rlib.R_curErrorBuf()).decode('utf-8').rstrip()
        if error_message.startswith("Error in vec_to_Array(x, type) : \n  "
                                    "STRING_ELT() can only be applied to a "
                                    "'character vector', not a"):
            # work around github.com/apache/arrow/issues/40886
            R_statement = re.findall('`(.*)`', error_description)[0]
            error_message = \
                f"{R_statement!r} is a POSIXct with an invalid 'tz' value"
            raise ValueError(error_message)
        else:
            error_message = \
                f'internal ryp error: {error_description}\n{error_message}'
            raise RuntimeError(error_message)
    return result


def _check_R_variable_name(R_variable_name: str) -> None:
    """
    Raise an error if R_variable_name is not a valid variable name in R.
    
    Args:
        R_variable_name: the R variable name to be checked
    """
    if not R_variable_name:
        error_message = 'R_variable_name is an empty string'
        raise ValueError(error_message)
    if R_variable_name[0] == '.':
        if len(R_variable_name) > 1 and R_variable_name[1].isdigit():
            error_message = (
                f'R_variable_name {R_variable_name!r} starts with a period '
                f'followed by a digit, which is not a valid R variable name')
            raise ValueError(error_message)
    elif not R_variable_name[0].isidentifier():
        error_message = (
            f'R_variable_name {R_variable_name!r} must start with a letter, '
            f'number, period or underscore')
        raise ValueError(error_message)
    if not re.fullmatch(r'[\w.]*', R_variable_name[1:]):
        invalid_characters = \
            sorted(set(re.findall(r'[^\w.]',
                                  ''.join(dict.fromkeys(R_variable_name)))))
        if len(invalid_characters) == 1:
            description = f'the character {invalid_characters[0]!r}'
        else:
            description = f"the characters " + ", ".join(
                f'{character!r}' for character in invalid_characters) + \
                f' and {invalid_characters[-1]!r}'
        error_message = (
            f'R_variable_name {R_variable_name!r} contains {description}, but '
            f'must contain only letters, numbers, periods and underscores')
        raise ValueError(error_message)
    if R_variable_name in _R_keywords or (R_variable_name.startswith('..') and
                                          R_variable_name[2:].isdigit()):
        error_message = (
            f'R_variable_name {R_variable_name!r} is a reserved keyword in R, '
            f'and cannot be used as a variable name')
        raise ValueError(error_message)


def _is_valid_R_variable_name(R_variable_name: str) -> bool:
    """
    Return whether R_variable_name is a valid variable name in R.
    
    Args:
        R_variable_name: the R variable name to be checked

    Returns:
        True if R_variable_name is a valid R variable name, False otherwise.
    """
    return R_variable_name and (
            R_variable_name[0].isidentifier() or (
                R_variable_name[0] == '.' and (
                    len(R_variable_name) == 1 or
                    R_variable_name[1].isdigit()))) and \
        re.fullmatch(r'[\w.]*', R_variable_name[1:]) and \
        R_variable_name not in _R_keywords and not \
            (R_variable_name.startswith('..') and
             R_variable_name[2:].isdigit())


def _convert_names(names: Any, names_type: Literal['rownames', 'colnames'],
                   python_object: Any, python_object_name: str,
                   rmemory: _RMemory) -> _cffi_backend._CDataBase:
    """
    Convert rownames or colnames to R, raising an error if they are invalid.
    
    Args:
        names: a container (list, tuple, array, or Series) of Python strings
               that will be converted to R and used as the rownames or colnames
        names_type: whether names are rownames or colnames
        python_object: the python object that the names are with respect to
        python_object_name: the name of python_object
        rmemory: an instance of the _RMemory class

    Returns:
        The converted rownames or colnames.
    """
    if not hasattr(names, '__len__') or not hasattr(names, '__getitem__') or \
            isinstance(names, (str, bytes, bytearray)):
        error_message = \
            f'{names_type} has unsupported type {type(names).__name__!r}'
        raise TypeError(error_message)
    if names_type == 'rownames':
        # sparse arrays/matrices don't have len(), so use shape
        python_object_length = \
            python_object.shape[0] if hasattr(python_object, 'shape') else \
                len(python_object) if hasattr(python_object, '__len__') else 1
        if isinstance(python_object, (int, float, str, complex)):
            if len(names) != 1:
                error_message = (
                    f'rownames have length {len(names):,}, but '
                    f'{python_object_name} is a scalar, specifically of type '
                    f'{type(python_object).__name__!r}, so rownames must have '
                    f'a length of 1')
                raise ValueError(error_message)
        elif not isinstance(python_object, (list, tuple, dict)) and \
                len(names) != python_object_length:
            error_message = (
                f'rownames have length {len(names):,}, but '
                f'{python_object_name} has length {python_object_length:,}')
            raise ValueError(error_message)
    else:
        if not isinstance(python_object, (list, tuple, dict)) and \
                len(names) != python_object.shape[1]:
            error_message = (
                f'colnames have length {len(names):,}, but '
                f'{python_object_name}.shape[1] is {python_object.shape[1]:,}')
            raise ValueError(error_message)
    # Convert list and tuple to Arrow to allow conversion to R vectors rather
    # than lists, but leave everything else as-is (among other things, set,
    # frozenset and dict will still convert to lists and give an error)
    if isinstance(names, (list, tuple)):
        with ignore_sigint():
            import pyarrow as pa
        names = pa.array(names) if names else pa.array([], type=pa.string())
    try:
        converted_names = to_r(names, (names_type, rmemory))
    except TypeError as e:
        error_message = \
            f'{names_type} has unsupported type {type(names).__name__!r}'
        raise TypeError(error_message) from e
    # If a factor, coerce to a character vector; otherwise, give an error
    # message if the result is not a character vector
    if _rlib.Rf_inherits(converted_names, b'factor'):
        function_call = rmemory.protect(
            _rlib.Rf_lang2(_rlib.Rf_install(b'as.character'), converted_names))
        converted_names = _call(function_call, rmemory,
                                'cannot convert factor to character vector')
    elif _rlib.TYPEOF(converted_names) != _rlib.STRSXP:
        dtype_string = str(names.dtype) if hasattr(names, 'dtype') else None
        error_message = (
            f'{names_type} could not be converted to an R character vector or '
            f'factor; its type is {type(names).__name__!r}' + (
                f' and its data type is {dtype_string!r}'
                if dtype_string is not None else ''))
        raise TypeError(error_message)
    return converted_names


def _is_supported_pyarrow_dtype(pyarrow_dtype: 'DataType') -> bool:
    """
    Return whether pyarrow_dtype is supported by ryp.
    
    Args:
        pyarrow_dtype: the pyarrow dtype to check
    """
    import pyarrow as pa
    return pa.types.is_integer(pyarrow_dtype) or \
        pa.types.is_floating(pyarrow_dtype) or \
        pa.types.is_temporal(pyarrow_dtype) or \
        pa.types.is_string(pyarrow_dtype) or \
        pa.types.is_large_string(pyarrow_dtype) or (
                pa.types.is_dictionary(pyarrow_dtype) and
                pa.types.is_string(pyarrow_dtype.value_type)) or \
        pa.types.is_null(pyarrow_dtype)


def _as_data_frame(result: _cffi_backend._CDataBase, rmemory: _RMemory) -> \
        _cffi_backend._CDataBase:
    """
    Convert an R list to data.frame; set check.names=FALSE to not name-mangle.
    
    Handle vctrs_unspecified: as.data.frame(list(a=vctrs::unspecified(1)))
    gives the error `cannot coerce class '"vctrs_unspecified"' to a data.frame`
    on Windows, so strip vctrs_unspecified beforehand and re-add after.
    
    Args:
        result: an R list
        rmemory: an instance of the _RMemory class

    Returns:
        result, converted to a data.frame.
    """
    if platform.system() == 'Windows':
        vctrs_unspecified_cols = []
        for i in range(_rlib.Rf_xlength(result)):
            R_column = _rlib.VECTOR_ELT(result, i)
            if _rlib.Rf_inherits(R_column, b'vctrs_unspecified'):
                _rlib.Rf_setAttrib(R_column, _rlib.R_ClassSymbol,
                                   _rlib.R_NilValue)
                vctrs_unspecified_cols.append(i)
    function_call = rmemory.protect(_rlib.Rf_lang3(
        _rlib.Rf_install(b'as.data.frame'),
        result,
        rmemory.protect(_rlib.Rf_ScalarLogical(0))))
    _rlib.SET_TAG(_rlib.CDDR(function_call), _rlib.Rf_install(b'check.names'))
    result = _call(function_call, rmemory, 'cannot convert list to data.frame')
    if platform.system() == 'Windows':
        new_classes = _bytestring_to_character_vector(b'vctrs_unspecified',
                                                      rmemory)
        for i in vctrs_unspecified_cols:
            _rlib.Rf_setAttrib(_rlib.VECTOR_ELT(result, i),
                               _rlib.R_ClassSymbol,
                               new_classes)
    return result


def _check_object_elements(python_object: Any,
                           python_object_description: str,
                           is_pandas: bool,
                           allowed_types: tuple[type, ...] | type,
                           disallowed_types: tuple[type, ...] | type,
                           allowed_dtypes: set) -> None:
    """
    Raise a TypeError if not all elements of python_object have the specified
    types/dtypes.
    
    Args:
        python_object: a 1D NumPy array, pandas Series or Index, or polars
                       Series with dtype=object or dtype=pl.Object
        python_object_description: a description of python_object
        is_pandas: whether python_object is a pandas Series or Index
        allowed_types: the allowed types of the elements
        disallowed_types: the unallowed types of the elements, necessary if
                          they are subtypes of allowed_types (e.g. bool -> int)
        allowed_dtypes: the allowed NumPy dtypes of the elements, if np.generic
    """
    values = python_object.values if is_pandas else python_object
    if not all(element is None or isinstance(element, allowed_types) and
               not isinstance(element, disallowed_types) or
               isinstance(element, np.generic) and
               element.dtype.type in allowed_dtypes for element in values):
        error_message = (
            f'{python_object_description} and elements with a mix of types, '
            f'and cannot be represented as any single R vector type')
        raise TypeError(error_message)


def _convert_object_to_arrow(python_object: Any,
                             python_object_description: str,
                             is_pandas: bool = False,
                             is_polars: bool = False) -> Any:
    """
    Converts 1D data with object dtype to Arrow (or to complex, if complex).
    
    Args:
        python_object: a 1D NumPy array, pandas Series or Index, or polars
                       Series with dtype=object or dtype=pl.Object
        python_object_description: a description of python_object
        is_pandas: whether python_object is a pandas Series or Index
        is_polars: whether python_object is a polars Series

    Returns:
        python_object, converted to Arrow (or to complex, if complex).
    """
    with ignore_sigint():
        import pyarrow as pa
    arrow = None
    try:
        with ignore_sigint():
            import pandas as pd
        has_pandas = True
    except ImportError:
        has_pandas = False
    # Normalize all missing values to None. Why? pa.array(from_pandas=True)
    # converts None, np.nan, pd.NA, pd.NaT to null in object-dtyped arrays, but
    # pa.array(from_pandas=False) only converts None to null and raises errors
    # on the others. Meanwhile, np.datetime64('NaT') and np.timedelta64('NaT')
    # are handled inconsistently depending on context, and are incorrectly cast
    # to non-object dtypes due to a bug (github.com/numpy/numpy/issues/26177).
    python_object = python_object.to_numpy(writable=True) if is_polars else \
        python_object.copy()
    if has_pandas:
        if isinstance(python_object, pd.Index):
            python_object = python_object.to_series().reset_index(drop=True)
        # Handle pd.NA/pd.NaT, converting everything to NaN (pandas doesn't
        # support filling missing values with None using fillna())
        if not is_pandas:
            python_object = pd.Series(python_object, dtype=object)
        try:
            with pd.option_context('future.no_silent_downcasting', True):
                python_object.fillna(np.nan, inplace=True)
        except pd._config.config.OptionError:
            # old version of pandas without future.no_silent_downcasting
            python_object.fillna(np.nan, inplace=True)
        if not is_pandas:
            python_object = python_object.values
    # This works for everything except pd.NA/pd.NaT, which are handled above
    # if pandas is installed and are impossible if pandas is not installed
    python_object[python_object != python_object] = None
    while True:
        is_complex = False
        is_float128 = False
        is_period = False
        is_datetime = False
        try:
            if np.__version__[0] == 1:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', np.ComplexWarning)
                    arrow = pa.array(python_object)
            else:
                arrow = pa.array(python_object)
        except pa.ArrowInvalid as e:
            error_string = str(e)
            if error_string.endswith('with type complex: did not recognize '
                                     'Python value type when inferring an '
                                     'Arrow data type') or \
                    re.search(r'with type (numpy\.complex64|numpy\.complex128|'
                              r'complex): tried to convert to', error_string):
                is_complex = True
            elif error_string.endswith('with type Period: did not recognize '
                                       'Python value type when inferring an '
                                       'Arrow data type'):
                is_period = True
            elif error_string == (
                    'numpy.datetime64 scalars cannot be mixed with other '
                    'Python scalar values currently'):
                # Mix of np.datetime64 and datetime.datetime
                is_datetime = True
            elif re.match('Cannot mix NumPy dtypes u?int(8|16|32|64) and '
                          'u?int(8|16|32|64)', error_string):
                # Mix of NumPy integer generics; cast to the supertype
                # (following the same rules as Arrow) after checking that only
                # numeric types are present
                allowed_dtypes = {np.complex64, np.complex128, np.int8,
                                  np.int16, np.int32, np.int64, np.uint8,
                                  np.uint16, np.uint32, np.uint64, np.float16,
                                  np.float32, np.float64}
                if platform.system() != 'Windows':
                    allowed_dtypes.add(np.float128)
                _check_object_elements(
                    python_object, python_object_description, is_pandas,
                    allowed_types=(int, float, complex), disallowed_types=bool,
                    allowed_dtypes=allowed_dtypes)
                values = python_object.values if is_pandas else python_object
                types = set(map(type, values))
                if {complex, np.complex64, np.complex128} & types:
                    supertype = complex
                elif {float, np.float16, np.float32, np.float64} & types or \
                        platform.system() != 'Windows' and \
                        np.float128 in types:
                    supertype = float
                elif {np.uint32, np.uint64} & types:
                    supertype = np.int32 \
                        if python_object.max() <= 2_147_483_647 else float
                elif np.int64 in types:
                    supertype = np.int32 \
                        if python_object.max() <= 2_147_483_647 and \
                           python_object.min() >= -2_147_483_647 else int
                else:
                    supertype = np.int32
                python_object = python_object.astype(supertype)
                continue
            else:
                # e.g. "Integer too large for date32" when mixing a
                # datetime.date and an int > 2_147_483_647 on Linux
                error_message = (
                    f'{python_object_description} and elements with a mix of '
                    f'types, and cannot be represented as any single R vector '
                    f'type')
                raise TypeError(error_message) from e
        except pa.ArrowNotImplementedError as e:
            error_string = str(e)
            if (
                    # datetime64[m] or datetime64[h]
                    error_string == 'Unsupported datetime64 time unit' or
                    # mix of np.datetime64 and pd.Timestamp
                    error_string.startswith('Expected np.datetime64 but got: '
                                            'timestamp') or
                    # mix of datetime64[D] and other datetime64s
                    error_string == 'Expected np.datetime64 but got: '
                                    'date32[day]'):
                is_datetime = True
            elif (
                    # timedelta64 with time unit of >= minutes
                    error_string == 'Unsupported timedelta64 time unit' or
                    # mix of timedelta64[ns] and other timedeltas
                    error_string.startswith('Expected np.timedelta64 but got: '
                                            'duration')):
                _check_object_elements(
                    python_object, python_object_description, is_pandas,
                    allowed_types=(datetime.timedelta, pd.Timedelta)
                                  if has_pandas else (datetime.timedelta,),
                    disallowed_types=(), allowed_dtypes={np.timedelta64})
                try:
                    python_object = python_object.astype('timedelta64[ns]')
                except TypeError:
                    # e.g. "Cannot cast NumPy timedelta64 scalar from metadata
                    # [Y] to [ns] according to the rule 'same_kind'"; occurs
                    # when mixing timedelta64 with time unit of months or years
                    # with missing values/NaNs
                    python_object = np.array([
                        np.timedelta64('NaT') if element is None else
                        element.astype('timedelta64[ns]')
                        if isinstance(element, np.timedelta64) else element
                        for element in python_object])
                continue
            elif error_string.startswith('Expected') or error_string == \
                    'Unsupported numpy type datetime64':
                # The latter error occurs when there's a mix of datetime64 and
                # timedelta64 (+ possibly other types?)
                error_message = (
                    f'{python_object_description} and elements with a mix of '
                    f'types, and cannot be represented as any single R vector '
                    f'type')
                raise TypeError(error_message) from e
            elif error_string.startswith('Unsupported numpy type'):
                if error_string.endswith('13'):
                    is_float128 = True
                elif error_string.endswith('14') or \
                        error_string.endswith('15'):
                    # complex64 (14) or complex128 (15)
                    is_complex = True
                elif any(isinstance(element, np.timedelta64) for element in (
                        python_object.values if is_pandas else python_object)):
                    # timedelta64s can cause lots of different "Unsupported
                    # numpy type" errors depending on which other types they
                    # co-occur with
                    error_message = (
                        f'{python_object_description} and elements with a mix '
                        f'of types, and cannot be represented as any single R '
                        f'vector type')
                    raise TypeError(error_message) from e
                else:
                    raise
            else:
                raise
        except pa.ArrowTypeError as e:
            error_string = str(e)
            if error_string == (
                    "object of type <class 'pandas._libs.tslibs.period."
                    "Period'> cannot be converted to int"):
                is_period = True
            elif error_string.startswith('Expected') or \
                    'cannot be converted to' in error_string:
                error_message = (
                    f'{python_object_description} and elements with a mix of '
                    f'types, and cannot be represented as any single R vector '
                    f'type')
                raise TypeError(error_message) from e
            else:
                raise
        except TypeError as e:
            error_string = str(e)
            if error_string == (
                    "int() argument must be a string, a bytes-like object or "
                    "a real number, not 'datetime.date'"):
                # datetime64[D], datetime64[M] or datetime64[Y]
                is_datetime = True
            elif error_string == (
                    "int() argument must be a string, a bytes-like object or "
                    "a real number, not 'datetime.datetime'"):
                # Mix of datetime.time and np.datetime64
                error_message = (
                    f'{python_object_description} and elements with a mix of '
                    f'types, and cannot be represented as any single R vector '
                    f'type')
                raise TypeError(error_message) from e
            elif error_string == (
                    "int() argument must be a string, a bytes-like object or "
                    "a real number, not 'datetime.timedelta'"):
                error_message = (
                    f'{python_object_description} and elements with a mix of '
                    f'types, and cannot be represented as any single R vector '
                    f'type')
                raise TypeError(error_message) from e
            else:
                raise
        except OverflowError as e:
            error_string = str(e)
            if error_string == 'cannot convert float infinity to integer':
                error_message = (
                    f'{python_object_description} and elements with a mix of '
                    f'types, and cannot be represented as any single R vector '
                    f'type')
                raise TypeError(error_message) from e
            elif platform.system() == 'Windows' and error_string == \
                        'Python int too large to convert to C long':
                # Occurs when mixing a datetime.date and an int > 2_147_483_647
                # on Windows, and possibly in other situations too
                error_message = (
                    f'{python_object_description} and elements with a mix of '
                    f'types, and cannot be represented as any single R vector '
                    f'type')
                raise TypeError(error_message) from e
            else:
                raise
        except np.ComplexWarning as e:
            if str(e) == 'Casting complex values to real discards the ' \
                         'imaginary part':
                error_message = (
                    f'{python_object_description} and elements with a mix of '
                    f'types, and cannot be represented as any single R vector '
                    f'type')
                raise TypeError(error_message) from e
            else:
                raise
        is_float16 = arrow is not None and pa.types.is_float16(arrow.type)
        if is_float16 or is_float128:
            # No need to call _check_object_elements(): Arrow seems to only
            # make the result HalfFloat if all elements are float16, and the
            # float128 error seems to only trigger if all elements are float128
            python_object = python_object.astype(float)
            continue
        if is_complex:
            # Complex numbers aren't supported by Arrow, so cast to complex
            # (after checking that all entries are numeric) and return as-is,
            # without converting to Arrow. However, to ensure missing entries
            # are converted to NA rather than NaN+NaNi, set both their real and
            # imaginary parts to the value 0xa20700000000f87f, which represents
            # NA in R complex vectors. (It is one of many floating-point
            # numbers with the value nan.)
            import struct
            allowed_dtypes = {np.complex64, np.complex128, np.int8, np.int16,
                              np.int32, np.int64, np.uint8, np.uint16,
                              np.uint32, np.uint64, np.float16, np.float32,
                              np.float64}
            if platform.system() != 'Windows':
                allowed_dtypes.add(np.float128)
            _check_object_elements(
                python_object, python_object_description, is_pandas,
                allowed_types=(int, float, complex), disallowed_types=bool,
                allowed_dtypes=allowed_dtypes)
            python_object = python_object.astype(complex)
            python_object[np.isnan(python_object)] = complex(
                *struct.unpack('<dd', b'\xa2\x07\x00\x00\x00\x00\xf8\x7f' * 2))
            return python_object
        if is_period:
            # pd.Period; convert to pd.Timestamp
            values = python_object.values if is_pandas else python_object
            for i, element in enumerate(values):
                if isinstance(element, pd.Period):
                    values[i] = element.to_timestamp()
            continue
        if arrow is not None and python_object.dtype == object:
            # The dtype == object check avoids an infinite loop by making sure
            # the dtype has not been converted from object yet
            if pa.types.is_int64(arrow.type):
                # Work around github.com/apache/arrow/issues/40906
                _check_object_elements(
                    python_object, python_object_description, is_pandas,
                    allowed_types=(int, float, complex), disallowed_types=bool,
                    allowed_dtypes={np.int8, np.int16, np.int32, np.int64,
                                    np.uint8, np.uint16, np.uint32, np.uint64,
                # Work around github.com/apache/arrow/issues/40910
                                    np.float16, np.float128})
                values = python_object.values if is_pandas else python_object
                float16_or_float128 = np.float16, np.float128
                if any(isinstance(element, float16_or_float128)
                       for element in values):
                    python_object = python_object.astype(float)
                    continue
            elif pa.types.is_float64(arrow.type):
                # Work around github.com/apache/arrow/issues/40909, part 1
                allowed_dtypes = {np.int8, np.int16, np.int32, np.int64,
                                  np.uint8, np.uint16, np.uint32, np.uint64,
                                  np.float16, np.float32, np.float64}
                if platform.system() != 'Windows':
                    allowed_dtypes.add(np.float128)
                _check_object_elements(
                    python_object, python_object_description, is_pandas,
                    allowed_types=(int, float), disallowed_types=bool,
                    allowed_dtypes=allowed_dtypes)
            elif pa.types.is_time64(arrow.type):
                # Work around github.com/apache/arrow/issues/40909, part 2,
                # and also give an error if any element has a time zone
                values = python_object.values if is_pandas else python_object
                for element in values:
                    if element is None:
                        continue
                    elif not isinstance(element, datetime.time):
                        error_message = (
                            f'{python_object_description} and elements with a '
                            f'mix of types, and cannot be represented as any '
                            f'single R vector type')
                        raise TypeError(error_message)
                    elif element.tzinfo is not None:
                        error_message = (
                            f'{python_object_description} and contains a '
                            f'datetime.time object with a non-missing time '
                            f'zone, which cannot be represented in R')
                        raise TypeError(error_message)
            elif pa.types.is_duration(arrow.type):
                # Work around github.com/apache/arrow/issues/40909, part 3
                _check_object_elements(
                    python_object, python_object_description, is_pandas,
                    allowed_types=(datetime.timedelta, pd.Timedelta)
                                  if has_pandas else (datetime.timedelta,),
                    disallowed_types=(), allowed_dtypes={np.timedelta64})
                # Work around github.com/apache/arrow/issues/40620
                python_object = python_object.astype('timedelta64[ns]')
                continue
            elif pa.types.is_timestamp(arrow.type):
                # Work around github.com/apache/arrow/issues/40909, part 4
                is_datetime = True
            elif pa.types.is_date32(arrow.type):
                # Work around github.com/apache/arrow/issues/40909, part 5
                # Note: datetime.datetime is a subclass of datetime.date
                _check_object_elements(
                    python_object, python_object_description, is_pandas,
                    allowed_types=(datetime.date,),
                    disallowed_types=(datetime.datetime,),
                    allowed_dtypes=set())
        if is_datetime:
            # Normalize np.datetime64, datetime.datetime and pd.Timestamp to
            # datetime.datetime; give errors on other types or mixed time zones
            time_zone = ...
            values = python_object.values if is_pandas else python_object
            for i, element in enumerate(values):
                if element is None:
                    continue
                elif isinstance(element, datetime.datetime):
                    new_time_zone = element.tzinfo
                elif isinstance(element, np.datetime64):
                    values[i] = element.astype('datetime64[us]').item()
                    new_time_zone = None
                elif has_pandas and isinstance(element, pd.Timestamp):
                    values[i] = element.to_pydatetime()
                    new_time_zone = element.tz
                else:
                    error_message = (
                        f'{python_object_description} and elements with a mix '
                        f'of types, and cannot be represented as any single R '
                        f'vector type')
                    raise TypeError(error_message)
                if time_zone != new_time_zone and time_zone is not ...:
                    error_message = (
                        f'{python_object_description} and elements with a mix '
                        f'of time zones; convert elements to a common time '
                        f'zone before calling to_r()')
                    raise TypeError(error_message)
                time_zone = new_time_zone
            if arrow is None:
                continue
        return arrow


def _check_to_py_format(format: str | dict[str, str],
                        variable_name='format') -> None:
    """
    Checks if format is a valid format argument for to_py().
    
    Args:
        format: the format argument
        variable_name: the name of the format variable, used in error messages
    """
    and_or = lambda valid: \
        f'{", ".join(f"{key!r}" for key in valid[:-1])}, and/or {valid[-1]!r}'
    valid_keys = 'vector', 'matrix', 'data.frame'
    valid_values = 'polars', 'pandas', 'pandas-pyarrow', 'numpy'
    if isinstance(format, dict):
        if not format:
            error_message = f'{variable_name} is an empty dictionary'
            raise ValueError(error_message)
        has_any_non_None_values = False
        for key, value in format.items():
            if not isinstance(key, str):
                error_message = (
                    f'{variable_name}.keys() contains a key of type '
                    f'{type(key).__name__!r}, but must contain only the '
                    f'strings {and_or(valid_keys)}')
                raise TypeError(error_message)
            if key not in valid_keys:
                error_message = (
                    f'{variable_name}.keys() contains the key {key!r}, but '
                    f'must contain only the strings {and_or(valid_keys)}')
                raise ValueError(error_message)
            if value is not None:
                has_any_non_None_values = True
                if not isinstance(value, str):
                    error_message = (
                        f'{variable_name}.values() contains a value of type '
                        f'{type(value).__name__!r}, but must contain only the '
                        f'strings {and_or(valid_values)}, or None')
                    raise TypeError(error_message)
                if value not in valid_values:
                    error_message = (
                        f'{variable_name}.values() contains the value '
                        f'{value!r}, but must contain only the strings '
                        f'{and_or(valid_values)}, or None')
                    raise ValueError(error_message)
        if not has_any_non_None_values:
            error_message = (
                f'{variable_name} is a non-empty dictionary, but all its '
                f'values are None')
            raise ValueError(error_message)
    else:
        if not isinstance(format, str):
            error_message = (
                f'{variable_name} must be str or dict, but has type '
                f'{type(format).__name__!r}')
            raise TypeError(error_message)
        if format not in valid_values:
            error_message = (
                f'invalid {variable_name} {format!r}; must be '
                f'{and_or(valid_values)}, or a dict with these as values and '
                f'{and_or(valid_keys)} as keys')
            raise ValueError(error_message)


def _plot_window_open() -> bool:
    """
    Checks whether a plot window is open, by checking whether R's .Devices
    variable contains a graphics device with a name that's not the empty string
    (which can happen if a plot window was closed) or 'null device' and does
    not have the 'filepath' attribute (which non-interactive graphics devices
    like png() and pdf() do).

    Returns:
        Whether the plot window is open.
    """
    devices = _rlib.Rf_findVar(_rlib.Rf_install(b'.Devices'), _rlib.R_BaseEnv)
    while True:
        device_name = \
            _ffi.string(_rlib.R_CHAR(_rlib.STRING_ELT(_rlib.CAR(devices), 0)))
        if device_name != b'' and device_name != b'null device' and \
                _rlib.Rf_getAttrib(_rlib.CAR(devices),
                                   _rlib.Rf_install(b'filepath')) == \
                _rlib.R_NilValue:
            return True
        devices = _rlib.CDR(devices)
        if devices == _rlib.R_NilValue:
            return False


def _check_activity(select: _cffi_backend._CDataBase,
                    timeout=0.1) -> _cffi_backend._CDataBase:
    """
    A reimplementation of R's R_checkActivity() that avoids the longjmp in
    R_SelectEx(), which causes it to segfault when run on a background thread.
    Checks for activity on each of the file descriptors in R_InputHandlers,
    such as a plot being resized or closed. Not used on Windows.
    
    Args:
        select: a pointer to the select function in the C standard library
        timeout: the timeout in seconds, passed to the select() system call

    Returns:
        A mask of file descriptors that are waiting to be handled, or NULL if
        none are waiting.
    """
    NFDBITS = 8 * _ffi.sizeof('unsigned long')
    read_mask = _ffi.new('fd_set *')  # initialized to 0, so skip FD_ZERO()
    timeval = _ffi.new('struct timeval *')
    timeval.tv_sec = int(timeout)
    timeval.tv_usec = int((timeout - timeval.tv_sec) * 1_000_000)
    max_fd = -1
    handlers = _rlib.R_InputHandlers
    while handlers != _ffi.NULL:
        fd = handlers.fileDescriptor
        if fd > 0:  # i.e. not stdin
            # The next line is equivalent to FD_SET(fd, read_mask)
            read_mask.fds_bits[fd // NFDBITS] |= 1 << (fd % NFDBITS)
            max_fd = max(max_fd, fd)
        handlers = handlers.next
    if select(max_fd + 1, read_mask, _ffi.NULL, _ffi.NULL, timeval) > 0:
        return read_mask
    else:
        return _ffi.NULL


def _handle_plot_events() -> None:
    """
    Handle plot events. Starts running in a background thread when the user
    opens an interactive plot, or on the main thread on Windows and Mac. On
    non-Windows systems, repeatedly polls for events from open file descriptors
    with _check_activity(), then handles them by calling R_runHandlers(). On
    Windows, repeatedly calls GA_doevent() to detect and handle plotting
    events. Returns once the plot window is no longer open.
    """
    if platform.system() != 'Windows':
        select = _ffi.dlopen(None).select
        while True:
            fd_set = _check_activity(select)
            if fd_set != _ffi.NULL:
                _rlib.R_runHandlers(_rlib.R_InputHandlers.next, fd_set)
            if not _plot_window_open():
                return
    else:
        global _graphapp
        while _graphapp.GA_doevent():
            pass


def _require(R_package_name: bytes, rmemory: _RMemory) -> bool:
    """
    Equivalent to
    `to_py(f'suppressPackageStartupMessages(require({R_package_name}))')`,
    but using the R C API.
    
    Args:
        R_package_name: the name of the package to require
        rmemory: an instance of the _RMemory class
    
    Returns:
        Whether the package was successfully loaded.
    """
    
    function_call = rmemory.protect(
        _rlib.Rf_lang2(_rlib.Rf_install(b'suppressPackageStartupMessages'),
                       _rlib.Rf_lang2(_rlib.Rf_install(b'require'),
                                      _bytestring_to_character_vector(
                                          R_package_name, rmemory))))
    return _rlib.LOGICAL_ELT(
        _call(function_call, rmemory, 'unable to run require'), 0)


def r(R_code: str = ...) -> None:
    """
    Run a string of R code. Or, call r() with no arguments to open an R
    terminal for interactive debugging. Use Ctrl + D to drop back into Python.
    Any variables you modify in the R terminal will be available from Python,
    and vice versa.
    
    Args:
        R_code: an optional string of R code to run. Pass nothing to open an R
                terminal. The absence of a value is indicated with the special
                "sentinel" value ... (Ellipsis) rather than None, to stop users
                from inadvertently opening the terminal when passing a variable
                that is supposed to be a string but is unexpectedly None.
    """
    # Initialize ryp, if not already initialized
    _initialize_ryp()
    if R_code is ...:
        # Only allow on the main thread
        if not _main_thread:
            error_message = (
                'opening an R terminal with r() is only allowed on the main '
                'thread')
            raise RuntimeError(error_message)
        _rlib.R_ReplDLLinit()
        # Override q() and quit() so the user doesn't accidentally close
        # Python when trying to exit the R terminal
        r(f'q = quit = function(...) {{'
          f'cat("Press {_EOF_instructions} to return to Python\n")}}')
        try:
            # Run R_ReplDLLdo1() in a loop until the user presses EOF
            # (Ctrl + D on non-Windows systems, Ctrl + Z followed by Enter
            # on Windows). Ignore KeyboardInterrupts in Python during this
            # loop.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            while _rlib.R_ReplDLLdo1() != -1:
                pass
            print()
            signal.signal(signal.SIGINT, signal.default_int_handler)
        finally:
            r(f'q = quit = function(...) {{ cat("Press '
              f'{_EOF_instructions} to exit the Python terminal, or run '
              f'exit()\n") }}')  # reset
        return
    elif isinstance(R_code, str):
        rmemory = _RMemory(_rlib)
        nested = False
        # Disallow empty code
        if not R_code:
            error_message = 'R_code is an empty string'
            raise ValueError(error_message)
    elif isinstance(R_code, tuple) and len(R_code) == 2 and \
            isinstance(R_code[0], str) and isinstance(R_code[1], _RMemory):
        R_code, rmemory = R_code
        nested = True
    else:
        error_message = \
            f'R_code has type {type(R_code).__name__!r}, but must be a string'
        raise TypeError(error_message)
    # Check that R_code does not contain null bytes, to avoid Rf_mkCharLenCE()
    # raising an error when converting it to a character vector later
    if '\0' in R_code:
        error_message = r'R_code contains a null byte (\0)'
        raise SyntaxError(error_message)
    # Wrap the code a withAutoprint({}) block so things like r('2 + 2') are
    # printed to the terminal, like in R's interactive mode. (Skip this in
    # nested mode, i.e. when r() is being called to evaluate an R statement
    # from to_py().)
    #
    # Ideally we'd use R's global variable R_Visible to determine whether a
    # line should be printed, like R's own REPL does in Rf_ReplIteration, but
    # R_Visible is no longer exported. The limitations of R's API in this
    # regard are discussed at:
    # - stat.ethz.ch/pipermail/r-devel/2007-January/044414.html
    # - stat.ethz.ch/pipermail/r-devel/2007-January/044336.html
    wrapped_R_code = R_code if nested else \
        f'withAutoprint({{{R_code}}}, echo=FALSE)'
    try:
        # Parse the code with R_ParseVector()
        # - Wrap in R_tryCatchError() to avoid crashing Python in certain edge
        #   cases where parsing actually raises an R error instead of setting
        #   status to PARSE_INCOMPLETE, PARSE_ERROR, or PARSE_EOF - for
        #   instance, when R_code contains an invalid escape sequence like \S.
        # - Because the withAutoprint() may affect the parse error, reparse the
        #   code without it, just to get the correct error message. (This isn't
        #   necessary in nested mode, since we didn't wrap in withAutoprint()).
        #   The reparse shouldn't raise an error if the original parse didn't,
        #   so there's no need to wrap the reparse in R_tryCatchError().
        status = _ffi.new('int[1]')
        e = None
        
        def onerror(exception, exc_value, traceback):
            # By default, tryCatchError() ignores exceptions during the
            # ffi.callback; instead, store them in a nonlocal and raise after
            nonlocal e
            e = exception
        
        parsed_expression = rmemory.protect(
            _rlib.R_tryCatchError(
                _ffi.callback('SEXP (void *data)', onerror=onerror)(
                    lambda data: _rlib.R_ParseVector(*_ffi.from_handle(data))),
                _ffi.new_handle((
                    _string_to_character_vector(wrapped_R_code, rmemory),
                    -1, status, _rlib.R_NilValue)),
                _ffi.callback('SEXP (SEXP cond, void *hdata)')(
                    lambda cond, hdata: _rlib.R_NilValue),
                _ffi.NULL))
        
        if e is not None:
            raise e
        if status[0] == 0:
            # Not sure how to get the error message from R in this case
            error_message = 'R_code is a malformed string'
            raise SyntaxError(error_message)
        elif status[0] != _rlib.PARSE_OK:
            if not nested:
                _rlib.R_ParseVector(
                    _string_to_character_vector(R_code, rmemory),
                    -1, status, _rlib.R_NilValue)  # reparse without autoprint
            parse_error_message = \
                _ffi.string(_rlib.R_ParseErrorMsg).decode('utf-8')
            # The code snipped that triggered the syntax error is stored in a
            # circular buffer that stretches back from `R_ParseContextLast` to
            # the first null byte, wrapping around to the end of the buffer if
            # the beginning of the buffer is reached before finding a null byte
            parse_context = bytes(_ffi.buffer(_rlib.R_ParseContext))
            parse_context_last = _rlib.R_ParseContextLast
            first_null_byte_index = parse_context.find(b'\0')
            if first_null_byte_index < parse_context_last:
                parse_context = parse_context[first_null_byte_index + 1:
                                              parse_context_last + 1]
            else:
                # wrap around
                last_null_byte_index = parse_context.rfind(b'\0')
                parse_context = parse_context[last_null_byte_index + 1:] + \
                                parse_context[:parse_context_last + 1]
            parse_context = parse_context.decode('utf-8')
            if '\n' in parse_context:
                parse_context = parse_context.lstrip('\n')
                error_message = f'{parse_error_message} in:\n"{parse_context}"'
            else:
                error_message = f'{parse_error_message} in "{parse_context}"'
            raise SyntaxError(error_message)
        # Evaluate the parsed expression; it's always a single expression, due
        # to the withAutoprint(). (Except in nested mode, in which case raise
        # an error if it's multiple expressions.) Use R_tryEvalSilent() rather
        # than Rf_eval() so we can check if it ran OK via the status argument.
        error_buffer = _rlib.R_curErrorBuf()
        error_buffer[0] = b'\0'  # reset error buffer
        if nested:
            num_statements = _rlib.Rf_xlength(parsed_expression)
            if num_statements > 1:
                error_message = (
                    f'R_statement contains {num_statements:,} R statements, '
                    f'but must contain a single statement')
                raise ValueError(error_message)
        expression = _rlib.VECTOR_ELT(parsed_expression, 0)
        result = rmemory.protect(
            _rlib.R_tryEvalSilent(expression, _rlib.R_GlobalEnv, status))
        # Raise RuntimeError on errors and KeyboardInterrupt on Ctrl + C.
        # Since there's no easy way to distinguish Ctrl + C from an error,
        # assume any blank error message was a Ctrl + C; this is a hack.
        # Also, remove the 'Error in eval(ei, envir) : ' prefix withAutoprint()
        # induces when the error occurs at the top level, the 'Error: '
        # prefix that evaluating a single statement from to_py() induces, and
        # the 'Error ' prefix that occurs when the error occurs in a function.
        # (The 'Error: ' prefix also happens at the top level sometimes.)
        if status[0] != 0:
            error_message = _ffi.string(error_buffer)
            if error_message:
                error_message = error_message.decode('utf-8').rstrip()\
                    .removeprefix('Error in eval(ei, envir) : ')\
                    .removeprefix('Error: ')\
                    .removeprefix('Error ')
                if error_message.startswith('object') and \
                        error_message.endswith('not found'):
                    raise NameError(error_message)
                else:
                    raise RuntimeError(error_message)
            else:
                raise KeyboardInterrupt
        else:
            error_message = False
        # If there was no error, call stop() inside R_tryEvalSilent() to
        # silently raise an error, just so that warnings are printed.
        # (R_Warnings and PrintWarnings() are not exposed, so this is a hacky
        # workaround.) Note that this resets error_buffer, which is why we get
        # the error message from evaluating the parsed expression first.
        if not error_message:
            function_call = \
                rmemory.protect(_rlib.Rf_lang1(_rlib.Rf_install(b'stop')))
            _rlib.R_tryEvalSilent(function_call, _rlib.R_BaseEnv,
                                  _ffi.new('int[1]'))
        # If the user just created an interactive plot, start a background
        # thread to handle plot events, or run on the main thread on Windows
        # and Mac
        global _jupyter_notebook
        if not _jupyter_notebook:
            if platform.system() == 'Windows':
                global _graphapp
                if _graphapp is None:
                    R_home, rlib_path = _get_R_home_and_rlib_path()
                    _graphapp = _ffi.dlopen(
                        rlib_path.replace('R.dll', 'Rgraphapp.dll'))
                    _graphapp.GA_initapp(0, _ffi.NULL)
                if _plot_window_open():
                    _handle_plot_events()
            elif platform.system() == 'Darwin':
                if _plot_window_open():
                    _handle_plot_events()
            else:
                global _plot_event_thread
                if (_plot_event_thread is None or not
                        _plot_event_thread.is_alive()) and _plot_window_open():
                    _plot_event_thread = threading.Thread(
                        target=_handle_plot_events, daemon=True)
                    _plot_event_thread.start()
        # Return the result to to_py(), if in nested mode
        if nested:
            return result
    finally:
        # Unprotect everything, unless we are in nested mode, in which case
        # defer the unprotection until to_py() returns
        if not nested:
            rmemory.__exit__(None, None, None)


def to_r(python_object: Any, R_variable_name: str, *,
         format: Literal['keep', 'matrix', 'data.frame'] | None = None,
         rownames: Any = None, colnames: Any = None) -> None:
    """
    Convert a Python object to R. If the object is a list or tuple, recursively
    convert each element and return an unnamed list; if the object is a dict,
    do the same but return a named list.
    
    Args:
        python_object: the Python object to convert to R
        R_variable_name: the name of the R variable that will be created in R's
                         global namespace (globalenv) to store the converted
                         object
        format: the R format to convert 2D NumPy arrays, pandas and polars
                DataFrames, and pandas MultiIndexes to:
                - None: default to the value in options()['to_r_format']
                - 'keep': keep the same format when converting: convert 2D
                  NumPy arrays to matrices, and DataFrames/MultiIndexes to
                  data.frames (the default in options()['to_r_format'])
                - 'matrix': convert all of these to matrices
                - 'data.frame': convert all of these to data.frames
                Must be None unless python_object is a 2D NumPy array,
                DataFrame or MultiIndex, or a type that might contain one
                (list, tuple, or dict).
        rownames: a container (list, tuple, Series, etc.) of strings that will
                  be converted to R and used as the rownames for converted
                  NumPy arrays, polars Series and DataFrames, and scipy sparse
                  arrays and matrices; or None to not include rownames for
                  these Python types. If python_object is a list, tuple, or
                  dict, recursively apply rownames to each element. Must be
                  None if python_object is bytes/bytearray (since raw vectors
                  lack rownames) or a pandas Series or DataFrame (since they
                  already have rownames, i.e. an index).
        colnames: a container (list, tuple, Series, etc.) of strings that will
                  be converted to R and used as the colnames for converted
                  multidimensional NumPy arrays and scipy sparse arrays and
                  matrices; or None to not include colnames for them. Must be
                  None unless python_object is a multidimensional NumPy array,
                  or a type that might contain one (list, tuple, or dict).
    """
    # Initialize ryp, if not already initialized
    _initialize_ryp()
    # Raise errors when format is not None and we're not recursing
    raise_format_errors = format is not None
    # When calling to_r recursively, allow R_variable_name to be a tuple of
    # (str, _RMemory), where str is the name of python_object (e.g.
    # "python_object['a']" if python_object is a dict, when converting the
    # value corresponding to the key 'a'). This serves three purposes: it
    # indicates it's a recursive call, it allows descriptive variable names in
    # error messages, and it stops protected objects from being
    # garbage-collected until the top-level to_r() call returns. When calling
    # to_r() recursively, return the converted R object rather than assigning
    # it to the R global namespace. (Among other advantages, this simplifies
    # stack traces relative to calling a second function when recursing.)
    if isinstance(R_variable_name, str):
        # Thread-safety check
        if threading.current_thread() is not _ryp_thread:
            error_message = (
                "the R interpreter is not thread-safe, so ryp can't be used "
                "by multiple Python threads simultaneously; see the README "
                "for alternative parallelization strategies")
            raise RuntimeError(error_message)
        # Check that R_variable_name is valid
        _check_R_variable_name(R_variable_name)
        # Check that format is valid
        if format not in ('matrix', 'data.frame', 'keep', None):
            if isinstance(format, str):
                error_message = \
                    "format must be 'matrix', 'data.frame', 'keep', or None"
                raise ValueError(error_message)
            else:
                error_message = (
                    f"format has type {type(format).__name__!r}, but must be "
                    f"'matrix', 'data.frame', 'keep', or None")
                raise TypeError(error_message)
        python_object_name = 'python_object'
        rmemory = _RMemory(_rlib)
        top_level = True
    elif isinstance(R_variable_name, tuple) and len(R_variable_name) == 2 and \
            isinstance(R_variable_name[0], str) and \
            isinstance(R_variable_name[1], _RMemory):
        python_object_name, rmemory = R_variable_name
        top_level = False
    else:
        error_message = (
            f'R_variable_name has type {type(R_variable_name).__name__!r}, '
            f'but must be a string')
        raise TypeError(error_message)
    # Get preliminary information about what type python_object is
    is_ndarray = isinstance(python_object, np.ndarray)
    is_matrix = is_ndarray and python_object.ndim == 2
    is_multidimensional_ndarray = is_ndarray and python_object.ndim >= 2
    is_numpy_generic = isinstance(python_object, np.generic)
    is_numpy = is_ndarray or is_numpy_generic
    if 'scipy' in sys.modules:
        from scipy.sparse import csr_array, csc_array, coo_array, \
            csr_matrix, csc_matrix, coo_matrix
        if isinstance(python_object, csr_array):
            is_sparse = True
            is_csr = True
            is_csc = False
            is_coo = False
            sparse_supertype = 'array'
        elif isinstance(python_object, csr_matrix):
            is_sparse = True
            is_csr = True
            is_csc = False
            is_coo = False
            sparse_supertype = 'matrix'
        elif isinstance(python_object, csc_array):
            is_sparse = True
            is_csr = False
            is_csc = True
            is_coo = False
            sparse_supertype = 'array'
        elif isinstance(python_object, csc_matrix):
            is_sparse = True
            is_csr = False
            is_csc = True
            is_coo = False
            sparse_supertype = 'matrix'
        elif isinstance(python_object, coo_array):
            is_sparse = True
            is_csr = False
            is_csc = False
            is_coo = True
            sparse_supertype = 'array'
        elif isinstance(python_object, coo_matrix):
            is_sparse = True
            is_csr = False
            is_csc = False
            is_coo = True
            sparse_supertype = 'matrix'
        else:
            is_sparse = False
    else:
        is_sparse = False
    if 'pandas' in sys.modules:
        import pandas as pd
        is_pandas_df = isinstance(python_object, pd.DataFrame)
        is_pandas_series = isinstance(python_object, pd.Series)
        is_index = isinstance(python_object, pd.Index)
        is_multiindex = isinstance(python_object, pd.MultiIndex)
        is_pandas_timestamp_or_timedelta = \
            isinstance(python_object, pd.Timestamp) or \
            isinstance(python_object, pd.Timedelta)
        is_pandas_period = isinstance(python_object, pd.Period)
        is_pandas = is_pandas_df or is_pandas_series or is_index or \
            is_pandas_timestamp_or_timedelta or is_pandas_period or \
            python_object is pd.NA or python_object is pd.NaT
    else:
        is_pandas_df = False
        is_pandas_series = False
        is_index = False
        is_multiindex = False
        is_pandas = False
    if 'polars' in sys.modules:
        import polars as pl
        is_polars_df = isinstance(python_object, pl.DataFrame)
        is_polars_series = isinstance(python_object, pl.Series)
        is_polars = is_polars_df or is_polars_series
    else:
        is_polars_df = False
        is_polars_series = False
        is_polars = False
    is_df = is_pandas_df or is_polars_df
    is_series = is_pandas_series or is_polars_series
    if (is_df or is_series or is_index) and \
            max(python_object.shape) > 2_147_483_647:
        dimension_name = 'elements' if len(python_object.shape) == 1 else \
            'rows' if python_object.shape[0] > python_object.shape[1] else \
            'columns'
        error_message = (
            f'{python_object_name} is a {"pandas" if is_pandas else "polars"} '
            f'{type(python_object).__name__!r} with '
            f'{max(python_object.shape):,} {dimension_name}, more than '
            f'INT32_MAX (2,147,483,647), the maximum supported in R')
        raise ValueError(error_message)
    if is_numpy or is_series or is_index:
        dtype = python_object.dtype
    shape = None
    converted_index_separately = False
    add_integer64 = False
    # Defer loading Arrow until calling to_py() or to_r() for speed
    if 'pyarrow' in sys.modules:
        import pyarrow as pa
        is_pyarrow_array = isinstance(python_object, pa.Array)
    else:
        with ignore_sigint():
            import pyarrow as pa
        is_pyarrow_array = False
    from pyarrow.cffi import ffi as pyarrow_ffi
    if top_level:
        # Require arrow
        if not _require(b'arrow', rmemory):
            error_message = 'please install the arrow R package to use ryp'
            raise ImportError(error_message)
    # If format is not None, and we are not recursing, raise an error if
    # python_object is anything but a DataFrame, MultiIndex, matrix (2D NumPy
    # array) or something that might contain one (list, tuple, or dict)
    if raise_format_errors and top_level and \
            not (is_matrix or is_df or is_multiindex) and \
            not isinstance(python_object, (list, tuple, dict)):
        if is_ndarray:
            error_message = (
                f'python_object is a {python_object.ndim:,}D NumPy array, not '
                f'a 2D one, so format must be None')
            raise TypeError(error_message)
        else:
            error_message = (
                f'python_object is not a 2D NumPy array, DataFrame, or '
                f'MultiIndex, so format must be None. It has type '
                f'{type(python_object).__name__!r}.\n(format can also be '
                f'specified when python_object is a list, tuple, or dict, '
                f'since these may contain 2D NumPy arrays, DataFrames, or '
                f'MultiIndexes.)')
            raise TypeError(error_message)
    # If format is None, default to what's in the config (or False for NumPy)
    if format is None:
        format = _config['to_r_format']
    # If format='keep', set format to 'data.frame' if python_object is a
    # DataFrame or MultiIndex, or 'matrix' if a 2D NumPy array
    if format == 'keep':
        if is_df:
            format = 'data.frame'
        if is_matrix:
            format = 'matrix'
    # Allow adding rownames of any length to 0 x 0 polars DataFrames
    if is_df and is_polars and rownames is not None and \
            len(python_object) == 0:
        if not hasattr(rownames, '__len__'):
            error_message = \
                f'rownames has unsupported type {type(rownames).__name__!r}'
            raise TypeError(error_message)
        python_object = np.empty((len(rownames), 0))
        is_df = is_polars = False
        is_numpy = is_ndarray = is_matrix = is_multidimensional_ndarray = True
        dtype = python_object.dtype
    try:
        # If rownames is not None, and we are not recursing, give an error if
        # python_object is bytes/bytearray (since raw vectors don't have
        # rownames) or a pandas Series or DataFrame (since they already have
        # rownames, i.e. an index). Otherwise, convert the rownames to R. When
        # recursing, assume they're already converted to R, and don't give an
        # error since it's ok if some objects we're recursing over support
        # rownames and some don't.
        if rownames is not None and top_level:
            if isinstance(python_object, (bytes, bytearray)):
                error_message = (
                    f'python_object has type '
                    f'{type(python_object).__name__!r}, but raw vectors lack '
                    f'rownames, so rownames must be None')
                raise TypeError(error_message)
            elif is_pandas and (is_df or is_series):
                error_message = (
                    f'python_object is a pandas '
                    f'{"DataFrame" if is_df else "Series"}, so rownames must '
                    f'be None; specify rownames via python_object.index '
                    f'instead')
                raise TypeError(error_message)
            rownames = _convert_names(rownames, 'rownames', python_object,
                                      python_object_name, rmemory)
        # Same for colnames, but raise an error if python_object is anything
        # but a multidimensional NumPy array, scipy sparse array or matrix, or
        # something that might contain one (list, tuple, or dict)
        if colnames is not None and top_level:
            if not is_multidimensional_ndarray and not is_sparse and not \
                    isinstance(python_object, (list, tuple, dict)):
                error_message = (
                    f'python_object is not a multidimensional NumPy array or '
                    f'scipy sparse array or matrix, so colnames must be None. '
                    f'It has type {type(python_object).__name__!r}.\n'
                    f'(colnames can also be specified when python_object is a '
                    f'list, tuple, or dict, since these may contain '
                    f'multidimensional NumPy arrays or sparse '
                    f'arrays/matrices.)')
                raise TypeError(error_message)
            colnames = _convert_names(colnames, 'colnames', python_object,
                                      python_object_name, rmemory)
        # Process python_object depending on its type
        arrow = None
        if python_object is None:
            result = _rlib.R_NilValue
        elif isinstance(python_object, bool):
            result = rmemory.protect(_rlib.Rf_ScalarLogical(python_object))
        elif isinstance(python_object, int):
            if abs(python_object) > 2_147_483_647:
                # Box in a PyArrow Array so the integer gets converted to a
                # length-1 bit64::integer64 vector
                result = to_r(pa.array([python_object]), (
                    f'pyarrow.array([{python_object_name}])', rmemory))
            else:
                result = rmemory.protect(_rlib.Rf_ScalarInteger(python_object))
        elif isinstance(python_object, float):
            result = rmemory.protect(_rlib.Rf_ScalarReal(python_object))
        elif isinstance(python_object, str):
            # Check that python_object does not contain null bytes, to avoid
            # Rf_mkCharLenCE() raising an error when converting it to a
            # character vector later
            if '\0' in python_object:
                error_message = \
                    r'python_object is a string that contains a null byte (\0)'
                raise SyntaxError(error_message)
            result = _string_to_character_vector(python_object, rmemory)
        elif isinstance(python_object, (bytes, bytearray)):
            result = rmemory.protect(_rlib.Rf_allocVector(_rlib.RAWSXP,
                                                          len(python_object)))
            _ffi.buffer(_ffi.cast('char*', _rlib.RAW(result)),
                       len(python_object))[:] = python_object
        elif isinstance(python_object, (list, tuple)):
            # Convert to unnamed list
            result = rmemory.protect(_rlib.Rf_allocVector(_rlib.VECSXP,
                                                          len(python_object)))
            try:
                for i, value in enumerate(python_object):
                    _rlib.SET_VECTOR_ELT(result, i, to_r(
                        value, (f'{python_object_name}[{i}]', rmemory),
                        format=format, rownames=rownames, colnames=colnames))
            except RecursionError as e:
                error_message = (
                    f'maximum recursion depth exceeded when converting '
                    f'{python_object_name} to R; this usually means the '
                    f'object contains a circular reference')
                raise ValueError(error_message) from e
        elif isinstance(python_object, dict):
            # Convert to named list
            for key in python_object:
                if not isinstance(key, str):
                    error_message = (
                        f'{python_object_name} is a dict that contains the '
                        f'key {key!r}, which has type {type(key).__name__!r} '
                        f'rather than str and thus cannot be converted into '
                        f'the name of an R list element')
                    raise TypeError(error_message)
            result = rmemory.protect(_rlib.Rf_allocVector(_rlib.VECSXP,
                                                          len(python_object)))
            try:
                for i, (key, value) in enumerate(python_object.items()):
                    _rlib.SET_VECTOR_ELT(result, i, to_r(
                        value, (f'{python_object_name}[{key!r}]', rmemory),
                        format=format, rownames=rownames, colnames=colnames))
            except RecursionError as e:
                error_message = (
                    f'maximum recursion depth exceeded when converting '
                    f'{python_object_name} to R; this usually means the '
                    f'object contains a circular reference')
                raise ValueError(error_message) from e
            _rlib.Rf_setAttrib(result, _rlib.R_NamesSymbol, to_r(
                tuple(python_object), (f'tuple({python_object_name})',
                                       rmemory)))
        elif is_polars:
            bad_dtypes = pl.Binary, pl.Decimal, pl.List, pl.Array, pl.Struct
            if is_df:
                # If format='matrix', all columns must have the same data type,
                # and the data type cannot be Date or Datetime; concatenate the
                # columns into a single Series
                unique_dtypes = set(python_object.dtypes)
                if format == 'matrix':
                    if len(unique_dtypes) > 1:
                        error_message = (
                            f"{python_object_name} is a polars DataFrame with "
                            f"a mix of data types (i.e. "
                            f"len(set({python_object_name}.dtypes)) > 1) so "
                            f"format='matrix' cannot be specified")
                        raise TypeError(error_message)
                    dtype = next(iter(unique_dtypes))
                    if dtype == pl.Date or dtype == pl.Datetime:
                        R_type = 'R Date' if dtype == pl.Date else 'POSIXct'
                        error_message = (
                            f"{python_object_name} is a polars DataFrame "
                            f"where all columns have data type {dtype!r}, so "
                            f"format='matrix' cannot be specified.\n({R_type} "
                            f"matrices are not supported, only vectors and "
                            f"data.frames.)")
                        raise TypeError(error_message)
                    result = to_r(pl.concat(python_object), (
                        f'pl.concat({python_object_name})', rmemory))
                    shape = python_object.shape
                    colnames = \
                        _convert_names(python_object.columns, 'colnames',
                                       python_object, python_object_name,
                                       rmemory)
                else:
                    # Disallow Binary, Decimal, List, Array and Struct; work
                    # around github.com/pola-rs/polars/issues/15286
                    bad_cols = [col for col, dtype in
                                python_object.schema.items()
                                if dtype.base_type() in bad_dtypes]
                    if len(bad_cols) > 0:
                        bad_col = bad_cols[0]
                        bad_dtype = python_object[bad_col].dtype
                        error_message = (
                            f'{python_object_name}[{bad_col!r}] is a polars '
                            f'Series with data type '
                            f'{bad_dtype.base_type()!r}, which lacks an R '
                            f'equivalent')
                        if bad_dtype == pl.Decimal:
                            error_message += (
                                '.\nConsider converting to Float64 before '
                                'calling to_r().')
                        raise TypeError(error_message)
                    # If any columns are pl.UInt32/pl.UInt64/pl.Int64 (which
                    # we handle zero-copy as Series) or pl.Object, fall back to
                    # converting to dict, converting each column to R
                    # separately, then converting to data.frame
                    if pl.UInt32 in unique_dtypes or \
                            pl.UInt64 in unique_dtypes or \
                            pl.Int64 in unique_dtypes or \
                            pl.Object in unique_dtypes:
                        result = to_r(
                            python_object.to_dict(),
                            # deliberately leave python_object_name as-is (if
                            # column 'a' has an error, the error message will
                            # correctly say python_object['a'])
                            (python_object_name, rmemory), format=format)
                        result = _as_data_frame(result, rmemory)
                    # Otherwise, convert to Arrow
                    elif len(python_object) > 0:
                        arrow = \
                            python_object.rechunk().to_arrow().to_batches()[0]
                    else:
                        # python_object.to_arrow().to_batches() == [], so do:
                        arrow = python_object.to_arrow()
                        arrow = pa.RecordBatch.from_arrays([
                            arr.combine_chunks() for arr in arrow],
                            schema=arrow.schema)
            elif is_series:
                # Disallow Binary, Decimal, List, Array and Struct
                if any(dtype == bad_dtype for bad_dtype in bad_dtypes):
                    error_message = (
                        f'{python_object_name} is a polars Series with data '
                        f'type {dtype.base_type()!r}, which lacks an R '
                        f'equivalent')
                    if dtype == pl.Decimal:
                        error_message += (
                            '.\nConsider converting to Float64 before calling '
                            'to_r().')
                    raise TypeError(error_message)
                if dtype == pl.Object:
                    arrow = _convert_object_to_arrow(
                        python_object,
                        f"{python_object_name} is a polars Series with "
                        f"data type 'object'", is_polars=True)
                    if not isinstance(arrow, pa.Array):
                        # complex; convert via NumPy, not Arrow
                        python_object = arrow
                        arrow = None
                        result = to_r(python_object,
                                      (python_object_name, rmemory))
                # View null-free, single-chunk UInt32 Series as int32 and
                # UInt64/Int64 as double, to allow zero-copy. For UInt64/Int64,
                # we will reinterpret the double array as bit64::integer64 on
                # the R side by manually adding 'integer64' as a class.
                elif dtype == pl.UInt32 and python_object.null_count() == 0 \
                        and python_object.n_chunks() == 1:
                    python_object = python_object\
                        .to_numpy(allow_copy=False)\
                        .view(np.int32)
                    arrow = pa.array(python_object)
                    is_ndarray = True
                    is_polars = False
                elif (dtype == pl.UInt64 or dtype == pl.Int64) and \
                        python_object.null_count() == 0 and \
                        python_object.n_chunks() == 1:
                    python_object = python_object\
                        .to_numpy(allow_copy=False)\
                        .view(np.float64)
                    arrow = pa.array(python_object)
                    is_ndarray = True
                    is_polars = False
                    add_integer64 = True
                else:
                    # Match the unsigned behavior in the zero-copy case by
                    # casting UInt32 to Int32 and UInt64 to Int64
                    if dtype == pl.UInt32:
                        python_object = python_object\
                            .cast(pl.Int32, wrap_numerical=True)
                    elif dtype == pl.UInt64:
                        python_object = python_object\
                            .cast(pl.Int64, wrap_numerical=True)
                    arrow = python_object.to_arrow()
            else:
                error_message = (
                    f'{python_object_name} has unsupported type '
                    f'{type(python_object).__name__!r}')
                raise TypeError(error_message)
        elif is_pandas:
            # Support DataFrame, Series, Index (including MultiIndex),
            # Timestamp, Timedelta, Period, NA, and NaT
            # Convert MultiIndex to DataFrame
            if is_multiindex:
                python_object = python_object.to_frame(index=False)
                python_object_name = \
                    f'{python_object_name}.to_frame(index=False)'
                is_df = True
            # Convert PeriodIndex to DatetimeIndex
            if isinstance(python_object, pd.PeriodIndex):
                python_object = python_object.to_timestamp()
            is_index = isinstance(python_object, pd.Index)
            # If python_object is a Series or DataFrame, handle its index
            if is_df or is_series:
                # Disallow MultiIndexes
                if isinstance(python_object.index, pd.MultiIndex):
                    error_message = (
                        f'{python_object_name}.index is a pandas MultiIndex, '
                        f'which lacks an R equivalent.\nUse reset_index() to '
                        f'retain the index as a column, or '
                        f'reset_index(drop=True) to discard the index.')
                    raise TypeError(error_message)
                # If python_object.index is not the default index (i.e.
                # range(len(python_object)), require it to be a string, or a
                # categorical with string categories, since R rownames must be
                # strings. Assume dtype = object is a string index for now; it
                # will fail to convert to R if not all elements are strings.
                # Convert length-0 object indices to StringDtype so that they
                # convert to character(0) rather than vctrs::unspecified(0).
                index = python_object.index
                if pd.api.types.is_string_dtype(index.dtype) or \
                        isinstance(index.dtype, pd.CategoricalDtype) and \
                        pd.api.types.is_string_dtype(
                            index.dtype.categories.dtype):
                    if len(index) == 0 and isinstance(index.dtype, object):
                        index = index.astype(pd.StringDtype())
                    result = to_r(
                        python_object.reset_index(drop=True),
                        (f'{python_object_name}.reset_index(drop=True)',
                         rmemory), format=format, rownames=_convert_names(
                            index, 'rownames', python_object,
                            python_object_name, rmemory))
                    converted_index_separately = True
                elif not (isinstance(index, pd.RangeIndex) and
                          index.start == 0 and index.stop == len(python_object)
                          and index.step == 1):
                    error_message = (
                        f'{python_object_name} is a pandas '
                        f'{"DataFrame" if is_df else "Series"} where '
                        f'{python_object_name}.index is neither the default '
                        f'index (pd.RangeIndex(len({python_object_name}))) '
                        f'nor a string or string categorical Index, but R '
                        f'rownames must be strings.\nBefore calling to_r(), '
                        f'call reset_index() to retain the index as a column, '
                        f'or reset_index(drop=True) to discard the index.')
                    raise TypeError(error_message)
                # Require columns to be strings (or string categoricals)
                if is_df and python_object.shape[1] > 0:
                    if not (pd.api.types.is_string_dtype(
                            python_object.columns.dtype) or
                            isinstance(python_object.columns.dtype,
                                       pd.CategoricalDtype) and
                            pd.api.types.is_string_dtype(
                                python_object.columns.dtype.categories.dtype)):
                        error_message = (
                            f'{python_object_name} is a pandas DataFrame '
                            f'where {python_object_name}.columns is not a '
                            f'string or string categorical Index (it has data '
                            f'type {python_object.columns.dtype!r}), but R '
                            f'colnames must be strings')
                        raise TypeError(error_message)
                    if python_object.columns.dtype == object:
                        for col in python_object.columns:
                            if not isinstance(col, str):
                                error_message = (
                                    f'{python_object_name} is a pandas '
                                    f'DataFrame containing a non-string '
                                    f'column named {col!r}, but R colnames '
                                    f'must be strings')
                                raise TypeError(error_message)
            # If python_object is a DataFrame with the default index (or for
            # which we've already handled the index)...
            if is_df and isinstance(index, pd.RangeIndex) and \
                    index.start == 0 and index.stop == len(python_object) and \
                    index.step == 1:
                # If format='matrix', all columns must have the same data type,
                # which cannot be datetime64, PeriodDtype or ArrowDtype(
                # pa.timestamp()); concatenate the columns into a single Series
                unique_dtypes = set(python_object.dtypes.unique())
                if format == 'matrix':
                    if len(unique_dtypes) > 1:
                        error_message = (
                            f"{python_object_name} is a pandas DataFrame with "
                            f"a mix of data types (i.e. "
                            f"len(set({python_object_name}.dtypes)) > 1), so "
                            f"format='matrix' cannot be specified")
                        raise TypeError(error_message)
                    dtype = next(iter(unique_dtypes))
                    is_perioddtype = isinstance(dtype, pd.PeriodDtype)
                    is_arrowdtype = isinstance(dtype, pd.ArrowDtype)
                    if ((pa.types.is_timestamp(dtype.pyarrow_dtype) or
                         pa.types.is_date64(dtype.pyarrow_dtype))
                            if is_arrowdtype else
                            (pd.api.types.is_datetime64_any_dtype(dtype) or
                             is_perioddtype)):
                        dtype = str(dtype) if is_arrowdtype else \
                            'PeriodDtype' if is_perioddtype else 'datetime64'
                        error_message = (
                            f"{python_object_name} is a pandas DataFrame "
                            f"where all columns have data type "
                            f"{str(dtype)!r}, so format='matrix' cannot be "
                            f"specified.\n(POSIXct matrices are not "
                            f"supported, only vectors and data.frames.)")
                        raise TypeError(error_message)
                    result = to_r(pd.concat([python_object[col]
                                             for col in python_object],
                                            ignore_index=True), (
                        f'pd.concat([df[col] for col in (df := '
                        f'{python_object_name})], ignore_index=True)',
                        rmemory))
                    shape = python_object.shape
                    colnames = _convert_names(python_object.columns,
                                              'colnames', python_object,
                                              python_object_name, rmemory)
                else:
                    # Disallow bytestring, IntervalDtype and SparseDtype
                    # columns, as well as pd.ArrowDtype if its pyarrow dtype is
                    # unsupported
                    for dtype in unique_dtypes:
                        # Use str(dtype).startswith('|S') rather than
                        # np.issubdtype(dtype, 'S') to avoid np.issubdtype
                        # erroring on pandas-specific dtypes
                        dtype = str(dtype)
                        if dtype.startswith('|S'):
                            error_message = (
                                f'{python_object_name} is a pandas DataFrame '
                                f'that contains one or more columns with the '
                                f'bytestring data type {dtype!r}, which '
                                f'lacks an R equivalent.\nConsider converting '
                                f'these columns to a string data type before '
                                f'calling to_r().')
                            raise TypeError(error_message)
                    if any(isinstance(dtype, pd.IntervalDtype)
                           for dtype in unique_dtypes):
                        error_message = (
                            f'{python_object_name} is a pandas DataFrame that '
                            f'contains one or more Interval columns '
                            f'(pd.IntervalDtype), which lack an R equivalent')
                        raise TypeError(error_message)
                    if any(isinstance(dtype, pd.SparseDtype)
                           for dtype in unique_dtypes):
                        error_message = (
                            f'{python_object_name} is a pandas DataFrame that '
                            f'contains one or more Sparse columns '
                            f'(pd.SparseDtype), which lack an R equivalent')
                        raise TypeError(error_message)
                    for dtype in unique_dtypes:
                        if isinstance(dtype, pd.ArrowDtype) and not \
                                _is_supported_pyarrow_dtype(
                                    dtype.pyarrow_dtype):
                            error_message = (
                                f'{python_object_name} is a pandas DataFrame '
                                f'that contains one or more columns with the '
                                f'pyarrow data type '
                                f'{str(dtype.pyarrow_dtype)!r}, which lacks '
                                f'an R equivalent.\n(Only integer, '
                                f'floating-point, temporal, string, '
                                f'large_string, dictionary<values=string>, '
                                f'and null pyarrow data types are supported.)')
                            raise TypeError(error_message)
                    # Disallow non-string categoricals
                    if any(isinstance(dtype, pd.CategoricalDtype) and not
                            pd.api.types.is_string_dtype(
                                dtype.categories.dtype)
                           for dtype in unique_dtypes):
                        error_message = (
                            f'{python_object_name} is a pandas DataFrame that '
                            f'contains one or more categorical columns with '
                            f'non-string categories, which lack an R '
                            f'equivalent')
                        raise TypeError(error_message)
                    # Convert float16/float128 (which Arrow does not support)
                    # to float. Use dtype == 'float128', not dtype ==
                    # np.float128, since np.float128 isn't defined on Windows.
                    if any(dtype == np.float16 or dtype == 'float128'
                           for dtype in unique_dtypes):
                        python_object = python_object.astype({
                            col: float for col in
                            python_object.select_dtypes(['float128'])})
                    # If any columns are uint32/uint64/int64 (which we handle
                    # zero-copy as Series), object, or complex, fall back to
                    # converting to dict, converting each column to R
                    # separately, then converting to data.frame
                    if any(dtype == np.uint32 or dtype == np.uint64 or
                           dtype == np.int64 or dtype == object or
                           dtype == pd.UInt32Dtype() or
                           dtype == pd.UInt64Dtype() or
                           dtype == pd.Int64Dtype() or
                           (isinstance(dtype, pd.ArrowDtype) and (
                                pa.types.is_uint32(dtype.pyarrow_dtype) or
                                pa.types.is_uint64(dtype.pyarrow_dtype) or
                                pa.types.is_int64(dtype.pyarrow_dtype))) or
                           dtype == np.complex64 or dtype == np.complex128
                           for dtype in unique_dtypes):
                        result = to_r(
                            dict(python_object.items()),
                            # deliberately leave python_object_name as-is (if
                            # column 'a' has an error, the error message will
                            # correctly say python_object['a'])
                            (python_object_name, rmemory), format=format)
                        result = _as_data_frame(result, rmemory)
                    # Otherwise, convert to Arrow
                    else:
                        # Convert PeriodDtype columns to datetime64
                        if any(isinstance(dtype, pd.PeriodDtype)
                               for dtype in unique_dtypes):
                            python_object = python_object.assign(**{
                                col: pd.Series(pd.Index(python_object[col])
                                               .to_timestamp())
                                for col, dtype in python_object.dtypes.items()
                                if isinstance(dtype, pd.PeriodDtype)})
                        # Convert ArrowDtype(pa.dictionary(pa.int64(),
                        # pa.string()) to ArrowDtype(pa.dictionary(pa.int32(),
                        # pa.string())
                        if any(isinstance(dtype, pd.ArrowDtype) and
                               pa.types.is_dictionary(dtype.pyarrow_dtype) and
                               pa.types.is_int64(
                                   dtype.pyarrow_dtype.index_type)
                               for dtype in unique_dtypes):
                            assigns = {}
                            for col, dtype in python_object.dtypes.items():
                                if not (isinstance(dtype, pd.ArrowDtype) and
                                        pa.types.is_dictionary(
                                            dtype.pyarrow_dtype) and
                                        pa.types.is_int64(
                                            dtype.pyarrow_dtype.index_type)):
                                    continue
                                new_dtype = pa.dictionary(
                                    pa.int32(), dtype.pyarrow_dtype.value_type,
                                    ordered=dtype.pyarrow_dtype.ordered)
                                assigns[col] = \
                                    pd.Series(pa.array(python_object[col])
                                              .cast(new_dtype),
                                              index=python_object.index,
                                              dtype=pd.ArrowDtype(new_dtype))
                            python_object = python_object.assign(**assigns)
                        # If length-0, convert ArrowDtype(pa.timestamp()) to
                        # DatetimeTZDtype to work around
                        # github.com/pandas-dev/pandas/issues/57840
                        if len(python_object) == 0:
                            if any(isinstance(dtype, pd.ArrowDtype) and
                                   pa.types.is_timestamp(dtype.pyarrow_dtype)
                                    for dtype in unique_dtypes):
                                python_object = python_object.astype({
                                    col: pd.DatetimeTZDtype(
                                        tz=dtype.pyarrow_dtype.tz)
                                    if dtype.pyarrow_dtype.tz is not None
                                    else 'datetime64[ns]'
                                    for col, dtype in
                                    python_object.dtypes.items()
                                    if isinstance(dtype, pd.ArrowDtype) and
                                    pa.types.is_timestamp(
                                        dtype.pyarrow_dtype)})
                        # Finally, convert to Arrow
                        arrow = pa.RecordBatch.from_pandas(python_object)
            # If python_object is a Series with the default index (or for which
            # we've already handled the index), or an Index/DatetimeIndex...
            elif (is_series and python_object.index.equals(
                    pd.RangeIndex(len(python_object)))) or is_index:
                # Disallow bytestring, IntervalDtype and SparseDtype columns
                # as well as pd.ArrowDtype if its pyarrow dtype is unsupported
                type_name = 'Series' if is_series else 'Index'
                if str(dtype).startswith('|S'):
                    error_message = (
                        f'{python_object_name} is a pandas {type_name} with '
                        f'the bytestring data type {str(dtype)!r}, which '
                        f'lacks an R equivalent.\nConsider converting to a '
                        f'string data type before calling to_r().')
                    raise TypeError(error_message)
                if isinstance(dtype, pd.IntervalDtype):
                    error_message = (
                        f'{python_object_name} is a pandas {type_name} with '
                        f'Interval data type (pd.IntervalDtype), which lacks '
                        f'an R equivalent')
                    raise TypeError(error_message)
                if isinstance(dtype, pd.SparseDtype):
                    error_message = (
                        f'{python_object_name} is a pandas {type_name} with '
                        f'Sparse data type (pd.SparseDtype), which lacks an R '
                        f'equivalent')
                    raise TypeError(error_message)
                is_arrowdtype = isinstance(dtype, pd.ArrowDtype)
                if is_arrowdtype and not \
                        _is_supported_pyarrow_dtype(dtype.pyarrow_dtype):
                    error_message = (
                        f'{python_object_name} is a pandas {type_name} with '
                        f'the pyarrow data type {str(dtype.pyarrow_dtype)!r}, '
                        f'which lacks an R equivalent.\n(Only integer, '
                        f'floating-point, temporal, string, large_string, '
                        f'dictionary<values=string>, and null pyarrow data '
                        f'types are supported.)')
                    raise TypeError(error_message)
                # Disallow non-string categoricals
                if isinstance(dtype, pd.CategoricalDtype) and not \
                        pd.api.types.is_string_dtype(dtype.categories.dtype):
                    error_message = (
                        f'{python_object_name} is a pandas categorical '
                        f'{type_name} with non-string categories, which lacks '
                        f'an R equivalent')
                    raise TypeError(error_message)
                # If complex, convert with NumPy
                if dtype == np.complex64 or dtype == np.complex128:
                    result = to_r(python_object.to_numpy(),
                                  (f'{python_object_name}.to_numpy()',
                                   rmemory))
                # Otherwise, convert to Arrow
                else:
                    # Convert float16/float128 (which Arrow does not support)
                    # to float
                    if dtype == np.float16 or dtype == 'float128':
                        python_object = python_object.astype(float)
                    # Convert PeriodDtype to datetime64
                    if is_series and isinstance(dtype, pd.PeriodDtype):
                        python_object = \
                            pd.Series(pd.Index(python_object).to_timestamp())
                    # Convert ArrowDtype(pa.dictionary(pa.int64(), pa.string())
                    # to ArrowDtype(pa.dictionary(pa.int32(), pa.string())
                    if is_arrowdtype and pa.types.is_dictionary(
                            dtype.pyarrow_dtype) and pa.types.is_int64(
                            dtype.pyarrow_dtype.index_type):
                        new_dtype = pa.dictionary(
                            pa.int32(), dtype.pyarrow_dtype.value_type,
                            ordered=dtype.pyarrow_dtype.ordered)
                        if is_series:
                            python_object = pd.Series(
                                pa.array(python_object).cast(new_dtype),
                                index=python_object.index,
                                dtype=pd.ArrowDtype(new_dtype))
                        else:  # is_index
                            python_object = pd.Index(
                                pa.array(python_object).cast(new_dtype),
                                dtype=pd.ArrowDtype(new_dtype))
                    # Finally, convert to Arrow
                    if len(python_object) == 0 and is_arrowdtype and \
                            pa.types.is_timestamp(dtype.pyarrow_dtype):
                        # Work around github.com/pandas-dev/pandas/issues/57840
                        arrow = pa.array([], type=python_object.dtype
                                                  .pyarrow_dtype)
                    else:
                        if dtype == object:
                            arrow = _convert_object_to_arrow(
                                python_object,
                                f"{python_object_name} is a pandas Series "
                                f"with data type 'object'", is_pandas=True)
                            if not isinstance(arrow, pa.Array):
                                # complex; convert via NumPy, not Arrow
                                python_object = arrow
                                arrow = None
                                result = to_r(python_object,
                                              (python_object_name, rmemory))
                        else:
                            # View uint32 as int32 and uint64/int64 as double,
                            # to allow zero-copy. For uint64/int64, we will
                            # reinterpret the double array as bit64::integer64
                            # on the R side by manually adding 'integer64' as a
                            # class.
                            if dtype == np.uint32:
                                arrow = pa.array(
                                    python_object.values.view(np.int32))
                            elif dtype == np.uint64 or dtype == np.int64:
                                arrow = pa.array(
                                    python_object.values.view(np.float64))
                                add_integer64 = True
                            elif isinstance(dtype, pd.ArrowDtype):
                                # Match the unsigned behavior in the zero-copy
                                # case by casting pa.uint32 to pa.int32 and
                                # pa.uint64 to pa.int64
                                arrow = python_object.array._pa_array
                                if pa.types.is_uint32(dtype.pyarrow_dtype):
                                    arrow = arrow\
                                        .cast(pa.int32(), safe=False)
                                elif pa.types.is_uint64(dtype.pyarrow_dtype):
                                    arrow = arrow\
                                        .cast(pa.int64(), safe=False)
                            else:
                                # Match the unsigned behavior in the zero-copy
                                # case by casting UInt32Dtype to Int32Dtype and
                                # UInt64Dtype to Int64Dtype
                                if dtype == pd.UInt32Dtype():
                                    python_object = python_object\
                                        .astype(pd.Int32Dtype())
                                elif dtype == pd.UInt64Dtype():
                                    python_object = python_object\
                                        .astype(pd.Int64Dtype())
                                arrow = pa.array(python_object,
                                                 from_pandas=True)
                        # Combine chunks when python_object is a
                        # pa.ChunkedArray instead of a pa.Array (which happens
                        # when converting an ArrowDtype DataFrame to a matrix,
                        # due to the pd.concat)
                        if isinstance(arrow, pa.ChunkedArray):
                            arrow = arrow.combine_chunks()
            elif is_pandas_timestamp_or_timedelta:
                result = to_r(pa.array([python_object]), (
                    f'pyarrow.array([{python_object_name}])', rmemory))
            elif is_pandas_period:
                result = to_r(pa.array([python_object.to_timestamp()]), (
                    f'pyarrow.array([{python_object_name}.to_timestamp()])',
                    rmemory))
            elif python_object is pd.NA or python_object is pd.NaT:
                result = rmemory.protect(_rlib.Rf_allocVector(_rlib.LGLSXP, 1))
                _rlib.SET_LOGICAL_ELT(result, 0, _rlib.R_NaInt)
        elif is_numpy:
            if is_ndarray and python_object.ndim > 0 and \
                    max(python_object.shape) > 2_147_483_647:
                max_dimension = np.argmax(python_object.shape)
                error_message = (
                    f'{python_object_name} is a {python_object.ndim:,}D NumPy '
                    f'array with length {max(python_object.shape):,} along '
                    f'dimension {max_dimension:,}, more than INT32_MAX '
                    f'(2,147,483,647), the maximum supported in R')
                raise ValueError(error_message)
            if python_object.ndim == 0:  # generic or 0D ndarray
                try:
                    is_nat = np.isnat(python_object)
                except TypeError:
                    is_nat = False
                if is_nat:
                    # np.datetime64('NaT', time_unit) or
                    # np.timedelta64('NaT', time_unit); return NA
                    result = rmemory.protect(
                        _rlib.Rf_allocVector(_rlib.LGLSXP, 1))
                    _rlib.SET_LOGICAL_ELT(result, 0, _rlib.R_NaInt)
                else:
                    if dtype == 'float128':
                        # np.float128(...).item() is a null-op; must cast
                        python_object = float(python_object)
                        python_object_name = f'float({python_object_name})'
                    else:
                        is_datetime64 = dtype.type == np.datetime64
                        is_timedelta64 = dtype.type == np.timedelta64
                        if is_datetime64 or is_timedelta64:
                            if dtype.name[-3:-1] == 'ns' or dtype.name[-2] in \
                                    ('DWMY' if is_datetime64 else 'MY'):
                                # datetime64[ns], timedelta64[ns],
                                # timedelta64[M] and timedelta64[Y] convert
                                # to int rather than datetime/timedelta;
                                # datetime64[D], datetime64[W],
                                # datetime64[M] and datetime64[Y] convert
                                # to date rather than datetime
                                new_dtype = 'datetime64[us]' \
                                    if is_datetime64 else 'timedelta64[us]'
                                python_object = python_object.astype(new_dtype)
                                python_object_name = \
                                    f'{python_object_name}' \
                                    f'.astype({new_dtype!r})'
                        python_object = python_object.item()
                        python_object_name = f'{python_object_name}.item()'
                        if dtype.type == np.void and \
                                isinstance(python_object, bytes):
                            # unstructured void array; equivalent to an R raw,
                            # but .item() converts it to bytes rather than
                            # bytearray for some reason
                            python_object = bytearray(python_object)
                            python_object_name = \
                                f'bytearray({python_object_name})'
                    result = to_r(python_object, (python_object_name, rmemory))
            else:
                # Disallow bytestring and structured array (np.void) dtypes and
                # zero-length datetime64 and timedelta64 arrays with no time
                # unit, and multidimensional datetime64 arrays (or dtype=object
                # arrays of datetimes)
                if np.issubdtype(dtype, 'S'):
                    error_message = (
                        f'{python_object_name} is a NumPy array with the '
                        f'bytestring data type {str(dtype)!r}, which lacks an '
                        f'R equivalent.\nConsider converting to a string data '
                        f'type before calling to_r().')
                    raise TypeError(error_message)
                if dtype.type == np.void:
                    error_message = (
                        f'{python_object_name} is a structured NumPy array, '
                        f'which lacks an R equivalent')
                    raise TypeError(error_message)
                is_datetime64 = dtype.type == np.datetime64
                is_timedelta64 = dtype.type == np.timedelta64
                if python_object.size == 0 and (
                        is_datetime64 or is_timedelta64) and \
                        '[' not in str(dtype):
                    dtype = 'datetime64' if is_datetime64 else 'timedelta64'
                    error_message = (
                        f'{python_object_name} is a zero-length NumPy array '
                        f'with data type {str(dtype)!r} and no time unit, '
                        f'which lacks an R equivalent')
                    raise TypeError(error_message)
                if is_datetime64 and is_multidimensional_ndarray and \
                        format != 'data.frame':
                    if is_matrix:
                        error_message = (
                            f"{python_object_name} is a 2D NumPy array of "
                            f"datetimes, so format='data.frame' must be "
                            f"specified.\n(POSIXct matrices are not "
                            f"supported, only vectors and data.frames.)")
                        raise TypeError(error_message)
                    else:
                        error_message = (
                            f"{python_object_name} is a "
                            f"{python_object.ndim:,}D NumPy array of "
                            f"datetimes, which cannot be converted to R.\n"
                            f"(POSIXct arrays are not supported, only vectors "
                            f"and data.frames.)")
                        raise TypeError(error_message)
                # Convert timedelta64 with time units of greater than
                # seconds to seconds
                if is_timedelta64:
                    time_unit = python_object.dtype.name[-2]
                    if time_unit in 'mhDWMY':
                        python_object = python_object.astype('timedelta64[s]')
                # Convert float16/float128 (which Arrow does not support) to
                # float
                if dtype == np.float16 or dtype == 'float128':
                    python_object = python_object.astype(float)
                # Handle complex dtypes separately, since Arrow doesn't support
                # them
                if dtype == np.complex64 or dtype == np.complex128:
                    if dtype == np.complex64:
                        python_object = python_object.astype(complex)
                    if python_object.size == 0:
                        # In NumPy, empty >=2D arrays have a length of 1, but
                        # in R they have a length of 0
                        result = rmemory.protect(
                            _rlib.Rf_allocVector(_rlib.CPLXSXP, 0))
                    else:
                        result = rmemory.protect(_rlib.Rf_allocVector(
                            _rlib.CPLXSXP, python_object.size))
                        _ffi.memmove(
                            _rlib.COMPLEX(result),
                            _ffi.buffer(_ffi.cast(
                                'Rcomplex *',
                                python_object.__array_interface__['data'][0]),
                                python_object.nbytes),
                            python_object.size * _ffi.sizeof('Rcomplex'))
                else:
                    # View uint32 as int32 and uint64/int64 as double, to allow
                    # zero-copy. For uint64/int64, we will reinterpret the
                    # double array as bit64::integer64 on the R side by
                    # manually adding 'integer64' as a class.
                    if dtype == np.uint32:
                        python_object = python_object.view(np.int32)
                    elif dtype == np.uint64 or dtype == np.int64:
                        python_object = python_object.view(np.float64)
                        add_integer64 = True
                    # Finally, convert to Arrow
                    flat_python_object = python_object.ravel('F')
                    if dtype == object:
                        arrow = _convert_object_to_arrow(
                            flat_python_object,
                            f"{python_object_name} is a NumPy array with data "
                            f"type 'object'")
                        if not isinstance(arrow, pa.Array):
                            # complex; convert via NumPy, not Arrow
                            python_object = arrow.reshape(python_object.shape)
                            arrow = None
                            result = to_r(python_object,
                                          (python_object_name, rmemory))
                    else:
                        arrow = pa.array(flat_python_object)
                    # If NumPy array was a timedelta64 (or an object array that
                    # converted to an Arrow DurationArray) and the output
                    # format is a data.frame, reshape now to avoid having to
                    # convert to a matrix as an intermediate step
                    if format == 'data.frame' and (
                            is_timedelta64 or arrow is not None and
                            pa.types.is_duration(arrow.type)):
                        nrows, ncols = python_object.shape
                        arrow = pa.RecordBatch.from_arrays([
                            arrow[i * nrows: (i + 1) * nrows]
                            for i in range(ncols)],
                            names=colnames if colnames is not None else [
                                f'V{i}' for i in range(1, ncols + 1)])
                        is_matrix = is_multidimensional_ndarray = False
        # Only support arrow Arrays when recursing, not at the top level
        elif is_pyarrow_array:
            if top_level:
                error_message = (
                    f'{python_object_name} has unsupported type '
                    f'{type(python_object).__name__!r}')
                raise TypeError(error_message)
            elif isinstance(python_object, pa.ExtensionArray):
                error_message = (
                    f'{python_object_name} has unsupported type '
                    f'pyarrow.ExtensionArray')
                raise TypeError(error_message)
            arrow = python_object
        elif is_sparse:
            if python_object.nnz > 2_147_483_647:
                error_message = (
                    f'{python_object_name} is a sparse {sparse_supertype} '
                    f'with {python_object.nnz:,} non-zero elements, more than '
                    f'INT32_MAX (2,147,483,647), the maximum supported in R')
                raise ValueError(error_message)
            if max(python_object.shape) > 2_147_483_647:
                dimension_name = \
                    'rows' if python_object.shape[0] > \
                              python_object.shape[1] else 'columns'
                error_message = (
                    f'{python_object_name} is a sparse {sparse_supertype} '
                    f'with {max(python_object.shape):,} {dimension_name}, '
                    f'more than INT32_MAX (2,147,483,647), the maximum '
                    f'supported in R')
                raise ValueError(error_message)
            if not _require(b'Matrix', rmemory):
                error_message = (
                    'please install the Matrix R package to convert sparse '
                    'arrays or matrices to R')
                raise ImportError(error_message)
            if python_object.dtype == np.float64:
                dtype_code = b'd'
            elif python_object.dtype == bool:
                dtype_code = b'l'
            else:
                error_message = (
                    f'{python_object_name} is a '
                    f'{type(python_object).__name__!r} of data type '
                    f'{python_object.dtype}, which is not supported in R')
                raise TypeError(error_message)
            
            # Create empty *gCMatrix, *gRMatrix or *gTMatrix
            result = rmemory.protect(
                _rlib.R_do_new_object(_rlib.R_getClassDef(
                    dtype_code + b'gRMatrix' if is_csr else
                    dtype_code + b'gCMatrix' if is_csc else
                    dtype_code + b'gTMatrix')))
        
            # Assign data (x)
            _rlib.R_do_slot_assign(result, _rlib.Rf_install(b'x'),
                to_r(python_object.data, (
                    f'{python_object_name}.data', rmemory)))
        
            # Assign indices (i/j) and indptr (p)
            if is_coo:
                _rlib.R_do_slot_assign(
                    result, _rlib.Rf_install(b'i'),
                    to_r(python_object.row.astype(np.int32, copy=False), (
                        f'{python_object_name}.row', rmemory)))
                _rlib.R_do_slot_assign(
                    result, _rlib.Rf_install(b'j'),
                    to_r(python_object.col.astype(np.int32, copy=False), (
                        f'{python_object_name}.col', rmemory)))
            else:
                _rlib.R_do_slot_assign(
                    result, _rlib.Rf_install(b'i' if is_csc else b'j'),
                    to_r(python_object.indices.astype(np.int32, copy=False), (
                        f'{python_object_name}.indices', rmemory)))
                _rlib.R_do_slot_assign(result, _rlib.Rf_install(b'p'),
                    to_r(python_object.indptr.astype(np.int32, copy=False), (
                        f'{python_object_name}.indptr', rmemory)))
        
            # Assign dims
            dims = rmemory.protect(_rlib.Rf_allocVector(_rlib.INTSXP, 2))
            _rlib.INTEGER(dims)[0] = python_object.shape[0]
            _rlib.INTEGER(dims)[1] = python_object.shape[1]
            _rlib.R_do_slot_assign(result, _rlib.Rf_install(b'Dim'), dims)
            
            # Assign dimnames
            dimnames = rmemory.protect(
                _rlib.Rf_allocVector(_rlib.VECSXP, 2))
            _rlib.SET_VECTOR_ELT(dimnames, 0, rownames
                if rownames is not None else _rlib.R_NilValue)
            _rlib.SET_VECTOR_ELT(dimnames, 1, colnames
                if colnames is not None else _rlib.R_NilValue)
            _rlib.R_do_slot_assign(result, _rlib.Rf_install(b'Dimnames'),
                                   dimnames)
        
            # Assign factors (empty list)
            _rlib.R_do_slot_assign(result, _rlib.Rf_install(b'factors'),
                                   rmemory.protect(
                                       _rlib.Rf_allocVector(_rlib.VECSXP, 0)))

        # Let Arrow handle conversions of temporal types
        elif isinstance(python_object, (datetime.date, datetime.datetime,
                                        datetime.time, datetime.timedelta)):
            if isinstance(python_object, datetime.time) and \
                    python_object.tzinfo is not None:
                error_message = (
                    f'{python_object_name} is a datetime.time object with a '
                    f'non-missing time zone, which cannot be represented in R')
                raise TypeError(error_message)
            result = to_r(pa.array([python_object]), (
                f'pyarrow.array([{python_object_name}])', rmemory))
        elif isinstance(python_object, complex):
            result = rmemory.protect(_rlib.Rf_allocVector(_rlib.CPLXSXP, 1))
            element = _rlib.COMPLEX(result)[0]
            element.r = python_object.real
            element.i = python_object.imag
        else:
            error_message = (
                f'{python_object_name} has unsupported type '
                f'{type(python_object).__name__!r}')
            raise TypeError(error_message)
        # If python_object was converted to Arrow...
        if arrow is not None:
            if is_polars:
                # ...for DictionaryArrays (Categoricals and Enums), work around
                # github.com/apache/arrow/issues/39603 by casting the index
                # type from uint32 to int32, work around
                # github.com/pola-rs/polars/issues/14709 by casting Enums to
                # ordered (not unordered) DictionaryArrays, and stop R warning
                # about "Coercing dictionary values to R character factor
                # levels" by casting large_string to string (and also work
                # around github.com/apache/arrow/issues/40128 when arrow has a
                # length of 0)...
                if isinstance(arrow, pa.Array):
                    if dtype == pl.Categorical or dtype == pl.Enum:
                        if len(arrow) == 0:
                            arrow = pa.DictionaryArray.from_arrays(
                                pa.array([], type=pa.int32()),
                                arrow.dictionary.cast(pa.string()),
                                ordered=dtype == pl.Enum)
                        else:
                            arrow = arrow.cast(pa.dictionary(
                                pa.int32(), pa.string(),
                                ordered=dtype == pl.Enum))
                else:  # isinstance(arrow, pa.RecordBatch)
                    if len(arrow) == 0:
                        arrow = pa.RecordBatch.from_arrays([
                            pa.DictionaryArray.from_arrays(
                                pa.array([], type=pa.int32()),
                                array.dictionary.cast(pa.string()),
                                ordered=dtype == pl.Enum)
                            if dtype == pl.Enum or dtype == pl.Categorical
                            else array for array, dtype in
                            zip(arrow, python_object.dtypes)],
                            names=arrow.schema.names)
                    else:
                        arrow = pa.RecordBatch.from_arrays([
                            array.cast(pa.dictionary(pa.int32(), pa.string(),
                                                     ordered=dtype == pl.Enum))
                            if dtype == pl.Enum or dtype == pl.Categorical
                            else array for array, dtype in
                            zip(arrow, python_object.dtypes)],
                            names=arrow.schema.names)
            # ...export to C...
            array = pyarrow_ffi.new('struct ArrowArray*')
            schema = pyarrow_ffi.new('struct ArrowSchema*')
            array_ptr = int(pyarrow_ffi.cast('uintptr_t', array))
            schema_ptr = int(pyarrow_ffi.cast('uintptr_t', schema))
            array_ptr_STRSXP = \
                _string_to_character_vector(str(array_ptr), rmemory)
            schema_ptr_STRSXP = \
                _string_to_character_vector(str(schema_ptr), rmemory)
            arrow._export_to_c(array_ptr, schema_ptr)
            # ...then import into R with arrow::Array/RecordBatch$import_from_c
            arrow_namespace = rmemory.protect(_rlib.R_FindNamespace(
                _bytestring_to_character_vector(b'arrow', rmemory)))
            Array_or_RecordBatch = _rlib.Rf_eval(_rlib.Rf_findVarInFrame(
                arrow_namespace, _rlib.Rf_install(
                    b'RecordBatch' if is_df else b'Array')), _rlib.R_BaseEnv)
            function_call = rmemory.protect(_rlib.Rf_lang3(
                _rlib.Rf_install(b'import_from_c'),
                array_ptr_STRSXP, schema_ptr_STRSXP))
            arrow_R = _call(function_call, rmemory,
                            'cannot import Arrow object from C into R',
                            environment=Array_or_RecordBatch)
            # ...and convert the R RecordBatch to a data.frame or matrix, or
            # the R Arrow Array to a vector (using array$as_vector() instead of
            # as.vector(array) to work around
            # github.com/Bioconductor/DelayedArray/issues/114)
            if is_df:
                function_call = rmemory.protect(
                    _rlib.Rf_lang2(_rlib.Rf_install(b'as.data.frame'),
                                   arrow_R))
                result = _call(function_call, rmemory,
                               'cannot convert R Arrow array to data.frame')
            else:
                function_call = rmemory.protect(
                    _rlib.Rf_lang1(_rlib.Rf_install(b'as_vector')))
                result = _call(function_call, rmemory,
                               'cannot convert R arrow array to vector',
                               environment=arrow_R)
        # If python_object is a multidimensional NumPy array, or a DataFrame
        # but format == 'matrix' and we did not convert the index separately,
        # convert to a matrix/array by adding dims and (if specified) rownames
        # and colnames
        if is_multidimensional_ndarray or is_df and format == 'matrix' and \
                not converted_index_separately:
            # For difftime and vctrs::unspecified, manually add 'array' and (if
            # 2D) 'matrix' as classes (R's array()/matrix() functions do this
            # too, but also overwrite any existing clases, like difftime)
            if _rlib.Rf_inherits(result, b'difftime') or \
                    _rlib.Rf_inherits(result, b'vctrs_unspecified'):
                classes_to_add = (b'array',) \
                    if is_ndarray and python_object.ndim > 2 else \
                    (b'matrix', b'array')
                classes = _rlib.Rf_getAttrib(result, _rlib.R_ClassSymbol)
                num_classes = _rlib.Rf_xlength(classes)
                new_classes = rmemory.protect(
                    _rlib.Rf_allocVector(_rlib.STRSXP,
                                         num_classes + len(classes_to_add)))
                for i in range(num_classes):
                    _rlib.SET_STRING_ELT(new_classes, i,
                                         _rlib.STRING_ELT(classes, i))
                for i, class_name in enumerate(classes_to_add,
                                               start=num_classes):
                    _rlib.SET_STRING_ELT(new_classes, i, _rlib.Rf_mkCharLenCE(
                        class_name, len(class_name), _rlib.CE_UTF8))
                _rlib.Rf_setAttrib(result, _rlib.R_ClassSymbol, new_classes)
            if shape is None:
                shape = python_object.shape
            dim = rmemory.protect(
                _rlib.Rf_allocVector(_rlib.INTSXP, len(shape)))
            for i in range(len(shape)):
                _rlib.INTEGER(dim)[i] = shape[i]
            _rlib.Rf_setAttrib(result, _rlib.R_DimSymbol, dim)
            if rownames is not None or colnames is not None:
                if not top_level:
                    if rownames is not None:
                        rowname_length = _rlib.Rf_xlength(rownames)
                        if rowname_length != len(python_object):
                            error_message = (
                                f'rownames have length {rowname_length:,}, '
                                f'but {python_object_name} has length '
                                f'{len(python_object):,}')
                            raise ValueError(error_message)
                    if colnames is not None:
                        colname_length = _rlib.Rf_xlength(colnames)
                        if _rlib.Rf_xlength(colnames) != shape[1]:
                            error_message = (
                                f'colnames have length {colname_length:,}, '
                                f'but {python_object_name}.shape[1] is '
                                f'{shape[1]:,}')
                            raise ValueError(error_message)
                dimnames = rmemory.protect(
                    _rlib.Rf_allocVector(_rlib.VECSXP, 2))
                _rlib.SET_VECTOR_ELT(dimnames, 0, rownames
                    if rownames is not None else _rlib.R_NilValue)
                _rlib.SET_VECTOR_ELT(dimnames, 1, colnames
                    if colnames is not None else _rlib.R_NilValue)
                _rlib.Rf_setAttrib(result, _rlib.R_DimNamesSymbol, dimnames)
        # Otherwise, just add the rownames/names, if specified (except for
        # sparse arrays/matrices, where we already added the rownames above)
        elif rownames is not None and not is_sparse and not \
                isinstance(python_object, (list, tuple, dict)):
            if not top_level:
                rowname_length = _rlib.Rf_xlength(rownames)
                if isinstance(python_object, (int, float, str, complex)):
                    if rowname_length != 1:
                        error_message = (
                            f'rownames have length {rowname_length:,}, but '
                            f'{python_object_name} is a scalar, specifically '
                            f'of type {type(python_object).__name__!r}, so '
                            f'rownames must have a length of 1')
                        raise ValueError(error_message)
                elif rowname_length != len(python_object):
                    error_message = (
                        f'rownames have length {rowname_length:,}, but '
                        f'{python_object_name} has length '
                        f'{len(python_object):,}')
                    raise ValueError(error_message)
            _rlib.Rf_setAttrib(result, _rlib.R_RowNamesSymbol if is_df else
                                       _rlib.R_NamesSymbol, rownames)
        # For int64 and uint64, manually add 'integer64' as a class (otherwise
        # the result would just be a vector of doubles)
        if add_integer64:
            classes = _rlib.Rf_getAttrib(result, _rlib.R_ClassSymbol)
            num_classes = _rlib.Rf_xlength(classes)
            new_classes = rmemory.protect(
                _rlib.Rf_allocVector(_rlib.STRSXP, num_classes + 1))
            for i in range(num_classes):
                _rlib.SET_STRING_ELT(new_classes, i,
                                     _rlib.STRING_ELT(classes, i))
            _rlib.SET_STRING_ELT(
                new_classes, num_classes,
                _rlib.Rf_mkCharLenCE(b'integer64', 9, _rlib.CE_UTF8))
            _rlib.Rf_setAttrib(result, _rlib.R_ClassSymbol, new_classes)
        # If converting 2D NumPy array -> data.frame, convert result (which is
        # now an R matrix/array) to a data.frame with as.data.frame()
        if is_matrix and format == 'data.frame':
            function_call = rmemory.protect(
                _rlib.Rf_lang2(_rlib.Rf_install(b'as.data.frame'), result))
            result = _call(function_call, rmemory,
                           'cannot convert matrix to data.frame')
        # Either return the object (if recursing) or assign it to globalenv
        if top_level:
            _rlib.Rf_defineVar(
                _rlib.Rf_install(R_variable_name.encode('utf-8')),
                result, _rlib.R_GlobalEnv)
        else:
            return result
    finally:
        if top_level:
            rmemory.__exit__(None, None, None)


def to_py(R_statement: str, *,
          format: Literal['polars', 'pandas', 'pandas-pyarrow', 'numpy'] |
                  dict[Literal['vector', 'matrix', 'data.frame'],
                       Literal['polars', 'pandas', 'pandas-pyarrow',
                               'numpy']] | None = None,
          index: str | Literal[False] | None = None,
          squeeze: bool | None = None) -> Any:
    """
    Convert an R object, or the R object resulting from evaluating a single
    statement (line of R code), to Python. If the object is a list/S3 object,
    S4 object, or environment/R6 object, recursively convert each attribute/
    slot/field.
    
    Args:
        R_statement: the name of the R object to convert to Python, or R code
                     for a single R statement that yields the object to convert
        format: the Python format ('polars', 'pandas', 'pandas-pyarrow', or
                'numpy') to convert to, or a dictionary with 'vector', 'matrix'
                and/or 'data.frame' as keys and one of those four Python
                formats as values, to override the output format for only
                certain R variable types. If None, or if some keys are missing
                or have None as the format, defaults to
                options()['to_py_format'] (default: 'polars').
                If the R object is a data.frame and format='matrix', all
                columns must have the same data type. Must be None when
                R_statement evaluates to NULL, when it evaluates to an array of
                3 or more dimensions (these are always converted to NumPy
                arrays), or when the final result would be a Python scalar (see
                squeeze below).
        index: if a string, include the names of R vectors and the rownames of
               R matrices and data.frames (if pre sent) in the returned Python
               object, either as the index (if format is 'pandas' or
               'pandas-pyarrow') or as the first column of the output (if
               format is 'polars'; this forces the output to be a DataFrame
               even if the input was 1D). The index or first column will have
               this string as its name. If False, does not include an index or
               additional first column. If None, defaults to
               options()['index'] (default: 'index'), or sets index=False when
               format='numpy' (since numeric NumPy arrays cannot store string
               indices (except with the inefficient dtype=object). Must be None
               when format='numpy', or when the final result would be a Python
               scalar (see squeeze below).
        squeeze: whether to convert single-element R vectors, matrices and
                 arrays to Python scalars. (R data.frames are never converted
                 to Python scalars even if squeeze=True.) If None, defaults to
                 options()['squeeze'] (default: True). Must be None unless the
                 R object is a vector, matrix or array (raw vectors don't
                 count, because they always convert to Python scalars).

    Returns:
        The Python object that results from converting the R variable.
    """
    # Initialize ryp, if not already initialized
    _initialize_ryp()
    # Raise errors when format/squeeze is not None and we're not recursing
    raise_format_errors = format is not None
    raise_squeeze_errors = squeeze is not None
    # Keyword arguments default to what's in the config (handle index
    # later)
    if format is None:
        format = _config['to_py_format']
    elif isinstance(format, dict):
        for key in 'vector', 'matrix', 'data.frame':
            if key not in format or format[key] is None:
                format[key] = _config['to_py_format'][key] \
                    if isinstance(_config['to_py_format'], dict) else \
                    _config['to_py_format']
    if squeeze is None:
        squeeze = _config['squeeze']
    with _RMemory(_rlib) as rmemory:
        # Defer loading Arrow until calling to_py() or to_r() for speed. Allow
        # R_statement to be a tuple of (R_statement, R_object) instead of a
        # string. This is used when recursively calling to_py(). In this case,
        # don't check inputs for validity.
        with ignore_sigint():
            import pyarrow as pa
        from pyarrow.cffi import ffi as pyarrow_ffi
        if isinstance(R_statement, str):
            # Thread-safety check
            if threading.current_thread() is not _ryp_thread:
                error_message = (
                    "the R interpreter is not thread-safe, so ryp can't be "
                    "used by multiple Python threads simultaneously; see the "
                    "README for alternative parallelization strategies")
                raise RuntimeError(error_message)
            # Require arrow
            if not _require(b'arrow', rmemory):
                error_message = \
                    'please install the arrow R package to use ryp'
                raise ImportError(error_message)
            # Disallow empty code
            if not R_statement:
                error_message = 'R_statement must be a non-empty string'
                raise ValueError(error_message)
            # Check that format is valid
            _check_to_py_format(format)
            # Check that index is valid
            if index is not None and index is not False and \
                    not isinstance(index, str):
                if index is True:
                    error_message = \
                        'index must be False, None, or str, not True'
                    raise ValueError(error_message)
                else:
                    error_message = (
                        f'index must be False, None, or str, but has type '
                        f'{type(index).__name__!r}')
                    raise TypeError(error_message)
            # Check that squeeze is valid
            if not isinstance(squeeze, bool):
                error_message = (
                    f'squeeze must be True, False or None, but has type '
                    f'{type(squeeze).__name__!r}')
                raise TypeError(error_message)
            # If R_statement is a valid R variable name, just look up the
            # object it names; otherwise, evaluate it with r() and return the
            # object it evaluates to
            if _is_valid_R_variable_name(R_statement):
                # Get an R symbol object (SYMSXP) for this variable name
                R_symbol = _rlib.Rf_install(R_statement.encode('utf-8'))
                # Get the R object corresponding to this symbol
                R_object = _rlib.Rf_findVar(R_symbol, _rlib.R_GlobalEnv)
            else:
                # When calling r() from to_py(), return the object that results
                # from evaluating R_code, instead of None. Also, allow R_code
                # to be a tuple of (str, RMemory), where str is the string of R
                # code. This serves two purposes: it indicates it's a nested
                # call, and it stops the object that results from evaluating
                # R_code from being garbage-collected until to_py() returns.
                R_object = r((R_statement, rmemory))
        elif isinstance(R_statement, tuple) and len(R_statement) == 2 and \
                isinstance(R_statement[0], str) and \
                hasattr(R_statement[1], 'sxpinfo'):
            R_statement, R_object = R_statement
            raise_format_errors = False
            raise_squeeze_errors = False
        else:
            error_message = (
                f'R_statement must be str, but has type '
                f'{type(R_statement).__name__!r}')
            raise TypeError(error_message)
        # Get the type of the R object
        R_object_type = _rlib.TYPEOF(R_object)
        # Process the R object depending on its type.
        # See adv-r.hadley.nz/base-types.html for an overview of R's C types,
        # and lines 110-142 of
        # github.com/wch/r-source/blob/master/src/include/Rinternals.h
        # for the full list.
        R_arrow_array = None
        if R_object_type == _rlib.NILSXP:  # NULL
            if raise_format_errors:
                error_message = \
                    f'{R_statement!r} is NULL, so format must be None'
                raise TypeError(error_message)
            elif raise_squeeze_errors:
                error_message = \
                    f'{R_statement!r} is NULL, so squeeze must be None'
                raise TypeError(error_message)
            else:
                return None
        elif R_object_type == _rlib.SYMSXP:
            # symbol object; indicates undefined variable
            error_message = f'object {R_statement!r} not found'
            raise NameError(error_message)
        elif R_object_type == _rlib.ENVSXP:  # environment or R6 object
            # Recursively convert all the variables in the environment/R6
            # object to a dict. Ignore variables that can't be converted,
            # like functions. Note: this only converts R6 objects' public
            # variables!
            if raise_squeeze_errors:
                error_message = (
                    f'{R_statement!r} is an environment or R6 object, so '
                    f'squeeze must be None')
                raise TypeError(error_message)
            result = {}
            variable_names = _rlib.R_lsInternal(R_object, False)
            for i in range(_rlib.Rf_xlength(variable_names)):
                variable_name = \
                    _rlib.R_CHAR(_rlib.STRING_ELT(variable_names, i))
                variable = _rlib.Rf_findVarInFrame(
                    R_object, _rlib.Rf_install(variable_name))
                variable_name_str = _ffi.string(variable_name).decode('utf-8')
                try:
                    converted_variable = to_py(
                        (f'({R_statement})${variable_name_str}', variable),
                        format=format, index=index, squeeze=squeeze)
                except RecursionError as e:
                    error_message = (
                        f'maximum recursion depth exceeded when converting '
                        f'the environment or R6 object {R_statement!r} to '
                        f'Python; this usually means the object contains a '
                        f'circular reference')
                    raise ValueError(error_message) from e
                except TypeError as e:
                    if 'cannot be converted to Python' in str(e):
                        continue
                    else:
                        raise
                result[variable_name_str] = converted_variable
            return result
        elif R_object_type == _rlib.LGLSXP:
            # logical vector (i.e. bool array)
            if _rlib.Rf_inherits(R_object, b'LogMap'):
                # LogMap from the Seurat package is a LGLSXP but has a
                # .Data slot containing the actual boolean data
                return to_py((
                    f'({R_statement})@.Data',
                    _rlib.R_do_slot(R_object, _rlib.Rf_install(b'.Data'))),
                    format=format, index=index, squeeze=squeeze)
        elif R_object_type == _rlib.INTSXP:
            # integer vector (i.e. int32 array)
            pass
        elif R_object_type == _rlib.REALSXP:
            # real vector (i.e. float64 array)
            pass
        elif R_object_type == _rlib.CPLXSXP:
            # complex vector (i.e. complex array)
            pass
        elif R_object_type == _rlib.STRSXP:
            # character vector (i.e. string array)
            pass
        elif R_object_type == _rlib.VECSXP:
            # list or data frame
            # Recursively convert named lists to dicts, unnamed lists to
            # lists, and data.frames into data frames of the selected
            # format. If only some elements of the list have names, convert
            # to a dict, using each element's integer index as its key if
            # it is unnamed.
            is_df = _rlib.Rf_inherits(R_object, b'data.frame')
            names = _rlib.Rf_getAttrib(R_object, _rlib.R_NamesSymbol)
            if raise_squeeze_errors:
                error_message = (
                    f'{R_statement!r} is a {"data.frame" if is_df else "list"}'
                    f', so squeeze must be None')
                raise TypeError(error_message)
            if names == _rlib.R_NilValue:
                if is_df:  # zero-column data.frames may lack names
                    result = {}
                else:
                    try:
                        return [to_py((f'({R_statement})[{i + 1}]',
                                       _rlib.VECTOR_ELT(R_object, i)),
                                      format=format, index=index,
                                      squeeze=squeeze)
                                for i in range(_rlib.Rf_xlength(R_object))]
                    except RecursionError as e:
                        error_message = (
                            f'maximum recursion depth exceeded when '
                            f'converting the list {R_statement!r} to Python; '
                            f'this usually means the list contains a circular '
                            f'reference')
                        raise ValueError(error_message) from e
            else:
                names = [_ffi.string(_rlib.R_CHAR(_rlib.STRING_ELT(names, i)))
                         .decode('utf-8')
                         for i in range(_rlib.Rf_xlength(R_object))]
                # Raise an error if names are not unique ('' doesn't count,
                # unless R_object is a data.frame; see the next comment
                # below)
                unique_names = set()
                for name in names:
                    if not (name or is_df):
                        continue
                    if name in unique_names:
                        error_message = (
                            f'{R_statement!r} is a '
                            f'{"data.frame" if is_df else "list"} with a '
                            f'duplicate {"column " if is_df else ""}name, '
                            f'{name!r}')
                        raise ValueError(error_message)
                    unique_names.add(name)
                try:
                    # The "name or is_df" condition is because names[i] can be
                    # '' if some list elements have names and some don't (in
                    # which case we use the 0-based integer index as the dict
                    # key), but also if R_object is a data.frame with a column
                    # named '' (in which case we use '' as the dict key)
                    result = {name if name or is_df else i: to_py(
                        (f'({R_statement})[{i + 1}]' if not name else
                         f'({R_statement})${name}'
                         if _is_valid_R_variable_name(name) else
                         f'({R_statement})$`{name}`',
                         _rlib.VECTOR_ELT(R_object, i)),
                        format=format, index=index, squeeze=not is_df)
                        for i, name in enumerate(names)}
                except RecursionError as e:
                    error_message = (
                        f'maximum recursion depth exceeded when converting '
                        f'the list {R_statement!r} to Python; this usually '
                        f'means the list contains a circular reference')
                    raise ValueError(error_message) from e
                if not is_df:
                    return result
            # R_object is a data.frame
            if isinstance(format, dict):
                format = format['data.frame']
            # If index is None, default to what's in the config
            if index is None:
                index = False if format == 'numpy' else _config['index']
            elif format == 'numpy':
                error_message = "index must be None when format='numpy'"
                raise ValueError(error_message)
            if index:
                if index in result:
                    error_message = (
                        f'the converted data frame already contains a column '
                        f'called {index!r}, which conflicts with the name '
                        f'that was going to be used for the index column.\n'
                        f'Set index to a different string, or set index=False '
                        f'to not convert the index.')
                    raise ValueError(error_message)
                rownames = _rlib.Rf_getAttrib(R_object, _rlib.R_RowNamesSymbol)
                if rownames != _rlib.R_NilValue:
                    result = {index: to_py((
                        f'rownames({R_statement})', rownames), format=format,
                        index=False, squeeze=False).rename(index)} | result
                else:
                    # The data frame doesn't have rownames, even though we
                    # specified that we wanted to include rownames
                    index = False
            if format == 'polars':
                try:
                    with ignore_sigint():
                        import polars as pl
                except ImportError as e:
                    error_message = (
                        "polars is not installed; consider setting "
                        "format='numpy', format='pandas', or "
                        "format='pandas-pyarrow' in to_py(), or call e.g. "
                        "options(to_py_format='pandas') to change the default "
                        "format")
                    raise ImportError(error_message) from e
                return pl.DataFrame(result)
            elif format == 'numpy':
                if result:
                    return np.stack(tuple(result.values()), axis=1)
                else:
                    # There's no data in the data.frame, so the dtype
                    # is undefined; just return an object-dtyped array
                    # of the same dimensions as the data.frame
                    function_call = rmemory.protect(_rlib.Rf_lang2(
                        _rlib.Rf_install(b'dim'), R_object))
                    dim = _call(function_call, rmemory,
                                'cannot get dimensions of data.frame')
                    return np.empty((_rlib.INTEGER_ELT(dim, 0),
                                     _rlib.INTEGER_ELT(dim, 1)), dtype=object)
            else:
                with ignore_sigint():
                    import pandas as pd
                if index:
                    index_object = pd.Index(result[index])
                    del result[index]
                    return pd.DataFrame(result).set_axis(index_object)
                else:
                    return pd.DataFrame(result)
        elif R_object_type == _rlib.RAWSXP:
            # raw vector (i.e. scalar bytearray)
            if raise_format_errors:
                error_message = (
                    f'{R_statement!r} is a raw vector, which always converts '
                    f'to a Python scalar (specifically, a bytearray), so '
                    f'format must be None')
                raise TypeError(error_message)
            if raise_squeeze_errors:
                error_message = (
                    f'{R_statement!r} is a raw vector, which always converts '
                    f'to a Python scalar (specifically, a bytearray), so '
                    f'squeeze must be None')
                raise TypeError(error_message)
            return bytearray(_ffi.buffer(_rlib.RAW(R_object),
                                         _rlib.Rf_xlength(R_object)))
        elif R_object_type == _rlib.S4SXP:
            # S4 object
            # Recursively convert all the variables in the S4 object to a dict.
            # Ignore variables that can't be converted, like functions. Handle
            # sparse matrices (which are S4 objects) separately.
            if raise_squeeze_errors:
                error_message = \
                    f'{R_statement!r} is an S4 object, so squeeze must be None'
                raise TypeError(error_message)
            classes = _rlib.Rf_getAttrib(R_object, _rlib.R_ClassSymbol)
            classes = {_ffi.string(_rlib.R_CHAR(_rlib.STRING_ELT(
                classes, i))) for i in range(_rlib.Rf_xlength(classes))}
            sparse_matrix_class = classes & _sparse_matrix_classes
            if sparse_matrix_class:
                sparse_matrix_class = sparse_matrix_class.pop()
                from scipy.sparse import csr_array, csc_array, coo_array
                if sparse_matrix_class[0] == ord(b'n'):
                    function_call = rmemory.protect(
                        _rlib.Rf_lang2(rmemory.protect(
                            _rlib.Rf_lang3(_rlib.R_DoubleColonSymbol,
                                           _rlib.Rf_install(b'Matrix'),
                                           _rlib.Rf_install(b'nnzero'))),
                            R_object))
                    num_non_zero = _call(function_call, rmemory,
                                         'cannot get the number of non-zero '
                                         'sparse matrix elements')
                    num_non_zero = _rlib.INTEGER(num_non_zero)[0]
                    data = np.ones(num_non_zero, dtype=bool)
                elif sparse_matrix_class[0] == ord(b'l'):
                    data = np.asarray(to_py(
                        (f'({R_statement})@x',
                         _rlib.R_do_slot(R_object, _rlib.Rf_install(b'x'))),
                        format='numpy', squeeze=False), dtype=bool)
                else:
                    data = to_py((f'({R_statement})@x',
                                  _rlib.R_do_slot(R_object,
                                                  _rlib.Rf_install(b'x'))),
                                 format='numpy', squeeze=False)
                dim = _rlib.R_do_slot(R_object, _rlib.Rf_install(b'Dim'))
                shape = tuple(_rlib.INTEGER_ELT(dim, i)
                              for i in range(_rlib.Rf_xlength(dim)))
                if sparse_matrix_class in \
                        (b'dgRMatrix', b'lgRMatrix', b'ngRMatrix'):
                    return csr_array((
                        data,
                        to_py((f'({R_statement})@j', _rlib.R_do_slot(
                            R_object, _rlib.Rf_install(b'j'))),
                            format='numpy', squeeze=False),
                        to_py((f'({R_statement})@p', _rlib.R_do_slot(
                            R_object, _rlib.Rf_install(b'p'))),
                              format='numpy', squeeze=False)),
                        shape=shape)
                elif sparse_matrix_class in (
                        b'dgCMatrix', b'lgCMatrix', b'ngCMatrix'):
                    return csc_array((
                        data,
                        to_py((f'({R_statement})@i', _rlib.R_do_slot(
                            R_object, _rlib.Rf_install(b'i'))),
                              format='numpy', squeeze=False),
                        to_py((f'({R_statement})@p', _rlib.R_do_slot(
                            R_object, _rlib.Rf_install(b'p'))),
                              format='numpy', squeeze=False)),
                        shape=shape)
                else:
                    return coo_array((data, (
                        to_py((f'({R_statement})@i', _rlib.R_do_slot(
                            R_object, _rlib.Rf_install(b'i'))),
                              format='numpy', squeeze=False),
                        to_py((f'({R_statement})@j', _rlib.R_do_slot(
                            R_object, _rlib.Rf_install(b'j'))),
                              format='numpy', squeeze=False))), shape=shape)
            # Get the slot names
            function_call = rmemory.protect(_rlib.Rf_lang2(
                _rlib.Rf_findFun(_rlib.Rf_install(b'slotNames'),
                                 _rlib.R_GlobalEnv),
                R_object))
            slot_names = _call(function_call, rmemory,
                               f'cannot convert {R_statement!r} to an arrow '
                               f'Array')
            # Recurse over them
            result = {}
            for i in range(_rlib.Rf_xlength(slot_names)):
                slot_name = _rlib.R_CHAR(_rlib.STRING_ELT(slot_names, i))
                variable = \
                    _rlib.R_do_slot(R_object, _rlib.Rf_install(slot_name))
                slot_name_str = _ffi.string(slot_name).decode('utf-8')
                try:
                    converted_variable = to_py(
                        (f'({R_statement})@{slot_name_str}', variable),
                        format=format, index=index, squeeze=squeeze)
                except RecursionError as e:
                    error_message = (
                        f'maximum recursion depth exceeded when converting '
                        f'the S4 object {R_statement!r} to Python; this '
                        f'usually means the object contains a circular '
                        f'reference')
                    raise ValueError(error_message) from e
                except TypeError as e:
                    if 'cannot be converted to Python' in str(e):
                        continue
                    else:
                        raise
                result[slot_name_str] = converted_variable
            return result
        else:
            if R_object_type == _rlib.LANGSXP and \
                    _rlib.CAR(R_object) == _rlib.Rf_install(b'~'):
                type_description = 'formula'
            else:
                # Note: PROMSXP includes builtins like t() and df()
                _unsupported_type_descriptions = {
                    _rlib.LISTSXP: 'pairlist',
                    _rlib.CLOSXP: 'function',
                    _rlib.PROMSXP: 'function',
                    _rlib.LANGSXP: 'language construct',
                    _rlib.SPECIALSXP: 'function',
                    _rlib.BUILTINSXP: 'function',
                    _rlib.CHARSXP:
                        'internal-only "scalar" string (CHARSXP)',
                    _rlib.DOTSXP:  'dot-dot-dot object',
                    _rlib.ANYSXP: '"any" object',
                    _rlib.EXPRSXP: 'expression',
                    _rlib.BCODESXP: 'byte code object',
                    _rlib.EXTPTRSXP: 'external pointer',
                    _rlib.WEAKREFSXP: 'weak reference',
                    _rlib.NEWSXP: 'NEWSXP',
                    _rlib.FREESXP: 'FREESXP',
                    _rlib.FUNSXP: 'function'}
                type_description = \
                    _unsupported_type_descriptions[R_object_type]
            error_message = (
                f'{R_statement!r} is an R {type_description} and cannot be '
                f'converted to Python')
            raise TypeError(error_message)
        # At this point, R_object is a vector, matrix or array
        # Get the dimensions (non-NULL if the vector is actually a
        # matrix/array); handle 1D arrays as if they were vectors
        dim = _rlib.Rf_getAttrib(R_object, _rlib.R_DimSymbol)
        shape = tuple(_rlib.INTEGER_ELT(dim, i)
                      for i in range(_rlib.Rf_xlength(dim)))
        multidimensional = dim != _rlib.R_NilValue and len(shape) >= 2
        if multidimensional:
            input_type = 'matrix' if len(shape) == 2 else 'array'
        else:
            input_type = 'vector'
        # Infer format based on the input_type
        if input_type == 'array':
            if raise_format_errors:
                error_message = (
                    f'{R_statement!r} is a {len(shape):,}D array, which can '
                    f'only be converted into a NumPy array, so format must be '
                    f'None')
                raise TypeError(error_message)
            if index is not None:
                error_message = (
                    f'{R_statement!r} is a {len(shape):,}D array, which can '
                    f'only be converted into a NumPy array, so index must be '
                    f'None')
                raise TypeError(error_message)
            format = 'numpy'
            index = None
        elif isinstance(format, dict):
            format = format[input_type]
        if R_object_type != _rlib.CPLXSXP:
            # Convert vectors and DataFrames to Python via Arrow:
            # arrow.apache.org/docs/dev/python/integration/python_r.html.
            # This code should be updated to use the capsule interface (arrow.
            # apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
            # once its docs are updated (github.com/apache/arrow/issues/39198).
            # See github.com/rpy2/rpy2-arrow/blob/main/rpy2_arrow/arrow.py for
            # a reference implementation.
            if R_arrow_array is None:
                arrow_namespace = rmemory.protect(
                    _rlib.R_FindNamespace(_bytestring_to_character_vector(
                        b'arrow', rmemory)))
                if R_object_type == _rlib.REALSXP and \
                        _rlib.Rf_inherits(R_object, b'integer64'):
                    # Use arrow::as_arrow_array(R_object, type=arrow::int64())
                    # for bit64::integer64 arrays. Arrow is not fooled by our
                    # hack in to_r() of reinterpreting int64/uint64 arrays as
                    # double and then adding 'integer64' as a class in to_r(),
                    # and will convert these arrays to a DoubleArray with
                    # arrow::Array$create().
                    as_arrow_array = _rlib.Rf_findVarInFrame(
                        arrow_namespace, _rlib.Rf_install(b'as_arrow_array'))
                    function_call = rmemory.protect(_rlib.Rf_lang1(
                        _rlib.Rf_findVarInFrame(
                            arrow_namespace, _rlib.Rf_install(b'int64'))))
                    int64 = _call(function_call, rmemory,
                                  'cannot get arrow::int64')
                    function_call = rmemory.protect(_rlib.Rf_lang3(
                        as_arrow_array, R_object, int64))
                    _rlib.SET_TAG(_rlib.CDDR(function_call),
                                  _rlib.Rf_install(b'type'))
                    R_arrow_array = _call(function_call, rmemory,
                                          f'cannot convert {R_statement!r} to '
                                          f'an arrow int64 Array')
                elif R_object_type == _rlib.REALSXP and \
                        _rlib.Rf_inherits(R_object, b'difftime'):
                    # Use arrow::as_arrow_array(R_object, type=arrow::duration(
                    # 'ns')) for timedeltas (i.e. difftimes without the hms
                    # class) and arrow::as_arrow_array(R_object,
                    # type=arrow::time64('ns')) for times (which have both the
                    # difftime and hms classes) to work around
                    # github.com/apache/arrow/issues/40109.
                    as_arrow_array = _rlib.Rf_findVarInFrame(
                        arrow_namespace, _rlib.Rf_install(b'as_arrow_array'))
                    hms = _rlib.Rf_inherits(R_object, b'hms')
                    duration_or_time64_name = b'time64' if hms else b'duration'
                    duration_or_time64 = _rlib.Rf_findVarInFrame(
                        arrow_namespace,
                        _rlib.Rf_install(duration_or_time64_name))
                    # Use 'us' for times when format is 'numpy' or 'pandas'
                    # because those store times as an object-dtyped array of
                    # datetime.time, which can't represent ns
                    time_unit = 'us' if hms and (
                        format == 'pandas' or format == 'numpy') else 'ns'
                    function_call = rmemory.protect(_rlib.Rf_lang2(
                        duration_or_time64,
                        _string_to_character_vector(time_unit, rmemory)))
                    duration = _call(
                        function_call, rmemory,
                        f'cannot get arrow::'
                        f'{duration_or_time64_name.decode("utf-8")}'
                        f'({time_unit!r})')
                    function_call = rmemory.protect(_rlib.Rf_lang3(
                        as_arrow_array, R_object, duration))
                    _rlib.SET_TAG(_rlib.CDDR(function_call),
                                  _rlib.Rf_install(b'type'))
                    R_arrow_array = _call(function_call, rmemory,
                                          f'cannot convert {R_statement!r} to '
                                          f'an arrow duration[{time_unit}] '
                                          f'Array')
                else:
                    # Convert the vector to an Arrow array via
                    # arrow::Array$create()
                    Array = _rlib.Rf_eval(_rlib.Rf_findVarInFrame(
                        arrow_namespace, _rlib.Rf_install(b'Array')),
                        _rlib.R_BaseEnv)
                    function_call = rmemory.protect(_rlib.Rf_lang2(
                        _rlib.Rf_install(b'create'), R_object))
                    R_arrow_array = _call(function_call, rmemory,
                                          f'cannot convert {R_statement!r} to '
                                          f'an arrow Array', environment=Array)
            # Export the Arrow array from R to C via the array$export_to_c
            # method. Note that R6 objects (like R_arrow_array) are just
            # environments, so the method call is identical to a function call,
            # but with the environment set to R_arrow_array, not R_BaseEnv.
            array = pyarrow_ffi.new('struct ArrowArray*')
            schema = pyarrow_ffi.new('struct ArrowSchema*')
            array_ptr = int(pyarrow_ffi.cast('uintptr_t', array))
            schema_ptr = int(pyarrow_ffi.cast('uintptr_t', schema))
            array_ptr_STRSXP = \
                _string_to_character_vector(str(array_ptr), rmemory)
            schema_ptr_STRSXP = \
                _string_to_character_vector(str(schema_ptr), rmemory)
            method_call = rmemory.protect(_rlib.Rf_lang3(_rlib.Rf_install(
                b'export_to_c'), array_ptr_STRSXP, schema_ptr_STRSXP))
            _call(method_call, rmemory,
                  f'cannot export {R_statement!r} to C as an Arrow array',
                  environment=R_arrow_array)
            # Import the Arrow array from C into Python
            result = pa.lib.Array._import_from_c(array_ptr, schema_ptr)
            # If POSIXct and attr(R_object, "tz") is NULL, Arrow erroneously
            # uses the user's time zone; remove it now that we've converted to
            # Python
            if _rlib.Rf_inherits(R_object, b'POSIXct'):
                function_call = rmemory.protect(_rlib.Rf_lang3(
                    _rlib.Rf_install(b'attr'),
                    R_object,
                    _bytestring_to_character_vector(b'tz', rmemory)))
                tz = _call(function_call, rmemory,
                           'cannot get attr(R_object, "tz")')
                if tz == _rlib.R_NilValue:
                    result = result.cast(pa.timestamp(result.type.unit))
        else:  # complex vector/matrix/array
            # If R_object is a scalar and squeeze=True, return a Python scalar
            if squeeze and _rlib.Rf_xlength(R_object) == 1:
                if raise_format_errors:
                    error_message = (
                        f'{R_statement!r} is a scalar and squeeze=True, so '
                        f'format must be None')
                    raise TypeError(error_message)
                if index is not None:
                    error_message = (
                        f'{R_statement!r} is a scalar and squeeze=True, so '
                        f'index must be None')
                    raise TypeError(error_message)
                first_element = _rlib.COMPLEX_ELT(R_object, 0)
                return complex(first_element.r, first_element.i)
            # NumPy and pandas have a complex dtype with the same memory layout
            # (a pair of float64s), so just reinterpret the memory. polars
            # doesn't, so leave it up to the user what to do.
            if format == 'polars':
                error_message = (
                    f"{R_statement!r} is a complex {input_type}, which is not "
                    f"supported by polars.\nConsider either switching to "
                    f"format='numpy', format='pandas' or "
                    f"format='pandas-pyarrow', or splitting the {input_type} "
                    f"into its real and imaginary parts with "
                    f"Re({R_statement}) and Im({R_statement}) and converting "
                    f"them separately.")
                raise TypeError(error_message)
            # Need to copy() to avoid sharing memory with R, which is dangerous
            # without Arrow's mechanism for lifetime management
            result = np.frombuffer(_ffi.buffer(
                _rlib.COMPLEX(R_object),
                _rlib.Rf_xlength(R_object) * _ffi.sizeof('Rcomplex')),
                dtype=np.complex128).copy()
            if multidimensional:
                result = result.reshape(shape, order='F')
            if format != 'numpy':
                with ignore_sigint():
                    import pandas as pd
                result = pd.DataFrame(result) if multidimensional else \
                    pd.Series(result)
        # If R_object is a scalar and squeeze=True, return a Python scalar
        if squeeze and _rlib.Rf_xlength(R_object) == 1:
            if raise_format_errors:
                error_message = (
                    f'{R_statement!r} is a scalar and squeeze=True, so format '
                    f'must be None')
                raise TypeError(error_message)
            if index is not None:
                error_message = (
                    f'{R_statement!r} is a scalar and squeeze=True, so index '
                    f'must be None')
                raise TypeError(error_message)
            result = result[0]
            # pyarrow converts DurationScalar[ns] to pd.Timedelta rather than
            # datetime.timedelta like for DurationScalar[us] and above, because
            # datetime.timedelta can't represent ns. Instead, just truncate to
            # the nearest microsecond, then convert to datetime.timedelta.
            if isinstance(result, pa.DurationScalar):
                result = result.cast(pa.duration('us'), safe=False)
            result = result.as_py()
            return result
        # If index is None, default to what's in the config (or False for
        # NumPy)
        if index is None:
            index = False if format == 'numpy' else _config['index']
        elif format == 'numpy':
            error_message = "index must be None when format='numpy'"
            raise ValueError(error_message)
        # Convert to the desired output format, adding names/rownames if
        # present. If R_object is multidimensional (>= 2D, i.e. an R array
        # rather than a vector), also reshape.
        # If R_object is a matrix (== 2D), also add colnames if present.
        if index or input_type == 'matrix' and format != 'numpy':
            # index=False for arrays, so R_object is a vector or matrix here
            if input_type == 'vector':
                rownames = _rlib.Rf_getAttrib(R_object, _rlib.R_NamesSymbol)
            else:  # input_type == 'matrix'
                dimnames = _rlib.Rf_getAttrib(R_object, _rlib.R_DimNamesSymbol)
                if dimnames != _rlib.R_NilValue:
                    rownames = _rlib.VECTOR_ELT(dimnames, 0)
                    colnames = _rlib.VECTOR_ELT(dimnames, 1)
                else:
                    rownames = colnames = _rlib.R_NilValue
            if rownames != _rlib.R_NilValue:
                rownames_name = \
                    'names' if input_type == 'vector' else 'rownames'
                rownames = to_py((f'{rownames_name}({R_statement})', rownames),
                                 format=format, index=False, squeeze=False)
            else:
                index = False
        if format == 'numpy':
            if R_object_type != _rlib.CPLXSXP:
                if result.null_count == 0 and (
                        pa.types.is_integer(result.type) or
                        pa.types.is_floating(result.type)):
                    # Create a writable NumPy array from the Arrow buffer
                    # without copying (hacky)
                    result = np.frombuffer(result.buffers()[-1],
                                           dtype=result.type.to_pandas_dtype())
                else:
                    # Copy to a NumPy array
                    result = result.to_numpy(zero_copy_only=False,
                                             writable=True)
                # If R_object is a matrix, reshape only once result is a NumPy
                # array since Arrow doesn't have a notion of memory-contiguous
                # 2D arrays
                if multidimensional:
                    result = result.reshape(shape, order='F')
        else:
            if multidimensional:
                # R_object is specifically a matrix here (i.e. 2D), not a >2D
                # array, since we set format='numpy' for >2D arrays above
                nrows, ncols = shape
                if colnames != _rlib.R_NilValue:
                    colnames = to_py((f'colnames({R_statement})', colnames),
                                     format=format, index=False, squeeze=False)
                else:
                    colnames = [f'column_{i}' for i in range(ncols)]
                if index and index in colnames:
                    error_message = (
                        f'the converted matrix already contains a column '
                        f'called {index!r}, which conflicts with the name '
                        f'that was going to be used for the index.\nSet index '
                        f'to a different string, or set index=False to not '
                        f'convert the index.')
                    raise ValueError(error_message)
                if R_object_type != _rlib.CPLXSXP:
                    result = pa.RecordBatch.from_arrays([
                        result[i * nrows: (i + 1) * nrows]
                        for i in range(ncols)], names=colnames)
                else:  # format='pandas' or format='pandas-pyarrow'
                    result.columns = pd.Index(colnames).rename(None)
            else:
                vector_name = re.split(r'[@$]', R_statement)[-1]
                if index == vector_name:
                    error_message = (
                        f'the converted vector {R_statement!r} has the same '
                        f'name as the index.\nSet index to a different '
                        f'string, or set index=False to not convert the '
                        f'index.')
                    raise ValueError(error_message)
            if format == 'polars':
                try:
                    with ignore_sigint():
                        import polars as pl
                except ImportError as e:
                    error_message = (
                        "polars is not installed; consider setting "
                        "format='numpy', format='pandas', or "
                        "format='pandas-pyarrow' in to_py(), or call e.g. "
                        "options(to_py_format='pandas') to change the default "
                        "format")
                    raise ImportError(error_message) from e
                if multidimensional and len(result) == 0:
                    # workaround for github.com/pola-rs/polars/issues/14659
                    result = pl.from_arrow(pa.Table.from_batches([result]))
                else:
                    result = pl.from_arrow(result)
                # Workaround for github.com/pola-rs/polars/issues/14709: if
                # R_object is an ordered factor or Arrow DictionaryArray,
                # convert Categorical to Enum, using the ordering from
                # levels(R_object)
                if (result.shape[1] > 0 and result.dtypes[0] == pl.Categorical
                    if multidimensional else result.dtype == pl.Categorical) \
                        and _rlib.LOGICAL_ELT(
                            _rlib.Rf_findVarInFrame(_rlib.Rf_findVarInFrame(
                                R_arrow_array, _rlib.Rf_install(b'type')),
                                _rlib.Rf_install(b'ordered')), 0):
                    function_call = rmemory.protect(
                        _rlib.Rf_lang2(_rlib.R_LevelsSymbol, R_object))
                    levels = _call(function_call, rmemory,
                                   f'cannot retrieve levels of '
                                   f'{R_statement!r}')
                    levels = to_py((f'levels({R_statement})', levels),
                                   format=format, index=False, squeeze=False)
                    result = result.cast(pl.Enum(levels))
                if index:
                    if not multidimensional:
                        result = result.to_frame(name=vector_name)
                    if result.width:
                        result.insert_column(0, pl.Series(name=index,
                                                          values=rownames))
                    else:
                        # polars forces all 0-column DataFrames to have length
                        # 0, so manually make a DataFrame of the correct length
                        result = pl.DataFrame({index: rownames})
            else:  # output_format in ('pandas', 'pandas-pyarrow')
                with ignore_sigint():
                    import pandas as pd
                if R_object_type != _rlib.CPLXSXP:
                    result = result.to_pandas(
                        date_as_object=False, split_blocks=True,
                        self_destruct=True, types_mapper=pd.ArrowDtype
                        if format == 'pandas-pyarrow' else None)
                if index:
                    result.index = pd.Index(rownames, name=index)
            if not index and not multidimensional:
                result = result.rename(vector_name)
        return result


def options(*,
            to_r_format: Literal['keep', 'matrix', 'data.frame'] | None = None,
            to_py_format: Literal['polars', 'pandas', 'pandas-pyarrow',
                                  'numpy'] |
                          dict[Literal['vector', 'matrix', 'data.frame'],
                               Literal['polars', 'pandas', 'pandas-pyarrow',
                               'numpy']] | None = None,
            index: str | Literal[False] | None = None,
            squeeze: bool | None = None,
            plot_width: int | float | None = None,
            plot_height: int | float | None = None) -> \
        dict[str, dict[str, str] | str | bool | int] | None:
    """
    Get or set ryp's configuration settings.
    
    options() with no arguments gets the current value of each option, while
    e.g. options(to_py_format='pandas') sets pandas as the default format in
    to_py() and leaves the other options unchanged.
    
    Args:
        to_r_format: the default value for the format parameter in to_r();
                     must be 'keep' (the default), 'matrix', or 'data.frame'
        to_py_format: the default value for the format parameter in to_py();
                      must be 'polars' (the default), 'pandas',
                      'pandas-pyarrow', 'numpy', or a dictionary with one of
                      those four Python formats and/or None as values and
                      'vector', 'matrix' and/or 'data.frame' as keys (if
                      certain keys are missing or have None as the format,
                      leave their format unchanged)
        index: the default value for the index parameter in to_py();
               must be a string (default: 'index') or False
        squeeze: the default value for the squeeze parameter in to_py();
                 must be True (the default) or False
        plot_width: the width, in inches, of inline plots in Jupyter notebooks;
                    must be a positive number. Defaults to 6.4 inches, to match
                    Matplotlib's default.
        plot_height: the height, in inches, of inline plots in Jupyter
                     notebooks; must be a positive number. Defaults to 4.8
                     inches, to match Matplotlib's default.
    
    Returns:
        A dictionary of the current values for each option if calling options()
        with no arguments, or None otherwise.
    """
    no_config_set = True
    if to_r_format is not None:
        if to_r_format not in ('matrix', 'data.frame', 'keep'):
            if isinstance(to_r_format, str):
                error_message = \
                    "to_r_format must be 'matrix', 'data.frame', or 'keep'"
                raise ValueError(error_message)
            else:
                error_message = (
                    f"to_r_format has type {type(to_r_format).__name__!r}, "
                    f"but must be 'matrix', 'data.frame', or 'keep'")
                raise TypeError(error_message)
        _config['to_r_format'] = to_r_format
        no_config_set = False
    if to_py_format is not None:
        _check_to_py_format(to_py_format, variable_name='to_py_format')
        if isinstance(to_py_format, dict):
            for key in _config['to_py_format']:
                key_to_py_format = to_py_format.get(key, None)
                if key_to_py_format is not None:
                    _config['to_py_format'][key] = key_to_py_format
        else:
            for key in _config['to_py_format']:
                _config['to_py_format'][key] = to_py_format
        no_config_set = False
    if index is not None:
        if index is not False and not isinstance(index, str):
            if index is True:
                error_message = 'index is True, but must be False or a string'
                raise ValueError(error_message)
            else:
                error_message = (
                    f'index has type {type(index).__name__!r}, but must be '
                    f'False or a string')
                raise TypeError(error_message)
        _config['index'] = index
        no_config_set = False
    if squeeze is not None:
        if not isinstance(squeeze, bool):
            error_message = (
                f'squeeze has type {type(squeeze).__name__!r}, but must be '
                f'boolean')
            raise TypeError(error_message)
        _config['squeeze'] = squeeze
        no_config_set = False
    if plot_width is not None:
        if not isinstance(plot_width, (int, float)):
            error_message = (
                f'plot_width has type {type(plot_width).__name__!r}, but must '
                f'be int or float')
            raise TypeError(error_message)
        if plot_width <= 0:
            error_message = f'plot_width is {plot_width}, but must be positive'
            raise ValueError(error_message)
        _config['plot_width'] = plot_width
        no_config_set = False
    if plot_height is not None:
        if not isinstance(plot_height, (int, float)):
            error_message = (
                f'plot_height has type {type(plot_height).__name__!r}, but '
                f'must be int or float')
            raise TypeError(error_message)
        if plot_height <= 0:
            error_message = \
                f'plot_height is {plot_height}, but must be positive'
            raise ValueError(error_message)
        _config['plot_height'] = plot_height
        no_config_set = False
    if plot_width is not None or plot_height is not None:
        r(f'options(device=function() {{ svglite(.tempfile, '
          f'width={_config["plot_width"]}, '
          f'height={_config["plot_height"]})}})')
    if no_config_set:
        return _config


# Set global variables

_config = {'to_r_format': 'keep',
           'to_py_format': {'vector': 'polars', 'matrix': 'polars',
                            'data.frame': 'polars'},
           'index': 'index', 'squeeze': True,
           'plot_width': 6.4, 'plot_height': 4.8}
_EOF_instructions = 'Ctrl + D' if platform.system() != 'Windows' else \
    'Ctrl + Z followed by Enter'
_R_keywords = {'if', 'else', 'repeat', 'while', 'function', 'for', 'in',
               'next', 'break', 'TRUE', 'FALSE', 'NULL', 'Inf', 'NaN', 'NA',
               'NA_integer_', 'NA_real_', 'NA_complex_', 'NA_character_',
               '...'}
_sparse_matrix_classes = {b'dgRMatrix', b'dgCMatrix', b'dgTMatrix',
                          b'ngRMatrix', b'ngCMatrix', b'ngTMatrix',
                          b'lgRMatrix', b'lgCMatrix', b'lgTMatrix'}
_init_lock = threading.Lock()
_ffi = None
_rlib = None
_ryp_PID = None
_ryp_thread = None
_jupyter_notebook = None
_graphapp = None
_plot_event_thread = None
