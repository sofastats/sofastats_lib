from dataclasses import dataclass
from enum import StrEnum
import os
from pathlib import Path
import platform
from subprocess import Popen, PIPE
from typing import Any, Literal, TypeAlias

MAX_CHI_SQUARE_CELLS = 200  ## was 25
MAX_CHI_SQUARE_VALS_IN_DIM = 30  ## was 6
MIN_CHI_SQUARE_VALS_IN_DIM = 2
MAX_RANK_DATA_VALS = 50_000
MAX_VALUE_LENGTH_IN_SQL_CLAUSE = 90
MIN_VALS_FOR_NORMALITY_TEST = 20
N_WHERE_NORMALITY_USUALLY_FAILS_NO_MATTER_WHAT = 100

AVG_LINE_HEIGHT_PIXELS = 12

AVG_CHAR_WIDTH_PIXELS = 20
BOX_PLOT_AVG_CHAR_WIDTH_PIXELS = 9
HISTO_AVG_CHAR_WIDTH_PIXELS = 10.5

DOJO_Y_AXIS_TITLE_OFFSET_PIXELS = 25
TEXT_WIDTH_N_CHARACTERS_WHEN_ROTATED = 1
GAP_BEFORE_FIRST_X_LABEL_TICK_PIXELS = 180
PADDING_TO_RIGHT_OF_Y_AXIS_VALUES_PIXELS = 20

JS_BOOL = Literal['true', 'false']

DOJO_COLORS = ['indigo', 'gold', 'hotpink', 'firebrick', 'indianred',
    'mistyrose', 'darkolivegreen', 'darkseagreen', 'slategrey', 'tomato',
    'lightcoral', 'orangered', 'navajowhite', 'slategray', 'palegreen',
    'darkslategrey', 'greenyellow', 'burlywood', 'seashell',
    'mediumspringgreen', 'mediumorchid', 'papayawhip', 'blanchedalmond',
    'chartreuse', 'dimgray', 'lemonchiffon', 'peachpuff', 'springgreen',
    'aquamarine', 'orange', 'lightsalmon', 'darkslategray', 'brown', 'ivory',
    'dodgerblue', 'peru', 'lawngreen', 'chocolate', 'crimson', 'forestgreen',
    'darkgrey', 'lightseagreen', 'cyan', 'mintcream', 'transparent',
    'antiquewhite', 'skyblue', 'sienna', 'darkturquoise', 'goldenrod',
    'darkgreen', 'floralwhite', 'darkviolet', 'darkgray', 'moccasin',
    'saddlebrown', 'grey', 'darkslateblue', 'lightskyblue', 'lightpink',
    'mediumvioletred', 'deeppink', 'limegreen', 'darkmagenta', 'palegoldenrod',
    'plum', 'turquoise', 'lightgoldenrodyellow', 'darkgoldenrod', 'lavender',
    'slateblue', 'yellowgreen', 'sandybrown', 'thistle', 'violet', 'magenta',
    'dimgrey', 'tan', 'rosybrown', 'olivedrab', 'pink', 'lightblue',
    'ghostwhite', 'honeydew', 'cornflowerblue', 'linen', 'darkblue',
    'powderblue', 'seagreen', 'darkkhaki', 'snow', 'mediumblue', 'royalblue',
    'lightcyan', 'mediumpurple', 'midnightblue', 'cornsilk', 'paleturquoise',
    'bisque', 'darkcyan', 'khaki', 'wheat', 'darkorchid', 'deepskyblue',
    'salmon', 'darkred', 'steelblue', 'palevioletred', 'lightslategray',
    'aliceblue', 'lightslategrey', 'lightgreen', 'orchid', 'gainsboro',
    'mediumseagreen', 'lightgray', 'mediumturquoise', 'cadetblue',
    'lightyellow', 'lavenderblush', 'coral', 'lightgrey', 'whitesmoke',
    'mediumslateblue', 'darkorange', 'mediumaquamarine', 'darksalmon', 'beige',
    'blueviolet', 'azure', 'lightsteelblue', 'oldlace']

class Platform(StrEnum):
    LINUX = 'linux'
    WINDOWS = 'windows'
    MAC = 'mac'

PLATFORMS = {'Linux': Platform.LINUX, 'Windows': Platform.WINDOWS, 'Darwin': Platform.MAC}
PLATFORM = PLATFORMS.get(platform.system())

def get_local_folder(my_platform: Platform) -> Path:
    home_path = Path(os.path.expanduser('~'))
    if my_platform == Platform.LINUX:  ## see https://bugs.launchpad.net/sofastatistics/+bug/952077
        try:
            user_path = Path(str(Popen(['xdg-user-dir', 'DOCUMENTS'],
                stdout=PIPE).communicate()[0], encoding='utf-8').strip())  ## get output i.e. [0]. err is 2nd.
        except OSError:
            user_path = home_path
    else:
        user_path = home_path
    local_path = user_path / 'sofastats'
    return local_path

uv_run_mode = 'UV' in os.environ
local_folder = get_local_folder(PLATFORM)

if uv_run_mode:
    ## If running in uv run single script mode everything should just occur in the same folder as that the script being run is located in
    current_path = Path.cwd()
    INTERNAL_DATABASE_FPATH = current_path / 'sofastats.db'
    CUSTOM_STYLES_FOLDER = current_path
    CUSTOM_DBS_FOLDER = current_path
else:
    local_folder.mkdir(exist_ok=True)
    internal_folder = local_folder / '_internal'
    internal_folder.mkdir(exist_ok=True)
    INTERNAL_DATABASE_FPATH = internal_folder / 'sofastats.db'
    CUSTOM_STYLES_FOLDER = local_folder / 'custom_styles'
    CUSTOM_STYLES_FOLDER.mkdir(exist_ok=True)
    CUSTOM_DBS_FOLDER = local_folder / 'custom_databases'
    CUSTOM_DBS_FOLDER.mkdir(exist_ok=True)
## even a uvx sofastats user wants anything they produce to get put somewhere sensible
DEFAULT_OUTPUT_FOLDER = local_folder / 'output'
# DEFAULT_OUTPUT_FOLDER.mkdir(exist_ok=True)  ## make on demand only. Reason? So not messing with user's file system when using uv run - just puts back in same folder which is simple

class DbeName(StrEnum):  ## database engine
    SQLITE = 'sqlite'

@dataclass(frozen=True)
class DbeSpec:
    """
    entity: e.g. table name 'demo_tbl'
    string value: e.g. 'New Zealand'
    """
    dbe_name: str
    if_clause: str
    placeholder: str
    left_entity_quote: str  ## usually left and right are the same but in MS Access and MS SQL Server they are different: '[' and ']'
    right_entity_quote: str
    gte_not_equals: str
    cartesian_joiner: str
    str_value_quote: str
    str_value_quote_escaped: str
    summable: str

    def entity_quoter(self, entity: str) -> str:
        """
        E.g. "demo_tbl" -> "`demo_tbl`"
        or "table name with spaces" -> "`table name with spaces`"
        for use in
        SELECT * FROM `table name with spaces`
        """
        return f"{self.left_entity_quote}{entity}{self.right_entity_quote}"

    def str_value_quoter(self, str_value: str) -> str:
        """
        E.g. "New Zealand" -> "'New Zealand'"
        for use in
        SELECT * FROM `demo_tbl` WHERE `country` = 'New Zealand'
        """
        return f"{self.str_value_quote}{str_value}{self.str_value_quote}"

SortOrderSpecs: TypeAlias = dict[str, list[Any]]

class SortOrder(StrEnum):
    """
    Sort orders to apply.
    Note - INCREASING & DECREASING only apply to sorting at the final values level.
    E.g. If 'Age Group' > 'Handedness' > 'Home Location Type' then only 'Home Location Type'
    can potentially have sort order by frequency
    """
    CUSTOM = 'by custom order'
    "By custom order configured in YAML or dictionary for relevant variable"
    DECREASING = 'by decreasing frequency'
    "By decreasing frequency"
    INCREASING = 'by increasing frequency'
    "By increasing frequency"
    VALUE = 'by value'
    "By value alphabetically sorted"

class ChartMetric(StrEnum):
    AVG = 'avg'
    FREQ = 'freq'
    PCT = 'pct'
    SUM = 'sum'
