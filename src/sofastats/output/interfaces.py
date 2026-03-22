"""
Note - output.utils.get_report() replies on the template param names here so keep aligned.
Not worth formally aligning them given how easy to do manually and how static.
"""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, fields
import datetime
from enum import StrEnum
from pathlib import Path
import sqlite3 as sqlite
from typing import Any, Protocol  #, SupportsKeysAndGetItem (from https://github.com/python/typeshed but not worth another dependency)
from webbrowser import open_new_tab
import jinja2
import pandas as pd

from sofastats import SQLITE_DB, logger
from sofastats.conf.main import INTERNAL_DATABASE_FPATH, DbeName, SortOrderSpecs
from sofastats.data_extraction.db import ExtendedCursor, get_dbe_spec
from sofastats.output.charts.conf import DOJO_CHART_JS
import sofastats.output.styles as styles
from sofastats.output.styles.utils import (get_generic_unstyled_css, get_style_spec, get_styled_dojo_chart_css,
    get_styled_placeholder_css_for_main_tbls, get_styled_stats_tbl_css)
from sofastats.utils.misc import get_safer_name

from ruamel.yaml import YAML

DEFAULT_SUPPLIED_BUT_MANDATORY_ANYWAY = '__default_supplied_but_mandatory_anyway__'  ## enforced through ...

TUNDRA_CSS = (Path(styles.__file__).parent.parent / 'css' / 'tundra.css').read_text()
DOJO_XD_JS = (Path(styles.__file__).parent.parent / 'js' / 'dojo.xd.js').read_text()
SOFASTATS_CHARTS_JS = (Path(styles.__file__).parent.parent / 'js' / 'sofastats_charts.js').read_text()
SOFASTATS_DOJO_MINIFIED_JS = (Path(styles.__file__).parent.parent / 'js' / 'sofastats_dojo_minified.js').read_text()

class OutputItemType(StrEnum):
    CHART = 'chart'
    MAIN_TABLE = 'main_table'
    STATS = 'stats'

@dataclass(frozen=True)
class HTMLItemSpec:
    html_item_str: str
    output_item_type: OutputItemType
    output_title: str | None
    design_name: str
    style_name: str  ## so we know which styles we have to cover in the overall HTML

    def to_standalone_html(self) -> str:
        """
        output.utils.get_report() also handles final HTML output
        """
        style_spec = get_style_spec(self.style_name)
        tpl_bits = [HTML_AND_SOME_HEAD_TPL, ]
        if self.output_item_type == OutputItemType.CHART:
            tpl_bits.append(CHARTING_CSS_TPL)
            tpl_bits.append(CHARTING_JS_TPL)
        if self.output_item_type == OutputItemType.MAIN_TABLE:
            tpl_bits.append(SPACEHOLDER_CSS_TPL)
        if self.output_item_type == OutputItemType.STATS:
            tpl_bits.append(STATS_TBL_TPL)
        tpl_bits.append(HEAD_END_TPL)
        tpl_bits.append(BODY_START_TPL)
        tpl_bits.append(self.html_item_str)  ## <======= the actual item content e.g. chart
        tpl_bits.append(BODY_AND_HTML_END_TPL)
        tpl = '\n'.join(tpl_bits)

        environment = jinja2.Environment()
        template = environment.from_string(tpl)
        context = {
            'dojo_xd_js': DOJO_XD_JS,
            'generic_unstyled_css': get_generic_unstyled_css(),
            'sofastats_charts_js': SOFASTATS_CHARTS_JS,
            'sofastats_dojo_minified_js': SOFASTATS_DOJO_MINIFIED_JS,
            'title': self.output_title,
            'tundra_css': TUNDRA_CSS,
        }
        if self.output_item_type == OutputItemType.CHART:
            context['styled_dojo_chart_css'] = get_styled_dojo_chart_css(style_spec.dojo)
            context['dojo_chart_js'] = DOJO_CHART_JS
        if self.output_item_type == OutputItemType.MAIN_TABLE:
            context['styled_placeholder_css_for_main_tbls'] = get_styled_placeholder_css_for_main_tbls(self.style_name)
        if self.output_item_type == OutputItemType.STATS:
            context['styled_stats_tbl_css'] = get_styled_stats_tbl_css(style_spec)
        html = template.render(context)
        return html

    def to_file(self, *, fpath: Path | str):
        with open(fpath, 'w') as f:
            f.write(self.to_standalone_html())

    def __repr_html__(self):
        return ''

class HasToHTMLItemSpec(Protocol):
    def to_html_design(self) -> HTMLItemSpec: ...


@dataclass(frozen=False)
class CommonDesign(ABC):
    r"""
    Output dataclasses (e.g. ClusteredBoxplotChartDesign) inherit from CommonDesign.
    Can't have defaults in CommonDesign attributes (which go first) and then missing defaults for the output dataclasses.
    Therefore, we are required to supply defaults for everything in the output dataclasses.
    That includes mandatory fields.
    So how do we ensure those mandatory field arguments are supplied?
    We check in __post_init__ that the special value DEFAULT_SUPPLIED_BUT_MANDATORY_ANYWAY is not supplied.

    Args:
        csv_file_path: full file path to CSV file (if using CSV as source)
        csv_separator: CSV separator (if using CSV as source)
        cur: dpapi2 cursor i.e. an object able to run cur.execute, `cur.fetchall()` etc. (if using a cursor as source)
        database_engine_name: e.g. `DbeName.SQLITE` or 'sqlite' (if using a cursor as the source)
        source_table_name: source table name (if using the cursor as a source OR using the internal SOFA SQLite database)
        table_filter_sql: valid SQL to filter the source table as supplied in source_table_name -
            must be in the appropriate SQL dialect and entities should be quoted appropriately as needed
            e.g. SQLite requires backticks for field names with spaces such as \`Age Group\`.
            You cannot filter a CSV. CSVs must already be in the form required for analysis.
        style_name: e.g. 'default'. Either one of the built-in styles under `sofastats.output.styles`
            or a custom style defined by YAML in the custom_styles subfolder of the sofastats local folder
            e.g. `~/Documents/sofastats/custom_styles`
        output_file_path: full path to folder where output HTML will be generated.
        output_title: the title the HTML output will display in a web browser
        show_in_web_browser: if `True` will open a tab in your default browser to display the output file generated
        sort_orders: if supplied, a dictionary that provides the sort orders for any variables given a custom sort order
            (`SortOrder.CUSTOM`). Multiple sort orders can be defined - with each variable given a custom sort order
            being a key in the dictionary. Example:

            ```python
            {
                Age Group: [
                    '<20',
                    '20 to <30', '30 to <40', '40 to <50',
                    '50 to <60', '60 to <70', '70 to <80',
                    '80+',
                ]
            }
            ```

            If the sort order applied was `SortOrder.VALUES`, we would see '<20' appearing as the last value
            by alphabetical order. If a custom order is defined, every value must appear in the list defining the
            desired sequence.
            Don't supply both `sort_orders` and `sort_orders_yaml_file_path`.
        sort_orders_yaml_file_path: file path containing YAML defining custom sort orders. See structure and effect as
            discussed under `sort_orders`. Don't supply both `sort_orders` and `sort_orders_yaml_file_path`.
        decimal_points: defines the maximum number of decimal points displayed.
            If set to 3, for example, 1.23456789 will be displayed as 1.235. 1.320000000 will be displayed as 1.32, and
            1.60000000 as 1.6.
    """
    ## inputs ***********************************
    csv_file_path: Path | str | None = None
    csv_separator: str = ','
    cur: Any | None = None
    database_engine_name: DbeName | str | None = None
    source_table_name: str | None = None
    table_filter_sql: str | None = None
    ## outputs **********************************
    style_name: str = 'default'
    output_file_path: Path | str | None = None
    output_title: str | None = None
    show_in_web_browser: bool = True
    sort_orders: SortOrderSpecs | None = None
    sort_orders_yaml_file_path: Path | str | None = None
    decimal_points: int = 3

    @abstractmethod
    def to_html_design(self) -> HTMLItemSpec:
        """
        From the design produce the HTML to display as one of the attributes of the HTMLItemSpec.
        Also return the style name and output item type e.g. whether a chart, table, or statistical output
        """
        pass

    def _handle_inputs(self):
        """
        There are three main paths for specific data values to be supplied to the design:

        1. CSV - data will be ingested into internal sofastats SQLite database
        (`source_table_name` optional - later analyses might be referring to that ingested table
        so nice to let user choose the name)
        2. `cur`, `database_engine_name`, and `source_table_name`
        3. or just a `source_table_name` (assumed to be using internal sofastats SQLite database)

        Any supplied cursors are "wrapped" inside an `ExtendedCursor` so we can use `.exe()` instead of `.execute()`
        and to provide better error messages on query failure.

        Client code supplies `database_engine_name` rather than dbe_spec for simplicity but internally
        `CommonDesign` supplies all code that inherits from it a `dbe_spec` attribute ready to use.

        Settings are validated e.g. to prevent client code supplying both CSV settings and database settings.
        """
        if self.csv_file_path:
            if self.cur or self.database_engine_name or self.source_table_name or self.table_filter_sql:
                raise Exception("If supplying a CSV path don't also supply database requirements")
            if not self.csv_separator:
                self.csv_separator = ','
            if not SQLITE_DB.get('sqlite_default_cur'):
                SQLITE_DB['sqlite_default_con'] = sqlite.connect(INTERNAL_DATABASE_FPATH)
                SQLITE_DB['sqlite_default_cur'] = ExtendedCursor(SQLITE_DB['sqlite_default_con'].cursor())
            self.cur = SQLITE_DB['sqlite_default_cur']
            self.dbe_spec = get_dbe_spec(DbeName.SQLITE)
            if not self.source_table_name:
                self.source_table_name = get_safer_name(Path(self.csv_file_path).stem)
            ## ingest CSV into database
            df = pd.read_csv(self.csv_file_path, sep=self.csv_separator)
            try:
                df.to_sql(self.source_table_name, SQLITE_DB['sqlite_default_con'], if_exists='replace', index=False)
            except Exception as e:  ## TODO: supply more specific exception
                logger.info(f"Failed at attempt to ingest CSV from '{self.csv_file_path}' "
                    f"into internal pysofa SQLite database as table '{self.source_table_name}'.\nError: {e}")
            else:
                logger.info(f"Successfully ingested CSV from '{self.csv_file_path}' "
                    f"into internal pysofa SQLite database as table '{self.source_table_name}'")
        elif self.cur:
            self.cur = ExtendedCursor(self.cur)
            if not self.database_engine_name:
                supported_names = '"' + '", "'.join(name.value for name in DbeName) + '"'
                raise Exception("When supplying a cursor, a database_engine_name must also be supplied. "
                    f"Supported names currently are: {supported_names}")
            else:
                self.dbe_spec = get_dbe_spec(self.database_engine_name)
            if not self.source_table_name:
                raise Exception("When supplying a cursor, a source_table_name must also be supplied")
        elif self.source_table_name:
            if not SQLITE_DB.get('sqlite_default_cur'):
                SQLITE_DB['sqlite_default_con'] = sqlite.connect(INTERNAL_DATABASE_FPATH)
                SQLITE_DB['sqlite_default_cur'] = ExtendedCursor(SQLITE_DB['sqlite_default_con'].cursor())
            self.cur = SQLITE_DB['sqlite_default_cur']  ## not already set if in the third path - will have gone down first
            if self.database_engine_name and self.database_engine_name != DbeName.SQLITE:
                raise Exception("If not supplying a csv_file_path, or a cursor, the only permitted database engine is "
                    "SQLite (the dbe of the internal sofastats SQLite database)")
            self.dbe_spec = get_dbe_spec(DbeName.SQLITE)
        else:
            raise Exception("Either supply a path to a CSV "
                "(optional tbl_name for when ingested into internal sofastats SQLite database), "
                "a cursor (with dbe_name and tbl_name), "
                "or a tbl_name (data assumed to be in internal sofastats SQLite database)")

    def _handle_outputs(self):
        """
        Validate configuration and provide sane defaults for `output_title` and `output_file_path` if nothing set.
        """
        ## output file path and title
        nice_name = '_'.join(self.__module__.split('.')[-2:]) + f"_{self.__class__.__name__}"
        if not self.output_file_path:
            now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            self.output_file_path = Path.cwd() / f"{nice_name}_{now}.html"
        if not self.output_title:
            self.output_title = f"{nice_name} Output"
        ## sort orders
        if self.sort_orders:
            if self.sort_orders_yaml_file_path:
                raise Exception("Oops - it looks like you supplied settings for both sort_orders "
                    "and sort_orders_yaml_file_path. Please set one or both of them to None.")
            else:
                pass
        elif self.sort_orders_yaml_file_path:
            yaml = YAML(typ='safe')  ## default, if not specified, is 'rt' (round-trip)
            self.sort_orders = yaml.load(Path(self.sort_orders_yaml_file_path))  ## might be a str or Path so make sure
        else:
            self.sort_orders = {}

    def __post_init__(self):
        self._handle_inputs()
        self._handle_outputs()
        for field in fields(self):
            if self.__getattribute__(field.name) == DEFAULT_SUPPLIED_BUT_MANDATORY_ANYWAY:
                ## raise a friendly error for when they didn't supply a mandatory field that technically had a default (DEFAULT_SUPPLIED_BUT_MANDATORY_ANYWAY), but we want to insist they supply a real value
                client_module = self.__module__.split('.')[-1]
                nice_name = f"{client_module}.{self.__class__.__name__}"  ## e.g. anova.AnovaDesign
                raise Exception(f"Oops - you need to supply a value for {field.name} in your {nice_name}")

    def __repr_html__(self):
        return self.__str__

    def make_output(self):
        """
        Produce HTML output, e.g. charts and numerical results, save to `output_file_path`,
        and open in web browser if `show_in_web_browser=True`.
        """
        self.to_html_design().to_file(fpath=self.output_file_path)
        if self.show_in_web_browser:
            open_new_tab(url=f"file://{self.output_file_path}")


@dataclass(frozen=True)
class ReportDesignsSpec:
    title: str
    designs: Sequence[CommonDesign]


HTML_AND_SOME_HEAD_TPL = """\
<!DOCTYPE html>
<head>
<title>{{title}}</title>
<style type="text/css">
<!--
{{tundra_css}}
{{generic_unstyled_css}}
{{gallery_css}}
-->
</style>
<script>
{{dojo_xd_js}}
</script>
<script>
{{sofastats_charts_js}}
</script>
<script>
{{sofastats_dojo_minified_js}}
</script>
"""

CHARTING_CSS_TPL = """\
<style type="text/css">
<!--
    .dojoxLegendNode {
        border: 1px solid #ccc;
        margin: 5px 10px 5px 10px;
        padding: 3px
    }
    .dojoxLegendText {
        vertical-align: text-top;
        padding-right: 10px
    }
    @media print {
        .screen-float-only{
        float: none;
        }
    }
    @media screen {
        .screen-float-only{
        float: left;
        }
    }
{{styled_dojo_chart_css}}
-->
</style>
"""

CHARTING_JS_TPL = """\
{{dojo_chart_js}}
"""

SPACEHOLDER_CSS_TPL = """\
<style type="text/css">
<!--
{{styled_placeholder_css_for_main_tbls}}
-->
</style>
"""

STATS_TBL_TPL = """\
<style type="text/css">
<!--
{{styled_stats_tbl_css}}
-->
</style>
"""

HEAD_END_TPL = "</head>"

BODY_START_TPL = "<body class='tundra'>"

BODY_AND_HTML_END_TPL = """\
</body>
</html>
"""

@dataclass(frozen=True)
class Report:
    html: str  ## include title

    def to_file(self, fpath: Path | str):
        with open(fpath, 'w') as f:
            f.write(self.html)
