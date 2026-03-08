"""
cd /home/g/projects && uv run "/home/g/projects/sofastats_lib/utils/minify_sofastats_dojo.py"
"""

import jsmin

with open("/home/g/projects/sofastats_lib/src/sofastats/output/js/sofastats_dojo.js", "r") as f:
    oldjs = f.read()
newjs = jsmin.jsmin(oldjs)
with open("/home/g/projects/sofastats_lib/src/sofastats/output/js/sofastats_dojo_minified.js", "w") as f:
    f.write(newjs)
print("Finished")
