#Copyright 2013-2016 The Salish Sea MEOPAR contributors
# and The University of British Columbia

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Salish Sea NEMO Jupyter Notebook collection README generator
"""
import datetime
import glob
import json
import os
import re
import sys

NBVIEWER = 'https://nbviewer.jupyter.org/github'
REPO = 'SalishSeaCast/Analysis-Aline/blob/master'
#REPO_DIR_BASE = 'notebooks'
TITLE_PATTERN = re.compile('#{1,6} ?')

def main(REPO_DIR):
    url = f"{NBVIEWER}/{REPO}/{REPO_DIR}"
    readme = """\
The links below are to static renderings of the notebooks via
[nbviewer.jupyter.org](https://nbviewer.jupyter.org/).
Descriptions under the links below are from the first cell of the notebooks
(if that cell contains Markdown or raw text).

"""
    fnlist=glob.glob('*.ipynb')
    fnlist.sort()
    if len(fnlist)>0:
        for fn in fnlist:
            readme += f"* ## [{fn}]({url}/{fn})  \n    \n"
            readme += notebook_description(fn)
    license = """
##License

These notebooks and files are copyright 2013-{this_year}
by the Salish Sea MEOPAR Project Contributors
and The University of British Columbia.

They are licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
Please see the LICENSE file for details of the license.
""".format(this_year=datetime.date.today().year)
    with open('README.md', 'wt') as f:
        f.writelines(readme)
        f.writelines(license)

def notebook_description(fn):
    description = ""
    with open(fn, "rt") as notebook:
        contents = json.load(notebook)
    try:
        first_cell = contents["worksheets"][0]["cells"][0]
    except KeyError:
        first_cell = contents["cells"][0]
    first_cell_type = first_cell["cell_type"]
    if first_cell_type not in "markdown raw".split():
        return description
    desc_lines = first_cell["source"]
    for line in desc_lines:
        suffix = ""
        if TITLE_PATTERN.match(line):
            line = TITLE_PATTERN.sub("**", line)
            suffix = "**"
        if line.endswith("\n"):
            description += f"    {line[:-1]}{suffix}\n"
        else:
            description += f"    {line}{suffix}"
    description += "\n" * 2
    return description

if __name__ == '__main__':
    startdir=os.getcwd()
    for root, dirs, files in os.walk(startdir,topdown=True):
        rootend=re.split('/',root)[-1]
        if re.match('[a-zA-Z0-9]',rootend):
            os.chdir(root)
            REPODIR=re.split('Analysis-Aline/',root)[-1]
            main(REPODIR)
    os.chdir(startdir)
