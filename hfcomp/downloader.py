import os
# os.environ["NBOX_LOG_LEVEL"] = "warning"
# os.environ["NBOX_NO_AUTH"] = "1"
# os.environ["NBOX_NO_LOAD_GRPC"] = "1"
# os.environ["NBOX_NO_LOAD_WS"] = "1"
# os.environ["NBOX_NO_CHECK_VERSION"] = "1"

import json
from fire import Fire
from tqdm import tqdm
from functools import lru_cache
from datasets import load_dataset

from requests import Session
from nbox.subway import Subway
from nbox.nbxlib.astea import Astea, IndexTypes

from hfcomp import secret as S

# Some basic hardcoded data
lang_to_comment_prefix = {
  "Assembly": "; ",
  "Batchfile": ":: ",
  "C++": "// ",
  "C#": "// ",
  "CMake": "# ",
  "CSS": "/* ",
  "Dockerfile": "# ",
  "Fortran": "! ",
  "Go": "// ",
  "Haskell": "-- ",
  "Java": "// ",
  "JavaScript": "// ",
  "Julia": "# ",
  "Lua": "-- ",
  "Makefile": "# ",
  "Markdown": "# ",
  "Perl": "# ",
  "PHP": "// ",
  "PowerShell": "# ",
  "Python": "# ",
  "Ruby": "# ",
  "Rust": "// ",
  "Scala": "// ",
  "Shell": "# ",
  "SQL": "-- ",
  "TeX": "% ",
  "TypeScript": "// ",
  "Visual Basic": "' ",
}

def get_sample_from_data(data):
  tea = Astea(code = data["content"])
  classes = tea.find(types = [IndexTypes.CLASS])
  items = []

  def _create_sample_for_function(fn, prefix: str = ""):
    ds = fn.docstring().strip()
    if ds:
      code_wo_ds = fn._code.replace(fn.index[0]._code, "")
      _path = data["repository_name"] + ":" + data["path"]
      if prefix:
        _path += ":" + prefix
      if data["lang"] == "HTML":
        prefix = "<!-- " + _path + " -->"
      if data["lang"] in lang_to_comment_prefix:
        prefix = lang_to_comment_prefix[data["lang"]] + _path
      return prefix+"\n"+code_wo_ds, fn.docstring()
    return None, None

  # Iterate over all the classes and their functions in the file
  for c in classes:
    fns = c.find(types = [IndexTypes.FUNCTION])
    for fn in fns:
      code, docstring = _create_sample_for_function(fn, c.name)
      if code and docstring:
        items.append({"code": code, "docstring": docstring})

  # Iterate over all the functions in the file 
  fns = tea.find(types = [IndexTypes.FUNCTION])
  for fn in fns:
    code, docstring = _create_sample_for_function(fn)
    if code and docstring:
      items.append({"code": code, "docstring": docstring})
  return items


@lru_cache(1)
def _get_stub():
  url: str = S.ES_URL
  session = Session()
  session.auth = (S.ES_USERNAME, S.ES_PWD)
  es = Subway(url, session)
  return es

def main(
  n: int = 1_000_000,
  f_idx: str = 'files',
  cs_idx: str = 'code_snippets',
):
  """This script loads all the relevant data in our DBs and exits"""
  token = S.HF_TOKEN
  dataset = load_dataset(
    "bigcode/the-stack-dedup",
    data_dir = "data/python",
    use_auth_token = token,
    streaming = True,
    cache_dir = "./",
    split="train",
  )

  # check if "files" and "code_snippets" exists or not
  es = _get_stub()
  try:
    es.u(f_idx)()
  except:
    es.u(f_idx)(method = "put")

  try:
    es.u(cs_idx)()
  except:
    es.u(cs_idx)(method = "put")

  # update headers
  es._session.headers.update({'Content-Type': 'application/x-ndjson'})

  # main loop for processing all the datasets
  pbar = tqdm(enumerate(dataset), total = n)
  samples_count = 0
  failed = 0
  calls = 0
  data = []
  for i, x in pbar:
    if i == n or samples_count >= n:
      break

    # run the worker
    # process data packet, if not possible skip it, there are (mostly syntax) errors in about 10% of the data
    # items = get_sample_from_data(x)
    try:
      items = get_sample_from_data(x)
      samples_count += len(items)
    except:
      # somethings about the errors:
      # - errors creep up about 10% of the times
      # - most of them are caused due to python2 print statements
      failed += 1
      continue

    items_data = [
      {"index": {"_index": f_idx, "_type" : "_doc", "_id": i,}},
      {"repo": x["repository_name"],"file": x["path"], "language": x["lang"], "content": x["content"],}
    ]
    for item in items:
      items_data.extend([
        {"index": {"_index": cs_idx, "_type" : "_doc"}},
        {"file_idx": i, "data": item}
      ])

    data.extend(items_data)

    if len(data) > 100:
      es._bulk(method = "post", data = "\n".join([json.dumps(x) for x in data]) + "\n")
      calls += 1
      data = []

    if i % 100 == 0:
      pbar.set_description(f"Failed: {failed}, Calls: {calls}, Samples: {samples_count}")

  if data:
    es._bulk(method = "post", data = "\n".join([json.dumps(x) for x in data]) + "\n")
    calls += 1
    data = []

  print(f"Failed: {failed}, Calls: {calls}, Samples: {samples_count}")

def es_iterate_all_documents(index, pagesize=250):
  """
  Helper to iterate ALL values from
  Yields all the documents.
  """
  es = _get_stub()
  stub = es.u(index)._search
  offset = 0
  while True:
    result = stub(json={"size": pagesize, "from": offset})
    hits = result["hits"]["hits"]
    if not hits:
      break

    # Yield each entry
    yield from (hit['_source'] for hit in hits)
    # Continue from there
    offset += pagesize

if __name__ == "__main__":
  Fire(main)
