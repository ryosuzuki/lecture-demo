from pathlib import Path
import urllib.request,hashlib,json,os
root=Path(__file__).resolve().parent
record=json.loads((root/'recorded-traces.json').read_text())
expected=record['provenance']['checkpointSha256']
dest=Path(os.environ.get('COCONUT_CHECKPOINT','/tmp/openclaw/coconut-model/checkpoint_49'))
if dest.exists() and hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()==expected:
 print('Verified existing checkpoint:',dest)
else:
 dest.parent.mkdir(parents=True,exist_ok=True)
 temp=dest.with_suffix('.download')
 urllib.request.urlretrieve('https://huggingface.co/bmarti44/coconut-curriculum-checkpoints/resolve/main/coconut/checkpoint_49',temp)
 if hashlib.file_digest(temp.open('rb'),'sha256').hexdigest()!=expected:
  raise RuntimeError('Checkpoint digest differs from verified recording; investigate before use')
 temp.replace(dest)
 print('Downloaded and verified:',dest)
