"""Record closure of the authorized document correction, not model implementation."""
from pathlib import Path
import hashlib,json,datetime,re
root=Path(__file__).resolve().parents[2]
doc=root/'docs/H84_WHOLE_BODY_REACH_AND_GRASP_2026-10-26.html'
now=datetime.datetime.now()
p=root/'docs/gain_of_function_v4/coordinator/work_items/h84_whole_body_reach_document_20261026.json'
w=json.loads(p.read_text())
w['status']='document_specificity_correction_delivered'
w['next_action']='Operator reviews revised document. Model implementation, diagnostics and training remain stopped; no old launch authority resumes.'
w['correction_result']={
 'time':now.isoformat(),'elapsed_minutes':round((now-datetime.datetime.fromisoformat(w['correction_block']['start'])).total_seconds()/60,2),
 'sha256':hashlib.sha256(doc.read_bytes()).hexdigest(),
 'result':'Specific input/calculation/output/consumer and learning targets; concrete timed command search; actual MIMo thumb-opposed finger wrap; repaired continuous approach; independent specificity and geometry reviews reconciled.',
 'geometry':'61 displayed frames, 600 intermediate samples; minimum sampled hand-ball gap0.00157069mm; checked self gap0.070383mm; hand-stand12.712345mm. Sampled geometry only; zero physical integration steps.',
 'browser':'11 checks PASS, 1440px and390px,20 screenshots. Diagram parser issue corrected and rechecked.',
 'originals':'Both original source reports and writing guide hashes unchanged',
 'scope':'Document and static rendering only. Proposed settings require physical qualification after implementation authorization.',
 'reviews':['specificity_review_round1.txt','specificity_review_round2.txt','specificity_review_final.txt','specificity_closure.txt','wrap_review_final.txt']}
w['temporary_file_review']['reviewed']=now.isoformat()
w['temporary_file_review']['items']=[{'path':str(x.relative_to(root)),'purpose':'temporary screenshot browser profile','retention':'Regenerable; retained pending scoped authorized cleanup after document review'} for x in (root/'Saved/h84/whole_body_reach_document_20261026/tmp').glob('edge_profile_*')]
p.write_text(json.dumps(w,indent=2)+'\n',encoding='utf-8')
p=root/'docs/H84_RESUME_HERE.md';s=p.read_text(encoding='utf-8')
s=re.sub(r'^DOCUMENT CORRECTION ACTIVE:.*?\n\n','DOCUMENT SPECIFICITY CORRECTION DELIVERED: The rejected whole-body report has been revised with concrete algorithms, learning targets and corrected opposed-hand renderings. Prior acceptance remains withdrawn; current review evidence is linked below. Implementation and training remain stopped.\n\n',s,count=1,flags=re.S)
a=s.index('# DOCUMENT-ONLY RESULT');b=s.index('# H84: resume here',a)
s=s[:a]+'''# DOCUMENT-ONLY CORRECTION — October 26, 2026

[Revised whole-body reach-and-grasp design](H84_WHOLE_BODY_REACH_AND_GRASP_2026-10-26.html) now specifies command fields, timing/sample algorithm, observed-outcome retrieval, object identity/ego-motion, joint/object/contact prediction and finite-duration value targets. Independent reviews also corrected successful-goal maintenance, failed-attempt costs and missing score consumers. Both original reports remain unchanged. Six 500ms illustrated moments are distinct from the proposed50ms command clock and six-second goal-evaluation period.

Actual MIMo posed renderings show thumb opposition and finger bends from two close-up perspectives, with a repaired wrist approach. 61 displayed frames and600 intermediate samples checked; sampled geometry is not physical force validation. Desktop/narrow browser11checksPASS and20screenshots. Review and work record: docs/gain_of_function_v4/coordinator/work_items/h84_whole_body_reach_document_20261026.json.

Next action: operator reviews the corrected document. Model repairs, diagnostics and training remain STOPPED. No prior continuation/launch authority resumes. Temporary profiles and model cache retained for scoped cleanup; no deletion performed.

''' +s[b:]
p.write_text(s,encoding='utf-8')
print(json.dumps(w['correction_result']))
