from pathlib import Path
import re

root = Path(__file__).resolve().parents[3]
p = root / 'docs/h84_block_fall_20261027/addendum.html'
s = p.read_text(encoding='utf-8')
s = re.sub(r'<div class="callout"><strong>Confirmed scene dimensions:.*?</div>', '''<div class="callout"><strong>Revised to your requested size:</strong> the cube is <strong>0.75 foot per edge: 9 inches, or 0.2286 metres</strong>. MIMo keeps its original body proportions and is about 0.83 metres tall. Both cameras show the same movement, without a caregiver. This 12-second animation is an authored movement study.</div>''', s, count=1)
s = s.replace('0.0 / 14.0 s','0.0 / 12.0 s').replace('max="140"','max="360"')
s = s.replace('src="h84_block_fall_20261027/block_', 'src="h84_block_fall_20261027/revision2/block_')
s = re.sub(r'<p><strong>How to read the motion:</strong>.*?</p>', '''<p><strong>Watch the support change:</strong> the body tips from its toes, bends the arms and legs near the floor, gathers the knees beneath the hips, brings one foot forward, and rises. The revised renderer solves hand and foot placement throughout the movement. It no longer lifts the whole body to conceal floor intersections. Contact forces still require a physical simulation; these pictures show the proposed arrangement and timing.</p>''', s, count=1)
start=s.index('<h3>What each part of the movement entails</h3>')
end=s.index('<p><strong>The pictures suggest',start)
s=s[:start]+'''<h3>What moves together, and when</h3>
<div class="table-scroll"><table><thead><tr><th>Illustrated time</th><th>Movement to inspect</th><th>What experience must teach UEMA</th></tr></thead><tbody>
<tr><td>0–1.60 s<br>Stand, then tip</td><td>The body rotates toward the block edge. The toes remain at the edge while the heels rise. The arms begin reaching toward the floor.</td><td>Use changing pressure and tilt to predict loss of support early enough to correct it.</td></tr>
<tr><td>1.60–1.86 s<br>Leave the edge</td><td>The pelvis descends, the trunk turns forward, and hips and knees bend to clear the block. The hands approach the floor.</td><td>Predict where contact will occur and how a changed command affects the whole body, including wrists and head.</td></tr>
<tr><td>1.86–3.70 s<br>Lower and gather support</td><td>Elbows bend as the chest lowers, then the chest rises again. Hips and knees bring the legs underneath. The hands remain near their floor positions.</td><td>Learn how elbow motion changes loading, and how to obtain support without overloading a hand or knee.</td></tr>
<tr><td>3.70–5.50 s<br>Shift from hands to knees</td><td>The hands lift as the chest rises over the knees. The arms return beside the trunk. The knees remain in place.</td><td>Predict whether knee and shin contact can support the body before releasing the hands.</td></tr>
<tr><td>5.50–7.30 s<br>Plant one foot</td><td>The right knee moves forward while the left knee stays down. The right foot swings forward, turns toward the floor and plants. The trunk shifts toward that foot.</td><td>Coordinate hip, knee and ankle movement so the foot clears the floor and the remaining contacts support the body.</td></tr>
<tr><td>7.30–10.30 s<br>Bring the other foot under, then rise</td><td>The left knee lifts. The left foot comes forward while the right foot stays planted. Both hips and knees extend as the trunk returns upright.</td><td>Keep support during the transfer. Correct for slipping, changed strength or uncertain pressure readings.</td></tr>
<tr><td>10.30–12.00 s<br>Finish upright</td><td>The final stance is held for inspection. The example sensor state continues recovering.</td><td>Standing again must not erase remaining fatigue or loss of sensing precision.</td></tr>
</tbody></table></div>
''' + s[end:]
a=s.index('<details><summary>Timing and geometry:')
b=s.index('</details></section>',a)+len('</details>')
s=s[:a]+'''<details><summary>Timing, contact checks and the remaining physical question</summary>
<p>The display has 30 frames per second, with 361 poses from 0 to 12 seconds. A 500 ms viewing interval spans 15 frame intervals. The proposed model still receives feedback within that interval; the pictures are not a command schedule for the learner.</p>
<p>From edge departure to the first low floor pose takes about 0.26 seconds. The authored pelvis path includes gravity during that interval. Rotations and limb movements are posed, so this is not a solved whole-body fall. The later recovery takes several seconds because support transfers between hands, knees and feet. The total duration is an illustration choice, not a measured learning result.</p>
<p>Each displayed pose is checked against floor and block geometry. Hand and foot targets are solved within joint limits; the record reports fitting errors as well as clearances. These geometry checks cannot measure load-bearing ability or impact forces. The next physical qualification must measure those quantities in the simulator. <a href="../Saved/h84/block_fall_design_20261027/evidence/revision2/motion.json">Revised pose and clearance record</a> · <a href="h84_block_fall_20261027/revision2/render.py">Renderer source</a>.</p>
<p>The meter illustrates an affected skin region with q = 0.20 at 1.86 seconds. This is a chosen example, not damage calculated from the pictured fall. It follows q = 1 − 0.8 exp(−elapsed/10 s), the existing zero-load recovery rule. At 12 seconds q is about 0.71. A value of 0.71 is a precision setting, not “71% accuracy.” The meter changes neither the pictures nor a running model.</p>
</details>''' + s[b:]
s=s.replace('rest of a fourteen-second get-up','rest of a longer get-up')
attention = Path(__file__).parent / 'past_future.html'
if attention.exists():
    s=s[:s.index('<section id="past-future"')]+attention.read_text(encoding='utf-8')
p.write_text(s,encoding='utf-8')
