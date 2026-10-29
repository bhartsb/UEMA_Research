"""Builds docs/H84_REACH_SOLUTION_VISUAL_v2_2026-10-24.html from the three part files beside this script and the picture
records in the parent folder (matched_pictures.json, repairC_capture.json; plan_pictures.js is read by the page itself).
Run from anywhere: python docs/h84_reach_plan_visual_20261023/page_build/build_v2.py"""
import io, os, re, json, html as H
S = os.path.dirname(os.path.abspath(__file__))  # the three part files live beside this script
OUT = os.path.normpath(os.path.join(S, '..', '..', 'H84_REACH_SOLUTION_VISUAL_v2_2026-10-24.html'))
head = io.open(os.path.join(S, 'part_head.html'), encoding='utf-8').read()
diagrams = io.open(os.path.join(S, 'part_diagrams.html'), encoding='utf-8').read()
script = io.open(os.path.join(S, 'part_script.html'), encoding='utf-8').read()
M = json.load(open(os.path.join(S, '..', 'matched_pictures.json'), encoding='utf-8'))
P = 'h84_reach_plan_visual_20261023/'
f1 = lambda x: '%.1f' % x
GAP = lambda label: re.search(r'([0-9.]+) cm', label).group(1)  # the gap as the picture's own legend prints it


def fig(src, alt, cap, cls=''):
    return '<figure%s><img src="%s%s" alt="%s"><figcaption>%s</figcaption></figure>' % ((' class="%s"' % cls) if cls else '', P, src, alt, cap)


body = []
A = body.append
A('<main>\n<header id="top"><div class="eyebrow">UEMA on the MIMo infant body &bull; the solution for reaching &bull; October 24, 2026</div>')
A('<h1>How MIMo will reach the target, step by step</h1>')
A('<p><strong>The task.</strong> MIMo is a simulated infant body moved by the physics program MuJoCo; UEMA is the learning program that controls it. UEMA picks a target T, a pose the body held earlier. The job is to bring both hands to where they are in T and keep them there.</p>')
A('<div class="key"><strong>How to read every picture.</strong> A picture is a real saved state of MIMo unless its caption starts with "Drawn". MIMo lies on its back; the renderer draws it upright without the floor, so gravity points into the picture (the grey "down" arrow; the angle off the line of sight is in each picture\'s label or caption). The picture\'s right is MIMo\'s left: the orange arm is MIMo\'s right arm, the blue arm its left. Orange ring = <strong>hand point</strong>, a fixed point inside the palm. Green square = where that hand point is in T. The <strong>gap</strong> = the distance between them, in cm, printed in each caption; the dashed line is the gap as seen from the camera. A model step is 50 ms. <strong>Arrived</strong> = both gaps at or under 2 cm at some model step 1 to 40; <strong>stayed</strong> = still so at each of the 20 model steps after the first model step both are inside.</div>')
A('<div class="grid g2" style="max-width:900px">')
A(fig('start_b1.png', 'Body 1 at its start pose', '<strong>Start.</strong> Body 1. Gap: left 6.7 cm, right 10.5 cm.'))
A(fig('done_b1.png', 'The target with 2 cm rings', '<strong>Target T, and "done".</strong> Body 1\'s T is its own pose 63 model steps earlier. <strong>Done</strong> = both hand points inside the green rings (2 cm around T\'s hand points): arrived and stayed, from a learning run\'s own commands, from step 1\'s eight starts and then from new starts.'))
A('</div>')
A('<nav><a href="#parts">1 The parts</a><a href="#now">2 What happens now</a><a href="#why">3 Why</a><a href="#step1">4 Step 1, measured</a><a href="#pick">5 The sequence</a><a href="#repairE">6 The repairs</a><a href="#sources">7 Sources</a></nav></header>')

# 1 parts
A('<section id="parts"><div class="eyebrow">1 &bull; The parts of UEMA, and where each step of the solution acts</div>')
A(diagrams)
A('<div class="key">The numbered boxes are the parts this page refers to; the grey dashed boxes are not steps of the loop. <span class="badge b1">STEP 1</span> the <strong>direct command</strong>: T\'s own rows put straight into the command box, parts 5 to 8 skipped. <span class="badge b2">STEP 2</span> every command comes from part 8; today every second command skips it (section 2). <span class="badge b3">REPAIR E</span> gives part 8 a second starting point: T\'s own rows, step 1\'s direct command. <span class="badge b3">REPAIR A, C</span> change what part 4 predicts: its predicted pose error is the score that decides whether repair E\'s candidate is sent. <span class="badge b3">REPAIR B</span> corrects the command before part 9 reads it. <span class="badge b3">REPAIR D</span> is the motors and the physics.</div></section>')

# 2 now
A('<section id="now"><div class="eyebrow">2 &bull; What happens now: the hands wander and never arrive</div>')
A('<h2>Body 1 in the run of October 23: T, the start, then the end of each of the first 17 commands (4.25 s). The same T throughout. Gravity <span id="seqgrav">&hellip;</span> from the line of sight.</h2>')
A('<p>Planner commands come from part 8 and last 5 model steps. Between two planner commands the body receives part 7\'s command, with random noise added while a <strong>babble bout</strong> is on (a <strong>babble command</strong>, <span style="color:#c34b11;font-weight:700">orange border</span>) and without noise while it is off (orange dashed border). Step 2 removes every orange-bordered frame: the planner sends the command in their place.</p>')
A('<div id="tframe" class="strip" style="grid-template-columns:calc((100% - 16px) / 3);margin-bottom:8px"></div>\n<div class="strip" id="strip"></div>')
A('<div class="verdict"><ul style="margin:0;padding-left:20px"><li>Neither hand arrived and stayed. Left hand\'s gap: 6.7 cm at the start, 1.5 cm at best, 8.0 cm at the end. Right hand\'s gap: never below 8.0 cm.</li><li>The command between two planner commands moves the hands as much as a planner command, and in a random direction (nearer in 54 of 108 readings, farther in 54).</li></ul></div></section>')

# 3 why
A('<section id="why"><div class="eyebrow">3 &bull; Why: part 4\'s prediction is wrong, part 8 cannot produce a command at T</div>')
A('<h2>Body 5, at one of the run\'s own decisions, 21 model steps after step 1\'s rebuilt start.</h2>')
A('<div class="grid g2">')
A(fig('b5_T.png', 'Body 5 target T', '<strong>T</strong>. The orange rings sit on the green squares.'))
A(fig('b5_start.png', 'Body 5 start', '<strong>Start</strong>. Gap: right 19.3 cm, left 4.9 cm.'))
A(fig('b5_prediction.png', 'Drawn: the predicted parts', '<strong>Drawn: what part 4 predicted for the chosen command</strong>, over the start pose. Purple dots = the predicted shoulder, elbow and hand of each arm; grey rings and dashed line = the <strong>fitted arm</strong>, the nearest arm a jointed body can make. Enlarged under repair A.'))
A(fig('b5_end.png', 'Body 5 measured end', '<strong>What the body did</strong> (250 ms later). Gap: right 23.5 cm, left 8.7 cm. The prediction said the right hand would end 2.8 cm nearer; it ended 4.2 cm farther.'))
A('</div>')
A('<div class="grid g3">')
A(fig('base_b5_region.png', 'Body 5 at the rebuilt start with the search\'s bound', '<strong>Why part 8 cannot choose a command at T.</strong> Body 5 at the rebuilt start of step 1. The first command\'s request (blue dot) is farther from T than the hand is. T\'s row (green square) is out of reach of every version.'))
A(fig('base_b5_region_hand.png', 'The search\'s bound, enlarged', '<strong>The right hand, enlarged 4&times;.</strong> The green arrow points toward T\'s row; the bound is enlarged at right.'))
A(fig('base_b5_region_bound.png', 'The bound, enlarged 12 times', '<strong>The bound at 12&times;.</strong> After smoothing the search can move a row at most 0.63 cm from the smoothed request (purple ring); the row sent (dark dot) sits on the square\'s corner.'))
A('</div>')


A('</div></section>')

# 4 step 1
A('<section id="step1"><div class="eyebrow">4 &bull; Step 1, the baseline: run October 23. Told the exact pose, the body reaches T in 2 bodies of 8.</div>')
A('<h2>From each of eight saved starts, one command was held for 60 model steps (3 s): the <strong>direct command</strong> (T\'s own rows) and, from the same rebuilt start, the <strong>planner command</strong> the run sent. <strong>Pose error</strong> is part 8\'s score (0 = every body part at T).</h2>')
A('<h3>Body 5: the direct command brings both hands to T; the planner command does not (gravity <span id="b5grav">&hellip;</span> from the line of sight)</h3>')
A('<div class="grid g3" id="b5strip"></div>')
A('<h3>The other seven bodies: the start, T, the direct command at 3 s, the planner command at 3 s</h3>')
A('<div id="basegrid"></div>')

A('<div class="verdict" id="verdict1"><ul style="margin:0;padding-left:20px"><li>Direct command: both hands arrived and stayed in bodies 2 and 5; the planner command in none. In all 16 hands the direct command\'s gap was smaller than the planner command\'s at 250 ms and at 3 s.</li><li>Five hands missed by 2.0 to 4.0 cm (bodies 0, 1, 4, 7); bodies 3 and 6 arrived and then a hand left; none explained yet (section 5, item 3).</li></ul></div></section>')

# 5 sequence
A('<section id="pick"><div class="eyebrow">5 &bull; The sequence, from today to done</div>')
A('<h2>One at a time; the overseer decides each step. The reading for the keep lines of repair E, step 2, repair B and repair D: arrived and stayed, the gap per hand, by more than the spread; a kept repair is then read again from new starts, the second half of done.</h2>')
A('<ol style="padding-left:22px">')
A('<li><strong>The change that stops mixing the new body-position request with the previous one: built and reviewed, switched off by default.</strong> In the completed comparison, that mixing left the requested hand positions 5 to 15 cm from their targets. Its comparison run: Not run: memory shortage, preparation overruns and insufficient verified remaining time.</li>')
A('</ol><h3>Later steps, chosen from the results</h3><ol start="2" style="padding-left:22px">')
A('<li><strong>Step 2</strong>: the planner picks every command (plan every 5 model steps instead of every 10), removing the commands between planner commands (the orange-bordered frames of section 2); read against two runs of the unchanged schedule from the same saved state.</li>')
A('<li><strong>The rerun with motor controls recorded</strong>, for the bodies step 1\'s direct command did not bring to T (0, 1, 3, 4, 6, 7): step 1 recorded no motor controls, so whether a motor at its limit stopped them is unmeasured (what it records: repair D\'s first picture).</li>')
A('<li><strong>Repair B</strong> if the planner\'s commands contain rows no joint can make.</li>')
A('<li><strong>Repair A</strong> (section 6), so part 4\'s score sends repair E\'s candidate on more bodies.</li>')
A('<li><strong>Repair C</strong>: retrain part 4 on captured outcomes, whole bodies held out.</li>')
A('<li><strong>Repair D</strong> for any body the rerun shows stalled at a motor\'s limit: find the joint, motor or contact and change one thing at a time.</li>')
A('</ol></section>')

# 6 repairs
A('<section id="repairE"><div class="eyebrow">6 &bull; The repairs, each in pictures</div>')
A('<h3><span class="badge b3">REPAIR E</span> Give part 8 a second starting point: T\'s own rows</h3>')
A('<p><strong>What it changes.</strong> One more candidate is scored beside the search\'s winner: the first command with its rows set to T\'s own. It is not clipped by the bound of section 3; it is scored once by the same body head after the same smoothing, and sent only if its pose error is below the search winner\'s, the first command\'s and the start pose\'s.</p>')
A('<div class="grid g2">')
A(fig('base_b5_candidate.png', 'Drawn: the row if the candidate wins one decision', '<strong>Drawn: what the candidate does to one decision</strong> (body 5 at the rebuilt start of step 1).'))
A(fig('base_b5_candidate_hand.png', 'Drawn: the candidate\'s effect, enlarged', '<strong>Drawn: the right hand, enlarged.</strong> The green arrow points toward T\'s row.'))
A('</div>')
A('<h3>What the comparison showed (October 24, the tested version, switch off against on): the 4 of 8 bodies sent the candidate, each hand enlarged with its two 250 ms positions.</h3>')
for b in ('0', '1', '5', '6'):
    r = M['bodies'][b]; off, on = r['reading']['off'], r['reading']['on']; g = round(r['off']['gravity']['angle_from_line_of_sight_deg'])
    A('<div class="grid g3" style="margin-bottom:14px">')
    A(fig('matched_b%s_start.png' % b, 'body %s at the decision' % b, '<strong>Body %s at the decision</strong> (gravity %d&deg; in this row). Gap: left %s cm, right %s cm.' % (b, g, f1(r['start']['D_cm_by_picture']['L']), f1(r['start']['D_cm_by_picture']['R']))))
    for side, nm in (('L', 'left'), ('R', 'right')):
        h = r['hands'][side]
        A(fig(h['picture'], 'body %s, %s hand, switch off against on' % (b, nm), '<strong>%s hand.</strong>' % nm.capitalize()))
    A('</div>')
A('<div class="verdict"><ul style="margin:0;padding-left:20px"><li>In the 4 bodies sent the candidate, the pose error at 250 ms was lower with the switch on in 4 of 4, and 7 of their 8 hands ended nearer T than with the switch off.</li><li>No hand was within 2 cm at 250 ms in either run; arriving and staying were not measured.</li><li>This version remains switched off by default. The next change is still being built and reviewed.</li></ul></div>')

A('<h3 id="repairA"><span class="badge b3">REPAIR A</span> Make part 4 predict an arm that is joined</h3>')
A('<p><strong>What it changes.</strong> A new loss term reads four gaps on each predicted arm (how far each predicted part sits off a jointed arm); a jointed arm reads 0 on all four, so lowering the term pulls the predicted parts onto a joined arm.</p>')
A('<div class="grid g3">')

A(fig('b5_afterA.png', 'Drawn: after repair A', '<strong>Drawn: after repair A</strong> (body 5\'s prediction of section 3). Hollow rings = the predicted parts now; filled dots = the same parts on a joined arm.'))
A(fig('b5_afterA_right_arm.png', 'Right arm after repair A, enlarged', '<strong>Drawn: the right arm at 4&times; after repair A</strong>: each ring\'s move to the joined arm, labelled with its length.'))
A(fig('b5_afterA_left_arm.png', 'Left arm after repair A, enlarged', '<strong>Drawn: the left arm at 3&times; after repair A</strong>.'))
A('</div>')
A('<p><strong>Kept if</strong>, with the term on, the predicted arm is nearer a jointed one in at least 45 of 69 examples and the forecast and the body\'s movement are no worse.</p>')

A('<h3 id="repairB"><span class="badge b3">REPAIR B</span> Correct the command before part 9 so every row is possible</h3>')
A('<p><strong>What it changes.</strong> Each requested arm row is turned to a direction its joint can make (repair D\'s body 3 pictures show the fault). The one body it was tried on is below.</p>')
A('<div class="grid g2">')
A(fig('cmp_rods.png', 'Original and corrected requested directions', '<strong>The start, with both commands\' requests</strong> (run seg41j, body 5; gravity 6&deg;). Blue rods = the original command\'s directions for the upper arm (U), forearm (F) and hand (H, at 3&times;); purple rods from the same origins = the corrected command\'s. Gap: left 3.1 cm, right 6.6 cm.'))
A(fig('cmp_rods_right_arm.png', 'Right arm rods, enlarged', '<strong>The right arm (the picture\'s left), enlarged</strong>: each purple line is the turn from the original direction to the corrected one, labelled in degrees.'))
A(fig('cmp_original_250ms.png', 'Original command at 250 ms', '<strong>Original command, 250 ms.</strong> Gap: left 2.7 cm, right 12.9 cm; pose error 11.6.'))
A(fig('cmp_corrected_250ms.png', 'Corrected command at 250 ms', '<strong>Corrected command, 250 ms.</strong> Gap: left 3.2 cm, right 12.8 cm; pose error 11.5. On this body the correction alone did not reach T.'))
A('</div>')

A('<h3 id="repairC"><span class="badge b3">REPAIR C</span> Teach part 4 from what the body actually did</h3>')
A('<p><strong>What it changes.</strong> Today part 4 learns only from the one version that was sent. Repair C captures, at each decision of the next learning run, the exact start, replays all 21 versions from it for 250 ms each, records what each did, and retrains part 4 on those pairs with whole bodies held out.</p>')
RC = json.load(open(os.path.join(S, '..', 'repairC_capture.json'), encoding='utf-8'))
A('<div class="grid g3" id="repairCcapture">')
A(fig('repairC_capture.png', 'What repair C captures, body 5', '<strong>What repair C captures:</strong> body 5 at the start of the decision, with the two of the 21 versions the record holds.'))
for side, nm in (('L', 'left'), ('R', 'right')):
    c = RC['crops'][side]; n = RC['numbers'][side]
    A(fig(c['picture'], '%s hand of the capture, enlarged' % nm, '<strong>%s hand, enlarged.</strong> The real end of version 21 is %s cm from where part 4 put it.' % (nm.capitalize(), f1(n['real_end_to_v21_predicted_cm']))))
A('</div>')
A('<div class="grid g3" id="repairCgrid">')
A(fig('b5_afterC.png', 'Drawn: predicted parts after repair C', '<strong>Drawn: after repair C</strong>, on the measured end state. Hollow rings = the predicted parts now; filled dots = the real parts.'))
A(fig('b5_afterC_right_arm.png', 'Right arm after repair C, enlarged', '<strong>Drawn: the right arm at 4&times;</strong>; each ring\'s move to the real part, with its length.'))
A(fig('b5_afterC_left_arm.png', 'Left arm after repair C, enlarged', '<strong>Drawn: the left arm at 3&times;</strong>; the same, with its lengths.'))
A('</div>')
A('<p><strong>Kept if</strong>: on the held-out bodies the predicted gap misses the measured gap by less than today, by more than the spread.</p>')

A('<h3 id="repairD"><span class="badge b3">REPAIR D</span> The motors and the physics: for a body whose direct command stalls</h3>')
A('<div class="grid g2">')
A(fig('b3_rods.png', 'Body 3: the command\'s requested directions', '<strong>What repair D inspects, on body 3 of the October 21 run, not step 1\'s body 3:</strong> the contacts, each arm part against its requested direction (gravity 16&deg;). Blue rods = the directions the command requested for the left upper arm (U), forearm (F) and hand (H, drawn at 3&times;); blue ball = where it asked the hand to go; yellow dots = floor contacts. Purple rod = the nearest forearm direction the elbow can reach with the upper arm as requested. Gap: left 9.3 cm, right 1.6 cm.'))
A(fig('b3_end.png', 'Body 3 at the command\'s end', '<strong>What that body did</strong> (the command\'s end; gravity 16&deg;). The rods drawn again from where each part now is; yellow dots = contacts. Gap: left 14.6 cm, right 5.8 cm. The left hand followed the blue ball\'s direction, away from T.'))
A('</div>')
A('<p><strong>For a body the rerun of section 5 shows stalled</strong>, repair D reads: each joint\'s angle against its angle in T, each motor\'s control (&minus;1 to 1; a motor at &plusmn;1 is at its limit, as body 3\'s shoulder motor is above), and the contact points (the yellow dots above). <strong>What changes:</strong> one motor gain, strength or support change at a time, after the cause is found. <strong>Kept if</strong>: the direct command then arrives and stays from the same start.</p></section>')


# 8 sources
A('<section id="sources"><div class="eyebrow">7 &bull; Sources</div><ul>')
A('<li>Every number in a caption and the hash of every picture: <a href="h84_reach_plan_visual_20261023/plan_pictures.json">plan_pictures.json</a> and <a href="h84_reach_plan_visual_20261023/matched_pictures.json">matched_pictures.json</a>.</li>')
A('<li>The 72-command readings: <a href="h84_body_repair_v2_20261023/three_way_72.json">three_way_72.json</a>. Body 3: <a href="h84_body_repair_v2_20261023/gizmo_detail.json">gizmo_detail.json</a> and <a href="h84_body_repair_20261021/provenance.json">provenance.json</a>. The long form with every decision: <a href="H84_REACH_SOLUTION_REVISED_2026-10-23.html">H84_REACH_SOLUTION_REVISED_2026-10-23.html</a>.</li>')
A('</ul></section>\n</main>\n')

page = (head + '\n'.join(body) + script).replace(' (drawn with the bout off)', '')
io.open(OUT, 'w', encoding='utf-8').write(page)
txt = re.sub(r'<(script|style|svg).*?</\1>', '', page, flags=re.S); txt = H.unescape(re.sub(r'<[^>]+>', ' ', txt))
print('written; words', len(txt.split()), 'pictures', page.count('<img '))
