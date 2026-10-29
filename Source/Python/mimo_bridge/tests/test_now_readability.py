# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Current warnings and learning modes must not imply dead modules or proven skills."""
import json

from mimo_bridge.tools import development_dashboard as dashboard


def test_partial_layer_response_is_a_warning_and_names_repair_coverage(tmp_path):
    layer = 'motor_controller.residual_blocks.1.0'
    repaired = 'tenure_heads.pose.0'
    card = dashboard.liveness_card({'liveness': {
        'layers': {layer: .4609, repaired: 1.0},
        'pools': {'motor.pool': .001},
        'revived': [repaired], 'watching': {repaired: {'samples': 800}},
    }}, measurements=tmp_path)
    assert card['summary'] == '1 response reading needs attention'
    # the issue carries its part's definition, which the alert at the top of the page prints where it
    # names the part (review of 2026-09-25, finding 8)
    # since 2026-09-30 each flagged part also carries its measured verdict (the model alert's rewrite)
    issue = card['issues'][0]
    assert len(card['issues']) == 1 and {k: issue[k] for k in ('name', 'source', 'responding', 'reading',
                                                               'definition', 'repair')} == dict(
        name='Motor controller, block 2', source='model', responding=46, reading='46',
        definition=('The motor controller is the network that proposes a command for each body part; its '
                    'confidence head sets how much each of its residual blocks counts.'),
        repair='Monitored only; automatic repair does not cover this layer.')
    assert issue['verdict']
    assert card['repair_summary'] == '1 of 1 previously repaired layers are now above the warning line.'
    assert 'parts of the model, not body parts' in card['explanation']
    near_line = dashboard.liveness_card({'liveness': {'layers': {layer: .4999}}}, measurements=tmp_path)
    assert near_line['issues'][0]['reading'] == 'Just under 50'


def test_attention_concentration_does_not_create_a_layer_failure(tmp_path):
    card = dashboard.liveness_card({'liveness': {
        'layers': {'some.layer': .9}, 'pools': {'some.pool': 0.0},
    }}, measurements=tmp_path)
    assert card['status'] == 'ok' and card['issues'] == []
    assert 'No line is drawn for attention spread' in ' '.join(ln['text'] for ln in card['lines'])
    assert dashboard.liveness_card({}, measurements=tmp_path)['status'] == 'unmeasured'


def test_shuffled_comparison_response_cannot_hide_weak_actual_order_lessons(tmp_path):
    card = dashboard.liveness_card({
        'liveness': {'layers': {'efference.body_mlp.0': 1.0}},
        'efference': {'liveness': {'body_mlp.0': .3}},
    }, measurements=tmp_path)
    assert card['status'] == 'attention' and card['below'] == 1
    # the code's own name for the part (lem/model/efference.py: the body head's first layer), defined
    # in the card where it is first named (the operator's rule: never rename a part)
    assert card['issues'][0]['name'] == 'Body head — actual-command lessons'
    assert card['issues'][0]['responding'] == 30
    assert [(r['name'], r['responding']) for r in card['readings']] == [
        ('Body head — all monitored inputs', 100),
        ('Body head — actual-command lessons', 30),
    ]
    assert 'deliberately shuffled commands' in card['scope_explanation']
    # since 2026-09-30 definitions live in the card's glossary and the flagged part's issue
    shown = json.dumps([card['glossary'], card['issues']], ensure_ascii=False)
    assert 'The body head, one of the efference heads, predicts' in shown
    assert 'gaze copy' not in json.dumps(card['issues'], ensure_ascii=False), 'a part the card does not name is not defined'


def test_now_separates_learning_from_control_and_does_not_promote_a_measurement_to_a_skill():
    args = dict(motor_synergy='on', synergy_planner='off', synergy_babble='off')
    panel = {'items': [dict(beats=True), dict(beats=False), dict(measured=True)]}
    summary = dashboard.now_summary({}, args, panel)
    assert summary['synergy'] == 'Learning from movement; not yet directing it'
    assert 'not yet been demonstrated' in summary['title']
    assert 'identical starting states' in summary['next']
    args['synergy_babble'] = 'on'
    assert 'exploration' in dashboard.now_summary({}, args, panel)['synergy_detail']
    assert 'planning' in dashboard.now_summary({}, {'motor_synergy': 'on'}, panel)['synergy_detail']


def test_camera_and_viewer_freshness_are_independent():
    frame = dashboard.live_texts(dict(frame_age_seconds=0, state_age_seconds=100),
                                 {'model_steps_per_s': 1}, {'live_view_every': 1, 'live_state_every': 1})
    assert frame['frame_publishing'] and not frame['state_publishing']
    pose = dashboard.live_texts(dict(frame_age_seconds=100, state_age_seconds=0),
                                {'model_steps_per_s': 1}, {'live_view_every': 1, 'live_state_every': 1})
    assert pose['state_publishing'] and not pose['frame_publishing']
    assert pose['texts']['pub'].startswith('no:')


def test_health_names_identify_components_without_exposing_code_paths():
    expected = {
        # a sense is named as the model's registry spells it (the operator's rule, 2026-09-25)
        'recon_decoders.vestibular.0': 'the vestibular reconstruction decoder',
        'forward_model.blocks.1.ct_attn': 'forward model, block 2: attention over time',
        'encoder_registry.encoders.vision.blocks.2.attn': 'vision encoder, attention layer 3',
        'prediction_heads.heads.vestibular.attn': "vestibular prediction head's input selection",
        'recon_decoders.proprioception.0': 'the proprioception reconstruction decoder',
        'encoder_registry.encoders.tactile.pool_attn': "the tactile encoder's pool (which contacts the tactile token reads)",
        'gaze_pool': "gaze head's input selection",
        'body_pool': "body head's input selection",
        'contact_head.ctx1': "the contact head's body-and-command input layer",
    }
    for path, name in expected.items():
        assert dashboard._part(path) == name
    # the code's names for the efference parts, never a second name (the operator's rule, 2026-09-19)
    assert dashboard._response_name('efference', 'vision_in') == 'gaze copy — actual-command lessons'
    for path in ('gaze_pool', 'body_pool', 'voice_head.attn', 'prediction_heads.heads.vision.mlp.0'):
        assert 'predictor' not in dashboard._part(path), path


def test_the_page_names_the_efference_parts_by_the_codes_names():
    """The dashboard's own sources, swept for the names the page used to coin for the efference
    heads' parts and the contact head ('eye predictor', 'body predictor', 'eye-to-vision predictor',
    'contact predictor'): lem/model/efference.py calls them the gaze head, the body head and the gaze
    copy, lem/model/contact_head.py the contact head. Each of the code's names has a definition in the
    page's glossary."""
    import os
    import re
    here = os.path.dirname(dashboard.__file__)
    # any "<word> predictor" is a coined name for a part, except the generic uses: a separate
    # predictor (a model other than the running one) and a reference predictor (the physics-only or
    # fitted comparison a goal line is read from); a glossary entry "Vision predictor" passed the
    # earlier list of five names (review of 2026-09-25, finding 7). And the family is not the member:
    # the controllability channel reads the body head's order-long expectation (finding 9)
    generic = {'separate', 'reference', 'physics-only'}
    # a part's canonical plain name in the component manifest is not a coined name (the recovery directive of Oct 5,
    # 2026, section 10: one component, one name): "wellness predictor" is WELLNESS_PREDICTOR's, which the code calls
    # ApproximateEvaluator, and the page says so where it defines it
    manifest = json.loads((dashboard.REPO/'docs/gain_of_function_v4/component_manifest.json').read_text(encoding='utf-8'))
    canonical = {c['name'].lower().rsplit(' predictor', 1)[0].split()[-1] for c in manifest['components']
                 if c['name'].lower().endswith(' predictor')}
    assert canonical == {'wellness'}, canonical
    generic |= canonical
    for name in ('development_dashboard.py', 'development_dashboard.js', 'development_dashboard.html',
                 'development_dashboard_terms.js', 'lineage_series.py', 'dashboard.py'):
        text = open(os.path.join(here, name), encoding='utf-8').read()
        coined = [m.group(0) for m in re.finditer(r"([\w'-]+) predictors?\b", text, re.I)
                  if m.group(1).lower().strip("'") not in generic]
        assert not coined, f'{name} coins part names: {coined}'
        assert "efference head's expectation" not in text, f"{name} names the family for the body head"
    terms = open(os.path.join(here, 'development_dashboard_terms.js'), encoding='utf-8').read()
    for term in ('Body head', 'Gaze head', 'Gaze copy', 'Prediction head', 'Contact head', 'Voice head'):
        assert re.search(r"'%s':'[^']" % term, terms), f'the glossary has no definition of {term}'


def test_record_start_is_idempotent_and_saved_manual_clips_remain_linked(tmp_path):
    from pathlib import Path
    run = Path('Saved/mimo_bridge/experiments/test')
    live = tmp_path / run / 'live'
    (live / 'rec').mkdir(parents=True)
    training = dict(run=run.as_posix(), run_name='test')
    assert dashboard.live_command(tmp_path, training, {'action': 'record'})['recording']
    frame = live / 'rec/frame_00001.jpg'
    frame.write_bytes(b'captured frame')
    assert dashboard.live_command(tmp_path, training, {'action': 'record'})['kept_frames'] == 1
    assert frame.read_bytes() == b'captured frame'
    clip = live / 'rec/unnamed.mp4'
    clip.write_bytes(b'saved clip')
    dashboard.live_command(tmp_path, training, {'action': 'save', 'file': clip.name, 'name': 'My observation'})
    state = dashboard.live_state(tmp_path, training, {}, {})
    assert [c['name'] for c in state['ready']] == ['My observation.mp4']
    assert state['ready'][0]['url'].endswith('/videos/My%20observation.mp4')
