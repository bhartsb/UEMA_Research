# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The explainer: one snapshot schema, three producers, one viewer.

A snapshot is a JSON record of ONE model step - what the senses delivered, what the
core attended to, what was predicted against what arrived, what the drives scored and
what order went to the body. Producers: the live trainer (train_mimo --explain-*),
the checkpoint probe (tools/explain_probe.py), and nothing at all for the replay view,
which reads the stats log the run already writes. The viewer
(tools/build_explainer.py) renders all three into a single static HTML page.

Capture contract (the part that was easy to get wrong - see the plan's gap register):
  * context   = model.last_context, the PRE-shift buffer the live forward consumed;
                the post-step context_buffer already holds the current frame.
  * tokens    = model.last_tokens_fed, the post-gate tokens the core actually saw
                (model.last_tokens stays RAW - that is the prediction target).
  * weights   = one optimizer update NEWER than the live pass in a training step
                (labeled "post_update" in the header); exact in --eval-only probes
                ("frozen").
"""
