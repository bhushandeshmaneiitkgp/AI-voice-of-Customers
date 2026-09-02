"""
Layer 15 -- evaluation.

Every earlier phase produces numbers. This one asks whether they are worth
anything, and it is built around a distinction the rest of the pipeline cannot
make on its own:

* **Agreement** is two systems saying the same thing. It is cheap, it is
  available today, and it is not evidence of correctness -- two models can be
  wrong in the same way, and on this corpus they demonstrably are wrong in
  *different* ways only 39% of the time.
* **Accuracy** is a system matching a reference a human produced. It is
  expensive, it is the only thing that licenses the word "correct", and it does
  not exist until somebody labels the gold set.

The modules here are deliberately arranged so that the second is impossible to
fake. ``metrics`` scores a candidate against a reference and does not know or
care where the reference came from; ``agreement`` feeds it another model and
names its outputs accordingly; ``goldset`` feeds it human labels and refuses to
produce anything at all until they exist.

Two studies need no reference of either kind, and so run today:

* ``faults`` injects known errors into real enrichments and measures what share
  each validator catches. The pipeline reports 98.4% grounding; this is what
  says whether that number is a measurement or a formality.
* ``retrieval_eval`` scores the RAG layer against the corpus labels, which is a
  weaker reference than a human but a genuinely independent one -- the
  embeddings and the labels come from different models that never see each
  other's output.
"""
