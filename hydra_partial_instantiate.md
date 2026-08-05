# `_partial_` in Hydra, and why both CLIs use it

Why this doc exists: `openretina/cli/train.py` and `openretina/cli/eval.py` build their dataloaders in
a way that looks needlessly roundabout —

```python
build_dataloaders = hydra.utils.instantiate(cfg.dataloader, _partial_=True)
dataloaders = build_dataloaders(**dataloader_kwargs)
```

— instead of the obvious one-liner `hydra.utils.instantiate(cfg.dataloader, **dataloader_kwargs)`.
The two are **not** equivalent, and collapsing them back into the one-liner silently costs ~28 GB of
RAM on the qiu_2026 dataset with no error raised anywhere. This explains the mechanics.

---

## 1. What `instantiate` normally does

A Hydra config node with `_target_` is a recipe for calling something:

```yaml
# configs/dataloader/qiu_2026.yaml
_target_: openretina.data_io.qiu_2026.dataloaders.qiu_2026_dataloaders
batch_size: 32
clip_length: 300
release_movies: true
```

`hydra.utils.instantiate(cfg.dataloader)` imports that function, calls it with the config's keys as
keyword arguments, and gives you **the return value**. Extra kwargs you pass are merged over the
config's own:

```python
dataloaders = hydra.utils.instantiate(cfg.dataloader, movies_dictionary=md)
#             └─ calls qiu_2026_dataloaders(batch_size=32, clip_length=300,
#                                           release_movies=True, movies_dictionary=md)
#                and returns the dict of DataLoaders
```

## 2. What `_partial_=True` changes

It stops short of calling. You get back a **`functools.partial`** — the target with the config values
pre-bound, waiting to be called:

```python
build = hydra.utils.instantiate(cfg.dataloader, _partial_=True)
# build == functools.partial(qiu_2026_dataloaders,
#                            batch_size=32, clip_length=300, release_movies=True)

dataloaders = build(movies_dictionary=md, ...)   # you call it, with normal Python arguments
```

Same function, same arguments, same result. The difference is *who* makes the call and *how the
remaining arguments travel*.

## 3. Why the feature exists at all

The classic motivation is that config knows some arguments and runtime knows others. An optimizer is
the textbook case, already documented in this repo at `openretina/utils/optimizer_utils.py:38-45`:

```python
optimizer_config = OmegaConf.create({
    "_target_": "torch.optim.SGD",
    "_partial_": True,
    "lr": 0.01, "momentum": 0.9,
})
optimizer = instantiate_optimizer(optimizer_config, model.parameters(), 0.01)
```

`lr` and `momentum` are config decisions. `params` cannot be — the model does not exist when the YAML
is written. So the config yields a *factory*, and the call site supplies the live object.

Generally you reach for `_partial_` when the missing argument is a live Python object, when you need
to call the factory more than once, or when you want the call deferred.

## 4. The part that matters for the dataloaders

Our missing arguments *are* live Python objects — the dictionaries of loaded movies, responses and
pupil traces. And there is a second, much less advertised consequence of the choice:

> **Arguments passed *through* `instantiate(...)` do not reach the target verbatim.** Hydra merges
> them into the OmegaConf config tree, then converts back to Python according to `_convert_`. That
> round-trip rebuilds containers, and with this repo's `_convert_: object` it also reconstructs
> dataclass instances.

Measured directly, not assumed (one session, both call styles):

```
via instantiate(cfg.dataloader, movies_dictionary=md):
  same wrapper object?  False        <- new MoviesTrainTestSplit instance
  same .train buffer?   True         <- identical numpy data pointer
  caller dict entries after: 1       <- release_movies had NO effect

via _partial_=True then build(movies_dictionary=md):
  same wrapper object?  True
  same .train buffer?   True
  caller dict entries after: 0       <- release_movies worked
```

The numpy arrays are **not** copied either way, so this was never a hidden data duplication. What
gets rebuilt is the thin dataclass wrapper around them — which is enough to break the memory fix:

```
   plain instantiate                          _partial_ + plain call

train_model:  movies_dict ──► {k: Movies}     train_model: movies_dict ──► {k: Movies}
                                  │                                     │       │
                                  ▼                                     │       ▼
                            ┌── ndarray ──┐                             │   ndarray
                                  ▲                                     │       ▲
builder:  movies_dictionary ──► {k: Movies'}   builder: movies_dictionary┘───────┘
                                                        (the SAME dict)

pop() removes the builder's entry.             pop() removes the only entry.
train_model's wrapper still holds the          Last reference gone →
array → refcount > 0 → nothing freed.          the movie is actually freed.
```

`release_movies` (see `openretina/data_io/qiu_2026/dataloaders.py`) frees each session's source movie
as soon as its splits are taken. In the left-hand shape it is a silent no-op: no error, no warning,
just no memory saving. Measured over all 10 qiu_2026 sessions, working vs not working is a peak RSS
of **58.9 GB vs 86.7 GB** — and 86.7 GB is what OOM-killed the sweep's 80G SLURM tasks.

`tests/data_io/test_qiu_2026_train_wiring.py::test_release_movies_needs_the_partial_call_style` pins
this, so the one-liner cannot quietly come back.

## 5. Two practical mechanics worth knowing

**Call-time kwargs override bound ones.** `functools.partial` lets keyword arguments at call time win
over pre-bound ones, which is how the tests exercise the other arm despite the config saying `true`:

```python
build = hydra.utils.instantiate(cfg.dataloader, _partial_=True)  # release_movies=True bound
build(movies_dictionary=md, release_movies=False)                 # override wins
```

**The flag is passed at the call site, not in the YAML — deliberately.** Putting `_partial_: true`
into `configs/dataloader/qiu_2026.yaml` would change what *every* caller gets back. The notebooks do:

```python
dataloaders = hydra.utils.instantiate(cfg.dataloader, movies_dictionary=movies_dict, ...)
dataloaders["train"]   # would now be a functools.partial → TypeError
```

Keeping the flag at the two CLI call sites means the config still behaves as a plain "call me" node
for everyone else. That is also why the notebooks are unaffected in every respect: they keep the
copying path, so `release_movies` is inert there and their `movies_dict` survives for the
stimulus-plotting cells — at the cost of not getting the memory saving unless they are switched over
too (a two-line change per notebook).

---

## Reference

| what | where |
|---|---|
| the `_partial_` call sites | `openretina/cli/train.py`, `openretina/cli/eval.py` |
| `release_movies` implementation + docstring | `openretina/data_io/qiu_2026/dataloaders.py` |
| the flag and its measured numbers | `configs/dataloader/qiu_2026.yaml` |
| test pinning the call style | `tests/data_io/test_qiu_2026_train_wiring.py` |
| peak-RSS measurement harness | `scratch_qiu_measure_peak_rss.py` / `.sh` |
