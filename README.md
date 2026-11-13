# beets-similarity

Compare **tracks in your local beets library**, using
`beets.autotag.match.track_distance`. No external metadata service
is queried. Scores are stored in a new `similarity` table in
the existing beets database; your tags and global autotagger settings are not
modified.

## Installation

Requires Python 3.12+ and beets 2.14.1+:

```sh
pip install .
# For comparing stored Chromaprint fingerprints:
pip install '.[acoustic]'
```

For fingerprint comparison, also install your OS's **libchromaprint** package
(`libchromaprint1` on Debian/Ubuntu). Enable `similarity` in your beets plugins:

```yaml
plugins: [similarity]
```

## Commands

```sh
beet similar --add artist:"Daft Punk"
beet similar --add -f artist:"Toto"         # refresh existing scores
beet similar --list artist:"Daft Punk"
beet similar --list --thres 0.35 artist:"Daft Punk"
beet similar --list --matrix artist:"Daft Punk"
beet similar --list --matrix --debug artist:"Daft Punk"
beet similar --associate 12 34
beet similar --dissociate 12 34 -f
beet similar --remove artist:"Toto"
beet similar --remove                      # full reset, extra confirmation
beet similar --size                        # similarity + refresh-state storage, in bytes
```

`--add` computes every unordered pair **within** the matched query, not against
the rest of the library. It is also the refresh command. All normal beets track
queries work. An empty query selects all tracks. Overwriting scores, changing
an existing manual choice, and removing scores require confirmation unless
`-f`/`--force` is supplied. Refreshes preserve manual choices; remove the pair's
scores first to clear a manual choice. Scoped removal deletes all pairs
touching the selected tracks, including pairs with tracks outside the query.

### Score and output conventions

**Higher is more similar:** `1` means identical under the configured penalties,
`0` means maximally different. Similarity is `1 - distance`, with the two
directional autotagger distances averaged to make it symmetric. This resolves
the distance/similarity ambiguity in the original proposal: `--thres` is a
**minimum similarity**, not a maximum distance. It must be in `(0, 1]`.

To explain a computed score, use beets' `-vvv` verbosity and recompute the
pair's scope, for example `beet -vvv similar --add -f artist:"Daft Punk"`.
With `-v` or `-vv`, only the usual pair score is logged, without the breakdown.
At verbosity three or higher, debug logs show each directional distance (raw penalty / total applicable
weight), then its nonzero penalties, largest first: compared metadata values,
configured weight, normalized distance, and the amount subtracted from the
final similarity. Each directional contribution is divided by two, so all
reported subtractions sum to `1 - score`. Zero penalties, zero-weight components,
and unavailable evidence do not appear in the penalty lines. Acoustic penalties
identify whether saved fingerprints or AcoustID IDs were used, without dumping
fingerprints. The final line shows `score = 1 - (forward + reverse) / 2`.
This is separate from the `similar --debug` display option; listing stored
scores does not recompute or explain them.

The default action is `--list`. Its deterministic, ID-sorted output includes
every selected track, with neighbors at or above the threshold:

```text
12:	34(0.950000); 56(0.850000)
34:	12(0.950000)
56:	12(0.850000)
78:
```

The ID and neighbors are separated by a tab, neighbors by `; `, and scores have
six decimal places. Only neighbors **inside the query** appear. Warnings and
prompts use beets' usual logging/UI channels; use a scoped query and `--force`
for noninteractive computation.

`--matrix` displays all stored scores in an aligned, symmetric square, without
threshold filtering. It uses `1.000000` for the self-similarity diagonal and
`-` for an **uncomputed** pair (not zero).
`--debug` replaces IDs with the final two directories and filename; tabs and
newlines in paths are escaped. Use IDs, not debug paths, for script integration.
Debug lists apply the same minimum similarity threshold as regular lists
(`threshold` in configuration, overridden by `--thres`) and color retained scores
with beets' `ui.colors.text_success` (green by default). This honors `ui.color`
and `NO_COLOR`. Combining `--matrix` with `--debug` still shows all stored scores
without threshold filtering.

### Manual decisions

Associating a pair adds the bonus needed to reach similarity `1`. Dissociating
subtracts `dissociate_penalty` from the base score, with a floor of `0`; the
base computed score is retained. The default `to-zero` value uses the base score
itself as the penalty, preserving the previous behavior. These decisions
survive refreshes and are symmetric; manual operations require two existing,
distinct item IDs.

## Configuration

Defaults, all under the `similarity` section:

```yaml
similarity:
  auto: no
  refresh_state: yes
  remove_on_delete: no
  threshold: 0.8
  acoustic: yes
  dissociate_penalty: to-zero
  warn_items: 1000
  warn_pairs: 500000
  fuzzy_artists: no
  bare_ascii_artists: no
  artist_threshold: 0.2
  weights:
    track_title: 4.0
    track_artist: 3.0
    track_length: 2.0
    track_id: 4.0
    track_index: 0.5
    medium: 0.5
    album: 2.0
    release_group: 6.0
    acoustic: 8.0
    data_source: 0.0
```

Weights must be finite and nonnegative; at least one applicable weight must be
positive. The default release-group weight prioritizes the same MusicBrainz
release group when IDs exist. Missing album/release-group information contributes
a mismatch if only one side has it; if neither has it, that component is omitted.
Title, artist, duration, numbering and recording-ID penalties come from beets'
autotagger, including its normalization and duration tolerances. Weights are
applied only to this plugin's temporary distance objects.
**Bitrate, format, sample rate, and other quality fields never affect scores.**

### Acoustic evidence

With `acoustic: yes`, each comparison includes an acoustic distance component:

1. If both tracks have `acoustid_fingerprint`, compare the saved Chromaprint
   fingerprints using pyacoustid (even if their AcoustID IDs agree), averaging
   both alignment directions for symmetry.
2. Otherwise, if both have `acoustid_id`, use ID equality as acoustic evidence.
3. If neither source is available for both tracks, omit the acoustic component
   rather than penalizing the comparison.

Populate these fields with beets' **chroma** plugin before computing scores.
This plugin does not generate fingerprints or make AcoustID network requests.
Invalid fingerprints or missing fingerprint-decoding dependencies produce an
actionable error rather than silently ignoring acoustic evidence. Set
`acoustic: no` to disable the component entirely. Scores already stored are
unchanged by configuration edits until refreshed.

### Automatic computation

`auto: yes` adds an import stage to compute imported tracks against local tracks
sharing their artists. Both album imports and singleton imports are handled as
individual tracks. With `refresh_state: yes` (the default), after an item's
first refresh, beets' `database_change`
event queues it only when similarity-relevant metadata changes; unrelated
edits such as changing the composer do not trigger a refresh. Items without a
recorded refresh baseline are refreshed once. Queued changes are refreshed at
command exit, coalescing repeated stores. For long-running integrations, call
`refresh_items(lib, items)` explicitly. Set `refresh_state: no` to disable the
stored signatures and refresh after every item database change.
Set `remove_on_delete: yes` to list and confirm before removing similarity
pairs that touch a deleted item. This is disabled by default.

By default, artist names match case-insensitively and exactly. Multi-artist
tracks match any shared artist, considering both `artist` and `artists`.
`fuzzy_artists` additionally uses beets' string
distance with the maximum `artist_threshold`; `bare_ascii_artists` additionally
compares transliterated, case-folded names with punctuation/whitespace removed.
When both modes are enabled, their results are unioned by item ID, so a pair is
scored once per refresh batch. Tracks with no artist do not match all other
unknown-artist tracks. Pairs outside the current artist scope are left untouched,
including explicitly computed cross-artist pairs; use `--add` to refresh those
or `--remove` to discard them. Automatic refreshes intentionally replace computed
scores without interactive prompts, but never replace manual choices.

## Performance and storage

`beet similar --size` reports allocated SQLite page bytes for the `similarity`
table and all its indexes, plus the refresh-state table when it exists; it does
not report the size of the entire beets database. It accepts
no query or threshold and does not compute or change scores. Even an empty table
and index occupy pages. The measurement includes unused space within those
pages, but excludes database-wide free pages, other tables, and journal/WAL files.
It requires SQLite's `dbstat` support (`SQLITE_ENABLE_DBSTAT_VTAB`); builds without
it produce an explicit error.

All-vs-all computation is **O(N²)**: `N * (N - 1) / 2` unordered pairs (two
directional distance calls per pair). At **50,000 tracks this is 1,249,975,000
pairs**, so always scope large-library commands by artist, genre, etc.
An unscoped computation, at least `warn_items` selected tracks, or at least
`warn_pairs` pairs logs a warning and asks to proceed (`--force` bypasses the
prompt, not the warning). Automatic batches log warnings for large candidate
sets but do not prompt.

Automatic refreshes read distinct stored `artist` and `artists` values, match
each unique target artist once, then use beets SQL queries to fetch only tracks
in that artist scope. Targets sharing the same artist credits reuse the result.
This applies to exact Unicode case-folding, fuzzy, and bare-ASCII matching:
there is no full-library track-model fallback. Distinct-value discovery still
reads the library's artist columns; fuzzy matching compares unique artist names,
not every track against every imported track.

The `fuzzy` and `bareasc` query plugins are not required. Their `~` and `#`
queries have different matching semantics: fuzzy queries use a similarity ratio
and Python-side filtering, while bare-ASCII queries use substring/smart-case
matching. Automatic similarity refreshes retain `artist_threshold` and the
exact normalized-name matching described above.

Storage is sparse: only computed pairs are saved, once per unordered pair, with
a composite primary key and a reverse-lookup index. A fully populated 50k matrix
has a **10 GB decimal raw float64 score payload alone**. With IDs, manual flags,
SQLite record/page overhead, and the reverse index, budget roughly **50–100 GB**
for this schema (an estimate, not a capacity guarantee; IDs/page utilization and
journaling affect actual disk usage). Allow additional space for backups and
journals. Artist-scoped storage is usually orders of magnitude smaller:
sum `n_artist * (n_artist - 1) / 2` rather than the whole-library square.
Removing rows frees SQLite pages for reuse but does not shrink the database
file; use normal SQLite maintenance/backups if disk reclamation is needed.
List output streams one row at a time; a large matrix still has O(N²) output.

## Integration API

Other plugins can use the enabled plugin instance:

```python
from beets import plugins
from beetsplug.similarity import SimilarityPlugin

similarity = next(p for p in plugins.find_plugins() if isinstance(p, SimilarityPlugin))
for other, score in similarity.similar_items(lib, item, threshold=0.8):
    # Read a rating or apply a playlist-specific quality tie-break here.
    pass

similarity.refresh_items(lib, [item])
```

`similar_items` returns `(Item, similarity)` pairs, highest score first, with
item-ID tie-breaking, and honors manual choices. It reads stored pairs without
computing new ones. Consumers control rating prompts, asynchronous review, and
quality tie-breaking;
The `similarity` SQL table stores `item_a < item_b`, base `score`, and `manual`
(`-1` dissociated, `0` automatic, `1` associated); effective similarity is
`1` for associated pairs, `max(0, score - dissociate_penalty)` for dissociated
pairs, and `score` otherwise. Set `dissociate_penalty` to a number in `[0, 1]`
to subtract a fixed penalty, or `to-zero` to use each pair's base score as the
penalty and force its effective score to `0`.

## Tests

From the repository root, after installing the package:

```sh
python -m unittest discover -s tests -v
```
