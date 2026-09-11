# snapshot: the review web tool, before the port

A verbatim copy of the review web tool this package was extracted from, brought in so the port
work has the real code to carve from rather than a description of it. **It is reference material,
not part of the package**: nothing imports it, ruff skips it (`extend-exclude`), pytest never
collects it (`testpaths = ["tests"]`), and it is not in the wheel.

Module paths were rewritten so they read consistently here: `snapshot.review`, `snapshot.tools` and
`snapshot.vision` are the files in this folder; `parent.` marks a dependency that stays behind in
the parent project and was not copied.

**The last port card deletes this folder.** A module moved into `smolsmort/` is removed from here in
the same change, so what is left is always what is still to do.

## What is here

| folder | files | becomes |
|---|---|---|
| `review/` | `paths`, `naming`, `playback_state`, `splits`, `housekeeping`, `train` | carried across nearly as-is - none of them depends on the parent project |
| `review/` | `state`, `routes`, `classdefs` | split: the generic half moves behind the plugin seams, the domain half stays behind |
| `tools/` | `playback` | the frame playback the find tab reads through |
| `vision/` | `plate_templates`, `recentre` | template matching: the first *example source* behind its seam |
| `ui/` | `app.js`, `index.html`, `tabs.css` | the find, select, train, load, settings and housekeeping tabs |
| `tests/` | seven test files | the behaviour the port has to keep |

## What stays behind

Anything that encodes a fact about the domain it was built for rather than about finding and
judging things: the fixed class list and the colour-based class guesser (the *class scheme* seam
replaces both), screen profiles, and the interface, clusters, vlm, scripts, map, navigation, camera
and control tabs in `app.js`.

## Conventions that carry over

- **File naming and storage stay as they are** - boxes, tiles, sets, weights and the settings file
  keep their names and layout, so data already on disk opens unchanged.
- **Paths are referenced as attributes** (`paths.SESSIONS_DIR`, never imported by value). A test
  enforces it: by-value imports once let the suite read and write real recordings while passing.
- **No new runtime dependencies**, and `uv.lock` unchanged: the offline test gate runs in an image
  built from the lock. The server is the standard library's `http.server`.
- **Torch stays optional**, and every measured default keeps its provenance.
