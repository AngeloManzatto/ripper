# RIPPER — Changelog

What changed and when, oldest first. The ideas themselves live in `docs/IDEAS.md`; this file only records history. Add a line whenever the engine, a decision or an idea changes.

- **2026-11-20** — File created from the eight ideas of the walk, plus additions and references.
- **2026-11-20** — I-07: the reference is Rain World (it had been written down as "Cat World"); added what that kind of ecosystem would ask of the engine.
- **2026-11-20** — New idea I-09 (a learned generator for entity symbols); I-02 now points to it.
- **2026-11-21** — Engine reorganized into layers (`core`, `content`, `packs`, `world`, `render`); `Registry` became `Content` with a generic `Table<D>`; first `Level` and first terminal picture. Foundations table and dependency picture updated; engine map and decisions added; the entity bridge dropped; parking lot extended.
- **2026-11-22** — Colour plan agreed: RGB as hex strings, foreground and background required on terrain, a `Frame` of cells as the common source for terminal, image and Python outputs. I-02 gains transparent pixels and the palette thought.
- **2026-11-22** — Colour in the data pipeline is complete: `Color` (hex strings), required `fg_color` and `bg_color` on terrain, checked by the validator and parsed in build; shared test fixtures. `Cell` and `Frame` (`Grid<Cell>`) created.
- **2026-11-23** — Vocabulary settled: *paint* writes cells into a frame, *render* turns a finished frame into an output. `paint_terrain` and `paint_level`, `frame_to_text` and `frame_to_ansi` added; the old `draw_terrain` was removed and its tests moved. `examples/show_level` now prints a coloured map: the first coloured picture on screen.
- **2026-11-23** — The ideas registry and the changelog were split into two files. New idea I-10 (a collision mask on entities). The foundations table, engine map and parking lot were brought up to date.
