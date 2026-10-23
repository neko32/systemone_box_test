# neko-box 🐈📦

Finds the cat in a video using the **Box** primitive of [TypeSafe AI System One](https://docs.typesafe.ai/primitives/box) (early-adopter / Dreamer-ready).

Every second, OpenCV takes a snapshot of the playing video and asks System One _"猫はどこですか"_ ("Where is the cat?").
The returned box is drawn onto the video.
Because Box always returns a box even when nothing is there, the same request also asks a yes/no **Noul** question ("Is there a cat?") to decide whether to draw it.

```
frame ─▶ every 1 s ─▶ resize + JPEG ─▶ System One { Noul: cat present?, Box: where? } ─▶ draw box ─▶ window + mp4
```

## Quick start

```bash
cp .env.example .env               # set API_KEY_JEV=<your TypeSafe key>
uv sync
uv run neko-box                    # live: play + async detection
uv run neko-box --mode offline     # detect, write out/neko_boxed.mp4, then play it
uv run neko-box --interval 0.5 -v  # snapshot twice a second, debug logs
uv run neko-box --input movie/many_cats.mp4 --max-cats 5  # several cats at once
```

Or use `scripts_local/run.ps1` (Windows) or `scripts_local/run.sh` (Linux).

## Several cats (`--max-cats N`)

Box returns one box per question and ignores instructions like "the 2nd cat from the left".
So neko-box first counts the cats with Noul ("at least k cats?" for k = 1..N), then repeatedly paints the cats it has found grey and asks Box for "1匹の猫" ("one cat") again until the count is reached.
Asking for exactly *one* cat matters: the Japanese "猫はどこですか" has no singular or plural, so Box boxed the whole group.

## Results on `movie/neko.mp4`

- 150–250 ms per request, about 2.7k input tokens per snapshot
- Cat present: Noul 0.90–0.96. No cat: Noul 0.04–0.05. The two cases are cleanly separated.
- The boxes match the cat well, including a cat peeking from behind the sofa, a cat behind a curtain, and a close-up face.

## Development

```bash
scripts_local/test.sh               # ruff + format check + unit tests with coverage (≥90%)
uv run pytest -m integration        # real API (needs API_KEY_JEV)
```

See [docs/design.md](docs/design.md), [docs/api.md](docs/api.md), and [docs/dependencies.html](docs/dependencies.html).

## License

MIT
