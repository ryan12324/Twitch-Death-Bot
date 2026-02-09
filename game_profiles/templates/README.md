# Template Images

Drop **cropped screenshots** of the death screen indicator into the game's
folder. The detector slides each template across the captured frame to find
it — like ctrl+F for images.

## What to screenshot

Crop **just the death indicator**, not the full screen:

| Game | What to crop |
|------|-------------|
| Elden Ring | The red "YOU DIED" text |
| Dark Souls II/III | The red "YOU DIED" text |
| Sekiro | The red death kanji characters |
| Hollow Knight | The black screen with the shade |
| Celeste | The white death burst |

## Rules

- **Crop tight** — just the text/icon, not the whole screen
- **Any filename** works (`.png`, `.jpg`, `.jpeg`, `.bmp`)
- **Multiple templates** are fine — more = better coverage of different scenarios
- The detector tries **5 different scales** (0.5x to 1.5x) so the template
  doesn't need to be the exact resolution of the stream
- Screenshots from any resolution work — 1080p, 720p, 4K, whatever

## Example

For Elden Ring, take a screenshot of the death screen, crop just the
"YOU DIED" text, and save it as:

```
game_profiles/templates/elden_ring/you_died.png
```

You can add multiple variants:
```
game_profiles/templates/elden_ring/you_died_1080p.png
game_profiles/templates/elden_ring/you_died_720p.png
game_profiles/templates/elden_ring/you_died_boss.png
```

## Quick capture

Use the built-in template creator to grab frames from a VOD:

```bash
python tools/create_templates.py --source gameplay.mp4 --game elden_ring
```

Press `S` on a death screen, then crop the saved image to just the
death indicator before putting it in the template folder.
