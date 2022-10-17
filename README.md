# osu2ragna

Convert beatmaps from the `Songs` folder from osu! into `CustomLevels` folder of RagnaRock.

## Playability
Surprisingly good. There are no hold notes, but the three 1.5\*~2.2\* maps plays like 6~8 hammers.

## Requirements
 - Python
 - Pillow (Python Imaging Library)
 - ffmpeg (in PATH)

## Usage
### Single beatmap
`python -m osu2ragna <BEATMAPSET_FOLDER> <CUSTOMLEVELS_FOLDER>`

e.g.:

`python -m osu2ragna D:\osu\Songs\"511480 t+pazolite - Oshama Scramble!" %USERPROFILE%\Documents\Ragnarock\CustomSongs`

`python -m osu2ragna D:\osu\Songs\"511480 t+pazolite - Oshama Scramble!" $Env:USERPROFILE\Documents\Ragnarock\CustomSongs`

### All beatmaps
`python -m osu2ragna --songs <OSU_SONGS_FOLDER> <CUSTOMLEVELS_FOLDER>`

e.g.:

`python -m osu2ragna --songs D:\osu\Songs %USERPROFILE%\Documents\Ragnarock\CustomSongs`

`python -m osu2ragna --songs D:\osu\Songs $Env:USERPROFILE\Documents\Ragnarock\CustomSongs`

