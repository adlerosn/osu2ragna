# osu2ragna

Convert beatmaps from the `Songs` folder from osu!mania into `CustomLevels` folder of RagnaRock.

## Compatibility
Only `osu!mania 4k` beatmaps.

## Playability
Surprisingly good.

Hammer-oriented difficulties are now calculated during conversion based on a regression done over [84 ranked beatmapsets at RagnaCustoms](https://ragnacustoms.com/song-library?search=&downloads_submitted_date=&downloads_filter_difficulties=&converted_maps=&wip_maps=&only_ranked=1&search_btn=1). They are based on a hypothetical strain that playing a beatmap would bring to your arms.

Some medleys with 25+ minutes may exceed 50 hammers - they're as hard as it sounds.

## Requirements
 - Python
 - Pillow (Python Imaging Library)
 - ffmpeg (the executable, accessible from PATH)

## Usage
### Single beatmap
`python -m osu2ragna <BEATMAPSET_FOLDER> <CUSTOMLEVELS_FOLDER>`

e.g.:

(CMD) `python -m osu2ragna D:\osu\Songs\"511480 t+pazolite - Oshama Scramble!" %USERPROFILE%\Documents\Ragnarock\CustomSongs`

(PS) `python -m osu2ragna D:\osu\Songs\"511480 t+pazolite - Oshama Scramble!" $Env:USERPROFILE\Documents\Ragnarock\CustomSongs`

### All beatmaps
`python -m osu2ragna --songs <OSU_SONGS_FOLDER> <CUSTOMLEVELS_FOLDER>`

e.g.:

(CMD) `python -m osu2ragna --songs D:\osu\Songs %USERPROFILE%\Documents\Ragnarock\CustomSongs`

(PS) `python -m osu2ragna --songs D:\osu\Songs $Env:USERPROFILE\Documents\Ragnarock\CustomSongs`

