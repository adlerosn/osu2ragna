#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# pylint: disable=subprocess-run-check
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import PIL.Image  # type: ignore

from .generictools import (IntSpanDict, avg, distance_of_incresing_values,
                           filter_ascii_alnum, flatten, linear_clusterization,
                           pick_the_largest_sublist)
from .osutools import (OsuFileSomeMetadata, OsuHitObject, OsuModesEnum,
                       get_audio_file_from_section,
                       get_osu_hit_objects_from_section, get_osu_sections,
                       get_some_metadata_from_section)
from .ragnatools import (HandEnum, NoteLineIndexEnum, RagnaRockDifficultyV1,
                         RagnaRockHandsPositionsSimulator,
                         RagnaRockInfoButDifficulties)

PREVIEW_DURATION = 10
RGX_OSU_SCHEMA = re.compile(r'(\d+) (.+?) - (.+)')
BEATMAPSET_RAGNA_BORKED_MTIME = -1
BEATMAPSET_RAGNA_BORKED_PATH = Path()
BEATMAPSET_RAGNA_BORKED_DATA: Dict[str, str] = {}
PREVIEW_OGG = 'preview.ogg'
SONG_OGG = 'song.ogg'
COVER_JPG = 'cover.jpg'
INFO_DAT = 'info.dat'


def beatmapset_ragna_borked_init(file: Path) -> int:
    if not file.exists():
        file.touch(exist_ok=True)
    return file.stat().st_mtime_ns


def beatmapset_ragna_borked_load(file: Path) -> bool:
    global BEATMAPSET_RAGNA_BORKED_MTIME
    global BEATMAPSET_RAGNA_BORKED_PATH
    global BEATMAPSET_RAGNA_BORKED_DATA
    if (mt := beatmapset_ragna_borked_init(file)) != BEATMAPSET_RAGNA_BORKED_MTIME or file != BEATMAPSET_RAGNA_BORKED_PATH:
        BEATMAPSET_RAGNA_BORKED_MTIME = mt
        BEATMAPSET_RAGNA_BORKED_PATH = file
        BEATMAPSET_RAGNA_BORKED_DATA = dict(filter(lambda a: len(a) == 2, map(lambda a: a.split(
            ';', 1), filter(len, map(str.strip, file.read_text(encoding='utf-8').splitlines())))))
        return True
    return False


def beatmapset_ragna_borked_add(k: str, v: str):
    global BEATMAPSET_RAGNA_BORKED_MTIME
    global BEATMAPSET_RAGNA_BORKED_PATH
    global BEATMAPSET_RAGNA_BORKED_DATA
    if k not in BEATMAPSET_RAGNA_BORKED_DATA or BEATMAPSET_RAGNA_BORKED_DATA[k] != v:
        BEATMAPSET_RAGNA_BORKED_DATA[k] = v
        BEATMAPSET_RAGNA_BORKED_PATH.write_text(
            '\n'.join(f'{k};{v}' for k, v in
                      BEATMAPSET_RAGNA_BORKED_DATA.items()),
            encoding='utf-8',
        )
        BEATMAPSET_RAGNA_BORKED_MTIME = BEATMAPSET_RAGNA_BORKED_PATH.stat().st_mtime_ns


def beatmapset_ragna_borked_get(k: str) -> Optional[str]:
    global BEATMAPSET_RAGNA_BORKED_DATA
    return BEATMAPSET_RAGNA_BORKED_DATA.get(k, None)


def get_parser():
    parser = argparse.ArgumentParser(
        prog=f'{Path(sys.executable).stem} -m {Path(__file__).parent.name}')
    parser.add_argument('osu_beatmapset_path', type=Path, metavar='OSU_SONG',
                        help='The folder inside your Songs folder that holds the audio file and .osu files')
    parser.add_argument('--songs', action='store_const',
                        default=False, const=True, help='if OSU_SONG is the Songs folder and all songs should be processed in batch')
    parser.add_argument('ragna_customsongs_path', type=Path,
                        metavar='RAGNA_CUSTOMSONGS',
                        default=None, nargs='?',
                        help='The CustomSongs folder for Raganarock')
    parser.add_argument('--overwrite', action='store_const',
                        const=True, default=False, help='Overwrite already processed beatmaps (default: skip)')
    return parser


def osu_folder_scheme_parser(name: str) -> Tuple[int, str, str]:
    if (match := RGX_OSU_SCHEMA.match(name)) is not None:
        a = list(match.groups())
        a[0] = int(a[0])
        return tuple(a)  # type: ignore
    raise ValueError('Regular expression did not match')


def osu_folder_scheme_parser_safer(name: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    try:
        return osu_folder_scheme_parser(name)
    except Exception:
        return None, None, None


def main():
    args = get_parser().parse_args()
    path: Path = args.osu_beatmapset_path
    if not args.songs:
        bmsid, artist, title = osu_folder_scheme_parser_safer(path.name)
        sbr = args.ragna_customsongs_path or path.parent
        convert_osu2ragna(
            path,
            sbr.joinpath(f'{artist}{title}osu{bmsid}'),
            bmsid, artist, title,
            args.overwrite,
        )
    else:
        for pth in list(path.glob('*')):
            if pth.is_dir():
                bmsid, artist, title = osu_folder_scheme_parser_safer(pth.name)
                if bmsid:
                    sbr = args.ragna_customsongs_path
                    o = sbr.joinpath(f'{artist}{title}osu{bmsid}')
                    print(o.name)
                    convert_osu2ragna(
                        pth,
                        o,
                        bmsid, artist, title,
                        args.overwrite,
                    )


def convert_osu2ragna(beatmapset_osu: Path,
                      beatmapset_ragna: Path,
                      osu_beatmapset_id_default: int,
                      osu_artist_default: str,
                      osu_title_default: str,
                      overwriting: bool,
                      ) -> bool:
    if not beatmapset_osu.is_dir():
        raise NotADirectoryError(beatmapset_osu)
    beatmapset_ragna_borked_load(
        beatmapset_ragna.parent.joinpath('borked.txt'))
    if beatmapset_ragna_borked_get(str(osu_beatmapset_id_default)):
        return True
    beatmapset_ragna_number = 1
    beatmapset_ragna_numbered = beatmapset_ragna.with_name(filter_ascii_alnum(
        f'{beatmapset_ragna.name}p{beatmapset_ragna_number}').lower())
    if beatmapset_ragna_numbered.exists():
        if not beatmapset_ragna_numbered.is_dir():
            raise FileExistsError(beatmapset_ragna_numbered)
        elif (
            beatmapset_ragna_numbered.joinpath(PREVIEW_OGG).is_file() and
            beatmapset_ragna_numbered.joinpath(SONG_OGG).is_file() and
            beatmapset_ragna_numbered.joinpath(INFO_DAT).is_file() and
            not overwriting
        ):
            return True
    osu_beatmap_paths: List[Path] = list(beatmapset_osu.glob('*.osu'))
    if len(osu_beatmap_paths) <= 0:
        raise FileNotFoundError(beatmapset_osu / '*.osu')
    beatmapset_ragna_numbered.mkdir(parents=True, exist_ok=True)
    # Abort on known failures
    if beatmapset_ragna_borked_get(str(osu_beatmapset_id_default)):
        for x in beatmapset_ragna_numbered.glob('*'):
            x.unlink()
            del x
        beatmapset_ragna_numbered.rmdir()
        return True
    # Load osu beatmaps in this set
    osu_beatsets: List[Tuple[OsuFileSomeMetadata, List[OsuHitObject]]] = (
        read_beats_osu(beatmapset_osu, beatmapset_ragna_numbered, osu_beatmap_paths,
                       osu_beatmapset_id_default, osu_artist_default, osu_title_default))
    # filter out non-mania4k
    osu_beatsets = [obs for obs in osu_beatsets if obs[0].mode ==
                    OsuModesEnum.Mania and obs[0].circle_size == 4]
    if osu_beatsets is None or len(osu_beatsets) <= 0:
        for x in beatmapset_ragna_numbered.glob('*'):
            x.unlink()
            del x
        beatmapset_ragna_numbered.rmdir()
        return True
    # Abort on known failures [yes, again]
    if osu_beatsets is None or len(osu_beatsets) <= 0:
        for x in beatmapset_ragna_numbered.glob('*'):
            x.unlink()
            del x
        beatmapset_ragna_numbered.rmdir()
        return True
    # Merging metadata
    osu_metadata_merged = OsuFileSomeMetadata.merge(
        next(zip(*osu_beatsets)))  # type: ignore
    make_audio_preview(beatmapset_ragna_numbered.joinpath(SONG_OGG),
                       beatmapset_ragna_numbered.joinpath(PREVIEW_OGG),
                       osu_metadata_merged.preview_start/1000,
                       PREVIEW_DURATION,
                       )
    audio_duration = probe_audio_duration(
        beatmapset_ragna_numbered.joinpath(SONG_OGG))
    # Edge case: Ragnarock only supports 3 difficulties per beatmapset
    beatmapset_ragna_number_total = math.ceil(len(osu_beatsets) / 3)
    for i in range(2, beatmapset_ragna_number_total+1):
        j = beatmapset_ragna.with_name(filter_ascii_alnum(
            f'{beatmapset_ragna.name}p{i}').lower())
        j.mkdir(parents=True, exist_ok=True)
        j.joinpath(SONG_OGG).write_bytes(
            beatmapset_ragna_numbered.joinpath(SONG_OGG).read_bytes())
        j.joinpath(PREVIEW_OGG).write_bytes(
            beatmapset_ragna_numbered.joinpath(PREVIEW_OGG).read_bytes())
        j.joinpath(COVER_JPG).write_bytes(
            beatmapset_ragna_numbered.joinpath(COVER_JPG).read_bytes())
        del j
        del i
    # convert osu_beatsets into ragna_beatsets
    ragna_beatsets: Tuple[RagnaRockInfoButDifficulties,
                          List[RagnaRockDifficultyV1]] = convert_beatsets_osu2ragna(osu_beatsets, audio_duration)
    # sort ragna_beatsets by difficulty
    ragna_beatsets[1].sort(key=lambda a: (
        a.difficulty_rankf,
        len(a.notes),
        a.difficulty_label))
    # write converted beatmap DATs into CustomSongs
    for i in range(1, beatmapset_ragna_number_total+1):
        (ragna_beatsets[0]
            .with_song_sub_name(f'[{i} of {beatmapset_ragna_number_total}]' if beatmapset_ragna_number_total > 1 else '')
            .with_difficulties(ragna_beatsets[1][i-1::beatmapset_ragna_number_total])
            .write_to(beatmapset_ragna.with_name(filter_ascii_alnum(f'{beatmapset_ragna.name}p{i}').lower())))
    return False


def convert_beatsets_osu2ragna(osu_beatsets: List[Tuple[OsuFileSomeMetadata, List[OsuHitObject]]],
                               audio_duration: float
                               ) -> Tuple[RagnaRockInfoButDifficulties, List[RagnaRockDifficultyV1]]:
    osu_metadata_merged = OsuFileSomeMetadata.merge(
        next(zip(*osu_beatsets)))  # type: ignore
    print(f'  ~> {osu_metadata_merged}')
    ragna_attention_pointss: List[IntSpanDict[OsuHitObject]] = []
    for osu_metadata, osu_hit_objects in osu_beatsets:
        osu_hit_objects = osu_hit_objects.copy()
        osu_hit_objects.sort(key=OsuHitObject.start_time)
        ragna_attention_points: IntSpanDict[OsuHitObject] = IntSpanDict()
        ragna_attention_pointss.append(ragna_attention_points)
        for osu_hit_object in osu_hit_objects:
            for sub_hit_obj in osu_hit_object.derive_pretending_hitcircle():
                ragna_attention_points.append_point(
                    round(sub_hit_obj.start_time()*1000),
                    sub_hit_obj)
                del sub_hit_obj
            del osu_hit_object
        del osu_metadata
    ragnainfo_without_difficulties = RagnaRockInfoButDifficulties(
        osu_metadata_merged.title,
        '',
        osu_metadata_merged.artist,
        osu_metadata_merged.creator,
        figure_out_bpm(ragna_attention_pointss, 50, 240),
        songApproximativeDuration=audio_duration,
        previewStartTime=osu_metadata_merged.preview_start/1000,
        previewDuration=max(0.1, min(PREVIEW_DURATION,
                                     audio_duration - osu_metadata_merged.preview_start/1000)),
        customData=dict(
            generator='osu2ragna',
            source='osu!',
            beatmapID=osu_metadata_merged.beatmap_id,
            beatmapSetID=osu_metadata_merged.beatmap_set_id,
            link=f'https://osu.ppy.sh/beatmaps/{osu_metadata_merged.beatmap_id}',
            link2=f'https://osu.ppy.sh/beatmapsets/{osu_metadata_merged.beatmap_set_id}',
            link3=f'https://chimu.moe/d/{osu_metadata_merged.beatmap_set_id}',
        )
    )
    round_to_beat = ragnainfo_without_difficulties.round_to_beat
    conv_to_beat = ragnainfo_without_difficulties.convert_to_beat
    ragna_bpmd_attention_pointss: List[IntSpanDict[OsuHitObject]] = []
    for ragna_attention_points in ragna_attention_pointss:
        ragna_bpmd_attention_pointss.append(
            ragna_attention_points.map_keys(round_to_beat))
        del ragna_attention_points
    del ragna_attention_pointss
    ragna_difficulties: List[RagnaRockDifficultyV1] = []
    for (osu_metadata, _), ragna_bpmd_attention_points in zip(osu_beatsets, ragna_bpmd_attention_pointss):
        difficulty = osu_metadata.difficulty
        if osu_metadata.mode != OsuModesEnum.Mania and osu_metadata.circle_size != 4:
            continue
        del _
        hands_pos_sim = RagnaRockHandsPositionsSimulator(
            ragnainfo_without_difficulties)
        ragna_bpmd_points_of_interest = ragna_bpmd_attention_points.points_of_interest()
        combos: List[Tuple[int, int]] = []
        LPOI = 0
        hand_moves = 0
        for point_of_interest in ragna_bpmd_points_of_interest:
            active_hand_goals: List[Tuple[NoteLineIndexEnum, OsuHitObject]] = [
                (
                    NoteLineIndexEnum(
                        max(0, min(int(osu_metadata.circle_size or 4)-1,
                                   math.floor(
                            active_point_of_interest.coord_x() *
                            int(osu_metadata.circle_size or 4) /
                            512
                        )))
                    ),
                    active_point_of_interest)
                for active_point_of_interest in sorted(
                    ragna_bpmd_attention_points.active_at_point(
                        point_of_interest),
                    key=lambda a: (a.start_time(), a.new_combo(), a.is_kiai())
                )
            ]
            preferred_hand = HandEnum((len(combos)+hand_moves) % 2)
            least_preferred_hand = HandEnum((preferred_hand.value+1) % 2)
            hand_moves += hands_pos_sim.move_to(
                point_of_interest,
                (preferred_hand, least_preferred_hand),
                [
                    (id(apoi), round_to_beat(apoi.start_time()*1000), canvas)
                    for canvas, apoi in active_hand_goals
                ],
            )
            del preferred_hand
            del least_preferred_hand
            del active_hand_goals
            LPOI = point_of_interest
            del point_of_interest
        del hand_moves
        del LPOI
        ragna_difficulties.append(hands_pos_sim.build_coreography())
        ragna_difficulties[-1].difficulty_label = difficulty
        print(
            f'      +-> {len(hands_pos_sim.coreography_ids):6d} - {ragna_difficulties[-1].difficulty_rankf:09.6f} - {ragna_difficulties[-1].difficulty_rank:02d} - {difficulty}')
        del osu_metadata
        del ragna_bpmd_attention_points
        del difficulty
        del hands_pos_sim
    del conv_to_beat
    del round_to_beat
    del ragna_bpmd_attention_pointss
    return (ragnainfo_without_difficulties, ragna_difficulties)


def figure_out_bpm(ragna_attention_pointss: List[IntSpanDict[OsuHitObject]], mn: float, mx: float) -> float:
    time_distance_between_notes: List[int] = sorted(
        flatten(map(distance_of_incresing_values,
                    map(IntSpanDict.points_of_interest,
                        filter(lambda a: a.retain_value(OsuHitObject.is_point),
                               ragna_attention_pointss
                               )))))
    time_distance_between_notes = list(
        filter(lambda a: 50 <= a <= 600, time_distance_between_notes))
    clusters_time_distance_between_notes = linear_clusterization(
        time_distance_between_notes, 5)
    del time_distance_between_notes
    cluster_time_distance_between_notes = pick_the_largest_sublist(
        clusters_time_distance_between_notes)
    del clusters_time_distance_between_notes
    if len(cluster_time_distance_between_notes) <= 0:
        return 120
    avg_cluster_time_distance_between_notes = avg(
        cluster_time_distance_between_notes)
    del cluster_time_distance_between_notes
    bpm = 60000/(1*avg_cluster_time_distance_between_notes)
    if bpm <= 0:
        return 120
    while bpm < mn:
        bpm *= 2
    while bpm > mx:
        bpm /= 2
    return bpm


def read_beats_osu(beatmapset_osu: Path, beatmapset_ragna: Path, osu_beatmap_paths: List[Path],
                   osu_beatmapset_id_default: int, osu_artist_default: str, osu_title_default: str
                   ) -> List[Tuple[OsuFileSomeMetadata, List[OsuHitObject]]]:
    osu_loaded_stuff: List[Tuple[OsuFileSomeMetadata,
                                 List[OsuHitObject]]] = []
    for osu_beatmap_path in osu_beatmap_paths:
        osu_beatmap_text = osu_beatmap_path.read_text('utf-8', errors='ignore'
                                                      ).replace('\r\n', '\n').replace('\r', '')
        osu_beatmap_sections: Dict[str, List[str]
                                   ] = get_osu_sections(osu_beatmap_text)
        del osu_beatmap_text
        metadata: OsuFileSomeMetadata = get_some_metadata_from_section(
            osu_beatmap_sections)
        metadata.title = metadata.title or osu_title_default
        metadata.artist = metadata.artist or osu_artist_default
        metadata.beatmap_set_id = metadata.beatmap_set_id or osu_beatmapset_id_default
        osu_hit_objects = get_osu_hit_objects_from_section(
            osu_beatmap_sections)
        if len(osu_hit_objects) > 0 and metadata.mode == OsuModesEnum.Mania and metadata.circle_size == 4:
            print(f'  |> {metadata.mode.name} ~ {osu_beatmap_path.stem}')
            ragna_beatmap_audio = beatmapset_ragna.joinpath(SONG_OGG)
            if convert_audio_osu2ragna(osu_beatmapset_id_default, beatmapset_osu, ragna_beatmap_audio, osu_beatmap_sections):
                return []
            ragna_beatmap_thumb = beatmapset_ragna.joinpath(COVER_JPG)
            if convert_thumb_osu2ragna(beatmapset_osu, ragna_beatmap_thumb, osu_beatmap_sections):
                return []
            osu_loaded_stuff.append((metadata, osu_hit_objects))
        del osu_beatmap_path
    if len(osu_loaded_stuff) <= 0:
        beatmapset_ragna.joinpath(SONG_OGG).unlink(missing_ok=True)
        beatmapset_ragna_borked_add(
            str(osu_beatmapset_id_default), 'no_eligible_osu_files')
    return osu_loaded_stuff


def convert_thumb_osu2ragna(beatmapset_osu: Path, ragna_beatmap_thumb: Path, osu_beatmap_sections: Dict[str, List[str]]) -> bool:
    if not ragna_beatmap_thumb.exists():
        imageline = next(filter(lambda x: x.startswith('0,0,'),
                                osu_beatmap_sections.get('Events', [])), None)
        im = PIL.Image.new('RGB', (512, 512), 0)
        if imageline is not None:
            osu_bg = beatmapset_osu.joinpath(
                imageline.split(',')[2].strip('"'))
            if osu_bg.exists():
                im = PIL.Image.open(str(osu_bg)).convert('RGB')
                im = im_crop_square_center(im)
                im.thumbnail((512, 512))
        im.save(str(ragna_beatmap_thumb))
    return False


def im_crop_square_center(im: PIL.Image.Image) -> PIL.Image.Image:
    sx, sy = im.size
    if sx != sy:
        d = min(sx, sy)
        if sx == d:
            # L T R B
            im = im.crop((
                0,
                int(sy/2 - d/2),
                d,
                int(sy/2 + d/2),
            ))
        else:
            # L T R B
            im = im.crop((
                int(sx/2 - d/2),
                0,
                int(sx/2 + d/2),
                d,
            ))
    return im


def convert_audio_osu2ragna(osu_beatmapset_id_default: int, beatmapset_osu: Path, ragna_beatmap_audio: Path, osu_beatmap_sections: Dict[str, List[str]]) -> bool:
    if not ragna_beatmap_audio.exists():
        ragna_beatmap_audio_tmp = ragna_beatmap_audio.with_suffix('.egg')
        osu_beatmap_audio = beatmapset_osu.joinpath(
            get_audio_file_from_section(osu_beatmap_sections))
        r = subprocess.run(
            ['ffmpeg', '-y',
                '-v', 'quiet',
                '-i', str(osu_beatmap_audio),
                '-q', '9',
                '-map_metadata', '-1',
                '-vn', '-acodec', 'libvorbis',
                '-f', 'ogg',
                str(ragna_beatmap_audio_tmp),
             ],
        )
        if r.returncode:
            beatmapset_ragna_borked_add(
                str(osu_beatmapset_id_default), 'broken_audio')
            if ragna_beatmap_audio_tmp.exists():
                ragna_beatmap_audio_tmp.unlink()
            return True
        del r
        ragna_beatmap_audio_tmp.rename(ragna_beatmap_audio)
        del ragna_beatmap_audio_tmp
        del osu_beatmap_audio
    del ragna_beatmap_audio
    return False


def make_audio_preview(audio: Path, preview: Path, start: float, duration: float):
    if not preview.exists():
        audio_duration = probe_audio_duration(audio)
        start = min(max(0.0, start), audio_duration-duration/2)
        fade_duration = 0.075
        actual_duration = max(0.0, min(duration, audio_duration-start))
        if actual_duration <= 2.0:
            fade_duration = 0.0
        preview_tmp = preview.with_suffix('.egg')
        subprocess.run(
            ['ffmpeg', '-y',
                '-v', 'quiet',
                '-ss', str(start),
                '-t', str(duration),
                '-i', str(audio),
                '-filter:a', f'afade=t=in:st=0:d={fade_duration},afade=t=out:st={actual_duration-fade_duration}:d={fade_duration}',
                '-q', '5',
                '-map_metadata', '-1',
                '-vn', '-acodec', 'libvorbis',
                '-f', 'ogg',
                str(preview_tmp),
             ],
            check=True,
        )
        preview_tmp.rename(preview)


def probe_audio_duration(audio: Path) -> float:
    if not audio.exists():
        return 0.0
    r = subprocess.run(
        ['ffprobe',
            '-v', 'quiet',
            '-show_format',
            '-print_format', 'json',
            str(audio),
         ],
        check=True,
        stdout=subprocess.PIPE
    )
    return float(json.loads(r.stdout)['format']['duration'])
