#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# pylint: disable=subprocess-run-check
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

import argparse
import math
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import PIL.Image  # type: ignore

from .generictools import (IntSpanDict, avg, distance_of_incresing_values, filter_ascii_alnum,
                           flatten, linear_clusterization,
                           pick_the_largest_sublist)
from .osutools import (OsuFileSomeMetadata, OsuHitObject, OsuModesEnum,
                       get_audio_file_from_section,
                       get_osu_hit_objects_from_section, get_osu_sections,
                       get_some_metadata_from_section)
from .ragnatools import (RagnaRockDifficultyV1,
                         RagnaRockHandsPositionsSimulator,
                         RagnaRockInfoButDifficulties, NoteLineIndexEnum,
                         HandEnum)

RGX_OSU_SCHEMA = re.compile(r'(\d+) (.+?) - (.+)')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('osu_beatmapset_path', type=Path,
                        help='The folder inside your Songs folder that holds the audio file and .osu files')
    parser.add_argument('--songs', action='store_const',
                        default=False, const=True)
    parser.add_argument('ragna_customsongs_path', type=Path, default=None, nargs='?',
                        help='The CustomSongs folder for Raganarock')
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
            bmsid, artist, title
        )
    else:
        for pth in list(path.glob('*')):
            if pth.is_dir():
                bmsid, artist, title = osu_folder_scheme_parser_safer(pth.name)
                if bmsid:
                    sbr = args.ragna_customsongs_path
                    o = sbr.joinpath(f'{artist}{title}osu{bmsid}')
                    print(o.name)
                    convert_osu2ragna(pth, o, bmsid, artist, title)


def convert_osu2ragna(beatmapset_osu: Path,
                      beatmapset_ragna: Path,
                      osu_beatmapset_id_default: int,
                      osu_artist_default: str,
                      osu_title_default: str) -> bool:
    if not beatmapset_osu.is_dir():
        raise NotADirectoryError(beatmapset_osu)
    beatmapset_ragna_number = 1
    beatmapset_ragna_numbered = beatmapset_ragna.with_name(filter_ascii_alnum(
        f'{beatmapset_ragna.name}p{beatmapset_ragna_number}').lower())
    if beatmapset_ragna_numbered.exists() and not beatmapset_ragna_numbered.is_dir():
        raise FileExistsError(beatmapset_ragna_numbered)
    osu_beatmap_paths: List[Path] = list(beatmapset_osu.glob('*.osu'))
    if len(osu_beatmap_paths) <= 0:
        raise FileNotFoundError(beatmapset_osu / '*.osu')
    beatmapset_ragna_numbered.mkdir(parents=True, exist_ok=True)
    # Abort on known failures
    if beatmapset_ragna_numbered.joinpath('broken_audio.flag').exists():
        return True
    if beatmapset_ragna_numbered.joinpath('no_eligible_osu_files.flag').exists():
        return True
    # Load osu beatmaps in this set
    osu_beatsets: List[Tuple[OsuFileSomeMetadata, List[OsuHitObject]]] = (
        read_beats_osu(beatmapset_osu, beatmapset_ragna_numbered, osu_beatmap_paths,
                       osu_beatmapset_id_default, osu_artist_default, osu_title_default))
    # filter out non-mania4k
    osu_beatsets = [obs for obs in osu_beatsets if obs[0].mode ==
                    OsuModesEnum.Mania and obs[0].circle_size == 4]
    # Abort on known failures [yes, again]
    if osu_beatsets is None:
        return True
    if len(osu_beatsets) <= 0:
        return True
    # Edge case: Ragnarock only supports 3 difficulties per beatmapset
    beatmapset_ragna_number_total = math.ceil(len(osu_beatsets) / 3)
    for i in range(2, beatmapset_ragna_number_total+1):
        j = beatmapset_ragna.with_name(filter_ascii_alnum(
            f'{beatmapset_ragna.name}p{i}').lower())
        j.mkdir(parents=True, exist_ok=True)
        j.joinpath('song.egg').write_bytes(
            beatmapset_ragna_numbered.joinpath('song.egg').read_bytes())
        j.joinpath('cover.jpg').write_bytes(
            beatmapset_ragna_numbered.joinpath('cover.jpg').read_bytes())
        del j
        del i
    # convert osu_beatsets into ragna_beatsets
    ragna_beatsets: Tuple[RagnaRockInfoButDifficulties,
                          List[RagnaRockDifficultyV1]] = convert_beatsets_osu2ragna(osu_beatsets)
    # sort ragna_beatsets by difficulty
    ragna_beatsets[1].sort(key=lambda a: (
        len(a.notes),
        a.difficulty_label))
    # write converted beatmap DATs into CustomSongs
    for i in range(1, beatmapset_ragna_number_total+1):
        (ragna_beatsets[0]
            .with_song_sub_name(f'[{i} of {beatmapset_ragna_number_total}]' if beatmapset_ragna_number_total > 1 else '')
            .with_difficulties(ragna_beatsets[1][i-1::beatmapset_ragna_number_total])
            .write_to(beatmapset_ragna.with_name(filter_ascii_alnum(f'{beatmapset_ragna.name}p{i}').lower())))
    return False


def convert_beatsets_osu2ragna(osu_beatsets: List[Tuple[OsuFileSomeMetadata, List[OsuHitObject]]]
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
            f'      +-> {len(hands_pos_sim.coreography_ids):6d} {difficulty}')
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
        return 120.0
    avg_cluster_time_distance_between_notes = avg(
        cluster_time_distance_between_notes)
    del cluster_time_distance_between_notes
    bpm = 60000/(1*avg_cluster_time_distance_between_notes)
    if bpm <= 0:
        return 120.0
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
        if len(osu_hit_objects) > 0 and metadata.mode != OsuModesEnum.Taiko:
            print(f'  |> {metadata.mode.name} ~ {osu_beatmap_path.stem}')
            ragna_beatmap_audio = beatmapset_ragna.joinpath('song.egg')
            if convert_audio_osu2ragna(beatmapset_osu, ragna_beatmap_audio, osu_beatmap_sections):
                return []
            ragna_beatmap_thumb = beatmapset_ragna.joinpath('cover.jpg')
            if convert_thumb_osu2ragna(beatmapset_osu, ragna_beatmap_thumb, osu_beatmap_sections):
                return []
            osu_loaded_stuff.append((metadata, osu_hit_objects))
        del osu_beatmap_path
    if len(osu_loaded_stuff) <= 0:
        beatmapset_ragna.joinpath('song.egg').unlink(missing_ok=True)
        beatmapset_ragna.joinpath(
            'no_eligible_osu_files.flag').write_text(
            'no_eligible_osu_files')
    return osu_loaded_stuff


def convert_thumb_osu2ragna(beatmapset_osu: Path, ragna_beatmap_thumb: Path, osu_beatmap_sections: Dict[str, List[str]]) -> bool:
    if ragna_beatmap_thumb.exists():
        return False
    imageline = next(filter(lambda x: x.startswith('0,0,'),
                            osu_beatmap_sections.get('Events', [])), None)
    im = PIL.Image.new('RGB', (512, 512), 0)
    if imageline is not None:
        osu_bg = beatmapset_osu.joinpath(imageline.split(',')[2].strip('"'))
        if osu_bg.exists():
            im = PIL.Image.open(str(osu_bg)).convert('RGB')
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
            im.thumbnail((512, 512))
    im.save(str(ragna_beatmap_thumb))
    return False


def convert_audio_osu2ragna(beatmapset_osu: Path, ragna_beatmap_audio: Path, osu_beatmap_sections: Dict[str, List[str]]) -> bool:
    if not ragna_beatmap_audio.exists():
        ragna_beatmap_audio_tmp = ragna_beatmap_audio.with_suffix('.ogg')
        osu_beatmap_audio = beatmapset_osu.joinpath(
            get_audio_file_from_section(osu_beatmap_sections))
        r = subprocess.run(
            ['ffmpeg', '-y',
                '-i', str(osu_beatmap_audio),
                '-q', '9',
                '-map_metadata', '-1',
                '-vn', '-acodec', 'libvorbis',
                str(ragna_beatmap_audio_tmp),
             ],
        )
        if r.returncode:
            ragna_beatmap_audio.parent.joinpath(
                'broken_audio.flag').write_text(
                    'broken_audio')
            if ragna_beatmap_audio_tmp.exists():
                ragna_beatmap_audio_tmp.unlink()
            return True
        del r
        ragna_beatmap_audio_tmp.rename(ragna_beatmap_audio)
        del ragna_beatmap_audio_tmp
        del osu_beatmap_audio
    del ragna_beatmap_audio
    return False
