#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=protected-access
# pylint: disable=keyword-arg-before-vararg

from collections import defaultdict
from enum import IntEnum
from typing import (Any, Callable, DefaultDict, Dict, Generator, List,
                    Optional, Tuple)

from .generictools import identity, pick_most_frequent_or_default_sorting


class OsuHitsound:
    def __init__(self, hitsound: int):
        self._hitsound = hitsound

    def normal(self) -> bool:
        return bool(self._hitsound & (1 << 0))

    def whistle(self) -> bool:
        return bool(self._hitsound & (1 << 1))

    def finish(self) -> bool:
        return bool(self._hitsound & (1 << 2))

    def clap(self) -> bool:
        return bool(self._hitsound & (1 << 3))


class OsuHitObjectType(IntEnum):
    HitCircle = 1 << 0
    Slider = 1 << 1
    Spinner = 1 << 3
    Hold = 1 << 7  # osu!mania
    # SliderEdges does not actually exist in the beatmap, but let's pretend that
    #  every in-game slider backwards arrow is an actual object, as it might be useful
    SliderEdge = 1 << 30
    # SliderHuffs does not actually exist in the beatmap, but let's pretend that
    #  every in-game huff in a slider is an actual object, as it might be useful
    SliderHuff = 1 << 31


class OsuHitObject:
    '''https://osu.ppy.sh/help/wiki/osu!_File_Formats/Osu_(file_format)'''

    def __init__(
            self,
            slider_multiplier: float,
            beat_length: float,
            x: int,
            y: int,
            time: float,
            tpe: int,
            hitsound: int,
            extras: List[str] = None,
            *args,
            parent: 'OsuHitObject' = None,
            timing_point: 'OsuTimingPoint',
            **kwargs
    ):
        '''
        x,y,time,type,hitSound,objectParams,hitSample
        x,y,time,type,hitSound,extras
        '''
        super().__init__(*args, **kwargs)
        self._x = x
        self._y = y
        self._time = time
        self._type = tpe
        self._hitsound = hitsound
        self._hitsound_object = OsuHitsound(self._hitsound)
        self._slider_multiplier = slider_multiplier
        self._beat_length = beat_length
        self._extras = extras if extras is not None else []
        self._parent = parent
        self._timing_point: OsuTimingPoint = getattr(
            parent, '_timing_point', timing_point)
        self._new_combo = self._compute_new_combo()
        self._osu_type = self._compute_osu_type()
        self._finish_time = self._compute_finish_time()

    def hitsound(self) -> OsuHitsound:
        return self._hitsound_object

    def _compute_new_combo(self) -> int:
        if self._type & (1 << 2):
            return 1 + ((self._type & 0b01110000) >> 4)
        return 0

    def _compute_osu_type(self) -> OsuHitObjectType:
        for enumeration in OsuHitObjectType:
            if self._type & enumeration.value:
                return enumeration
        raise ValueError(f"Invalid Osu Type: {bin(self._type)}")

    def _compute_finish_time(self) -> float:
        osu_type = self.osu_type()
        match osu_type:
            case OsuHitObjectType.HitCircle:
                return self._time
            case OsuHitObjectType.SliderEdge:
                return self._time + self._beat_length
            case OsuHitObjectType.SliderHuff:
                return self._time + self._beat_length
            case OsuHitObjectType.Spinner:
                # x,y,time,type,hitSound,endTime,hitSample
                # [...]                 ,endTime,hitSample
                # Extras idx:           ,0      ,1
                end_time = int(self._extras[0])
                return end_time/1000
            case OsuHitObjectType.Hold:
                # x,y,time,type,hitSound,endTime:hitSample
                # [...]                 ,endTime:hitSample
                # Extras idx:           ,0
                end_time = int(self._extras[0].split(':', 1)[0])
                return end_time / 1000
            case OsuHitObjectType.Slider:
                # x,y,time,type,hitSound,curveType|curvePoints,slides,length,edgeSounds,edgeSets,hitSample
                # [...]                 ,curveType|curvePoints,slides,length,edgeSounds,edgeSets,hitSample
                # Extras idx:           ,0                    ,1     ,2     ,3         ,4       ,5
                slides = int(self._extras[1])
                length = float(self._extras[2])
                # length / (SliderMultiplier * 100) * beatLength
                slide_duration = ((length / 1000) /
                                  (self._slider_multiplier * 100) *
                                  (self._beat_length))
                return self._time + slide_duration * slides
            case _:
                raise ValueError(f"Invalid Osu Type: {osu_type}")

    def coord_x(self) -> int:
        return self._x

    def coord_y(self) -> int:
        return self._y

    def new_combo(self) -> int:
        return self._new_combo

    def osu_type(self) -> OsuHitObjectType:
        return self._osu_type

    def finish_time(self) -> float:
        return self._finish_time

    def start_time(self) -> float:
        return self._time

    def beat_length(self) -> float:
        return self._beat_length

    def is_kiai(self) -> bool:
        return (False
                if self._timing_point is None else
                self._timing_point.is_kiai())

    def is_point(self, transformer: Callable[[int], int] = identity) -> bool:
        return transformer(round(self.start_time()*1000)) == transformer(round(self.finish_time()*1000))

    def slides(self) -> int:
        if self._osu_type == OsuHitObjectType.Slider:
            # x,y,time,type,hitSound,curveType|curvePoints,slides,length,edgeSounds,edgeSets,hitSample
            # [...]                 ,curveType|curvePoints,slides,length,edgeSounds,edgeSets,hitSample
            # Extras idx:           ,0                    ,1     ,2     ,3         ,4       ,5
            slides = int(self._extras[1])
            return max(1, slides)
        return 1

    def derive_pretending_hitcircle(self) -> List['OsuHitObject']:
        return [OsuHitObject(
            slider_multiplier=self._slider_multiplier,
            beat_length=self._beat_length,
            x=self.coord_x(),
            y=self.coord_y(),
            time=self.start_time(),
            tpe=OsuHitObjectType.HitCircle,
            hitsound=self.hitsound()._hitsound,
            timing_point=self._timing_point,
        )]

    def derive_hold_subobjects(self) -> List['OsuHitObject']:
        return list(self._derive_hold_subobjects())

    def _derive_hold_subobjects(self) -> Generator['OsuHitObject', None, None]:
        yield from []
        if self._osu_type == OsuHitObjectType.Spinner:
            time = self.start_time()
            while time <= self._finish_time:
                yield OsuHitObject(
                    self._slider_multiplier,
                    self._beat_length,
                    self._x,
                    self._y,
                    time+0.0,
                    OsuHitObjectType.HitCircle.value,
                    self._hitsound,
                    parent=self,
                    timing_point=self._timing_point,
                )
                time = time + (self._beat_length/1000)
        elif self._osu_type == OsuHitObjectType.Slider:
            tempo = 0
            time = self.start_time()
            while time <= self._finish_time:
                obj = OsuHitObject(
                    self._slider_multiplier,
                    self._beat_length,
                    self._x,
                    self._y,
                    time+0.0,
                    (OsuHitObjectType.SliderEdge.value
                     if tempo == 0 else
                     OsuHitObjectType.SliderHuff.value),
                    self._hitsound,
                    parent=self,
                    timing_point=self._timing_point,
                )
                if obj._finish_time > self._finish_time:
                    obj._finish_time = self._finish_time
                yield obj
                del obj
                time = time + (self._beat_length/1000)
                tempo += 1
                tempo %= self._timing_point.meter
        elif self._osu_type == OsuHitObjectType.Hold:
            yield self
        else:
            yield self

    @ classmethod
    def from_line(cls, osu_line: str, osu_timing_points_objects: List['OsuTimingPoint'], slide_multiplier: float) -> 'OsuHitObject':
        (x_str, y_str, time_str, tpe_str, hitsound_str, *other) = osu_line.split(',')
        (x, y, time, tpe, hitsound) = list(
            map(int, [x_str, y_str, time_str, tpe_str, hitsound_str]))
        time_sec = time/1000
        best_timing_point = next(iter(osu_timing_points_objects))
        for timing_point in osu_timing_points_objects:
            if timing_point.time > time_sec:
                break
            best_timing_point = timing_point
        beat_length = best_timing_point.beat_length
        return OsuHitObject(slide_multiplier, beat_length, x, y, time_sec, tpe, hitsound, other, timing_point=best_timing_point)

    def __repr__(self):
        return f'{type(self).__name__}(**{repr(self.__dict__)})'


class OsuModesEnum(IntEnum):
    Standard = 0
    Taiko = 1
    Catch = 2
    Mania = 3


class OsuFileSomeMetadata:
    def __init__(
        self,
        mode: OsuModesEnum,
        difficulty: str,
        title: str,
        artist: str,
        creator: str,
        beatmap_set_id: int,
        beatmap_id: int,
        circle_size: Optional[float] = None,
        preview_start: int = 0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.difficulty = difficulty
        self.title = title
        self.artist = artist
        self.creator = creator
        self.beatmap_set_id = beatmap_set_id
        self.beatmap_id = beatmap_id
        self.circle_size = circle_size
        self.preview_start = preview_start

    @classmethod
    def merge(cls, metadatas: List['OsuFileSomeMetadata']) -> 'OsuFileSomeMetadata':
        if len(metadatas) < 1:
            raise ValueError('metadatas cannot be an empty list')
        if len(metadatas) == 1:
            return cls(**metadatas[0].__dict__)
        data: DefaultDict[str, List[Any]] = defaultdict(list)
        for m in metadatas:
            for k, v in m.__dict__.items():
                data[k].append(v)
                del k
                del v
            del m
        freq_data: Dict[str, Any] = {}
        for k, v in data.items():
            freq_data[k] = pick_most_frequent_or_default_sorting(v)
            del k
            del v
        del data
        return cls(**freq_data)

    def __repr__(self) -> str:
        return (f'{type(self).__name__}(' +
                f'{type(self.mode).__name__}.{self.mode.name}, {self.difficulty!r}, ' +
                f'{self.title!r}, {self.artist!r}, {self.creator!r}, ' +
                f'{self.beatmap_set_id!r}, {self.beatmap_id!r}, ' +
                f'{self.circle_size!r}, {self.preview_start!r})')


class OsuTimingPoint:
    def __init__(
        self,
        time: float,
        beat_length: float,
        meter: int,
        sample_set: int,
        sample_index: int,
        volume: int,
        inherited: bool,
        effects: int,
        base_time: float,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = time
        self.beat_length = beat_length
        self.meter = meter
        self.sample_set = sample_set
        self.sample_index = sample_index
        self.volume = volume
        self.inherited = inherited
        self.effects = effects
        self.base_time = base_time

    def is_kiai(self) -> bool:
        return bool(self.effects & (1 << 0))

    @ classmethod
    def from_line(cls, osu_line: str, last_base: Optional['OsuTimingPoint'] = None) -> 'OsuTimingPoint':
        """time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects"""
        (time, beat_length, meter, sample_set, sample_index, volume,
         un_inherited, effects, *_) = [*osu_line.split(','), *(['0']*8)]
        args: List[Any] = [int(round(float(time))) / 1000, float(beat_length), int(meter),
                           int(sample_set), int(sample_index), int(volume),
                           not bool(int(un_inherited)), int(effects)]
        base_time = args[0]
        if args[6] == '1':  # inherits
            if last_base is None:
                raise ValueError("Cannot inherit value from nowhere")
            args[2] = last_base.meter
            if args[1] < 0:
                args[1] = (-100 / args[1]) * last_base.beat_length
            base_time = last_base.time
        if args[2] == 0:
            args[2] = 4
        if args[1] < 0:
            args[1] = -args[1]
        if args[1] == 0:
            args[1] = last_base.beat_length  # type: ignore
        return OsuTimingPoint(*args, base_time=base_time)  # type: ignore


def get_osu_sections(osu_text: str) -> Dict[str, List[str]]:
    osu_sections: Dict[str, List[str]] = {}
    for section in filter(lambda a: str.startswith(a, '['), filter(len, map(str.strip, osu_text.split('\n\n')))):
        header, *lines = section.splitlines()
        osu_sections[header.strip('[]').strip()] = lines
    return osu_sections


def get_osu_hit_objects(osu_text: str) -> List[OsuHitObject]:
    return get_osu_hit_objects_from_section(get_osu_sections(osu_text))


def get_osu_hit_objects_from_section(osu_sections: Dict[str, List[str]]) -> List[OsuHitObject]:
    osu_hit_objects: List[OsuHitObject] = []
    if 'Difficulty' not in osu_sections or 'HitObjects' not in osu_sections:
        return []
    slider_multiplier = float(next(map(lambda a: a.split(':')[1].strip(),
                                       filter(lambda a: str.startswith(a, 'SliderMultiplier'),
                                              map(str.strip,
                                                  osu_sections['Difficulty'])))))
    osu_timing_points_section = osu_sections.get('TimingPoints', [])
    osu_timing_points_objects = []
    osu_timing_points_last_base = None
    for osu_timing_point_line in osu_timing_points_section:
        otp = OsuTimingPoint.from_line(
            osu_timing_point_line, osu_timing_points_last_base)
        if not otp.inherited:
            osu_timing_points_last_base = otp
        osu_timing_points_objects.append(otp)
        del otp
        del osu_timing_point_line
    del osu_timing_points_last_base
    del osu_timing_points_section
    if len(osu_timing_points_objects) > 0:
        osu_hit_objects_section = osu_sections['HitObjects']
        for osu_hit_object_line in osu_hit_objects_section:
            oho = OsuHitObject.from_line(
                osu_hit_object_line, osu_timing_points_objects, slider_multiplier)
            osu_hit_objects.append(oho)
    return osu_hit_objects


def get_audio_file(osu_text: str) -> str:
    return get_audio_file_from_section(get_osu_sections(osu_text))


def get_audio_file_from_section(osu_sections: Dict[str, List[str]]) -> str:
    return get_section_properties_from_section(osu_sections, 'General', ': ')['AudioFilename']


def get_some_metadata(osu_text: str) -> OsuFileSomeMetadata:
    return get_some_metadata_from_section(get_osu_sections(osu_text))


def get_some_metadata_from_section(osu_sections: Dict[str, List[str]]) -> OsuFileSomeMetadata:
    mode = OsuModesEnum(int(get_section_properties_from_section(
        osu_sections, 'General', ': ').get('Mode', '0')))
    metadata = get_section_properties_from_section(
        osu_sections, 'Metadata', ':')
    difficulty = get_section_properties_from_section(
        osu_sections, 'Difficulty', ':')
    circle_size_t = difficulty.get('CircleSize', '').strip()
    circle_size = None if not circle_size_t else float(circle_size_t)
    preview_time = int(get_section_properties_from_section(
        osu_sections, 'General', ': ').get('PreviewTime', '0').strip())
    return OsuFileSomeMetadata(
        mode,
        metadata.get('Version', 'Normal'),
        metadata.get('Title', ''),
        metadata.get('Artist', ''),
        metadata.get('Creator', ''),
        int(metadata.get('BeatmapSetID', '0')),
        int(metadata.get('BeatmapID', '0')),
        circle_size,
        preview_time,
    )


def get_section_properties(osu_text: str, section: str, separator: str = ':') -> Dict[str, str]:
    return get_section_properties_from_section(
        get_osu_sections(osu_text),
        section,
        separator,
    )


def get_section_properties_from_section(osu_sections: Dict[str, List[str]], section: str, separator: str = ':') -> Dict[str, str]:
    return dict(map(lambda a: [*a.split(separator, 1), '', ''][:2], osu_sections.get(section, [])))


OSU_DEFAULT_COMBO_COLORS = [
    (124, 165, 0),
    (206, 118, 4),
    (200, 22, 3),
    (35, 145, 175),
]


def get_osu_combo_colors(osu_text: str) -> List[Tuple[int, int, int]]:
    return get_osu_combo_colors_from_section(get_osu_sections(osu_text))


def get_osu_combo_colors_from_section(osu_sections: Dict[str, List[str]]) -> List[Tuple[int, int, int]]:
    if 'Colours' not in osu_sections:
        return OSU_DEFAULT_COMBO_COLORS.copy()
    osu_custom_colors = []
    combo_to_rgb: Dict[str, str] = dict([tuple(  # type: ignore
        list(map(str.strip, combo.split(':', 1)))) for combo in osu_sections['Colours']])
    combo_count = 1
    while (rgb := combo_to_rgb.get(f'Combo{combo_count}', None)):
        combo_count += 1
        r, g, b = list(map(int, map(str.strip, rgb.split(','))))
        osu_custom_colors.append((r, g, b))
    return osu_custom_colors
