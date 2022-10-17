#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from enum import IntEnum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

'''https://bsmg.wiki/mapping/map-format.html'''


class RagnaRockInfoButDifficulties:
    def __init__(self,
                 songName: str = '',
                 songSubName: str = '',
                 songAuthorName: str = '',
                 levelAuthorName: str = '',
                 beatsPerMinute: float = 150.00,
                 previewStartTime: float = 20.25,
                 previewDuration: float = 10.00,
                 customData: Dict[str, Any] = None
                 ) -> None:
        self.version = '1'
        self.songName = songName
        self.songSubName = songSubName
        self.songAuthorName = songAuthorName
        self.levelAuthorName = levelAuthorName
        self.beatsPerMinute = beatsPerMinute
        # 4/4 timing was HARDCODED on BeatSaber side; no RagnaRock documentation says otherwise
        self._msBetweenTimePoints: float = 60000/(4*self.beatsPerMinute)
        self.shuffle = 0
        self.shufflePeriod = 0.5
        self.previewStartTime = previewStartTime
        self.previewDuration = previewDuration
        self.songFilename = 'song.egg'
        self.coverImageFilename = 'cover.jpg'
        self.environmentName = 'Midgard'
        self.songTimeOffset = 0
        self.customData: Dict[str, Any] = customData or {}
        self.difficultyBeatmapSets: list = []

    def round_to_beat(self, event_ms: Union[int, float]) -> int:
        return round(round(event_ms/self._msBetweenTimePoints)*self._msBetweenTimePoints)

    def convert_to_beat(self, event_ms: Union[int, float]) -> float:
        return round(round(event_ms/self._msBetweenTimePoints)/4, 2)

    def with_song_sub_name(self, songSubName: str) -> 'RagnaRockInfoButDifficulties':
        return type(self)(
            self.songName,
            songSubName,
            self.songAuthorName,
            self.levelAuthorName,
            self.beatsPerMinute,
            self.previewStartTime,
            self.previewDuration,
            self.customData,
        )

    def with_difficulties(self, difficultyBeatmapSets: List['RagnaRockDifficultyV1']) -> 'RagnaRockInfo':
        return RagnaRockInfo(
            self.songName,
            self.songSubName,
            self.songAuthorName,
            self.levelAuthorName,
            self.beatsPerMinute,
            self.previewStartTime,
            self.previewDuration,
            self.customData,
            difficultyBeatmapSets,
        )


class RagnaRockInfo(RagnaRockInfoButDifficulties):
    def __init__(self,
                 songName: str = '',
                 songSubName: str = '',
                 songAuthorName: str = '',
                 levelAuthorName: str = '',
                 beatsPerMinute: float = 150.00,
                 previewStartTime: float = 20.25,
                 previewDuration: float = 10.00,
                 customData: Dict[str, Any] = None,
                 difficultyBeatmapSets: List['RagnaRockDifficultyV1'] = None,
                 ) -> None:
        super().__init__(
            songName,
            songSubName,
            songAuthorName,
            levelAuthorName,
            beatsPerMinute,
            previewStartTime,
            previewDuration,
            customData,
        )
        self.difficultyBeatmapSets: List['RagnaRockDifficultyV1'] = (
            difficultyBeatmapSets or [])
        self._difficulty_internals = RagnarockDifficultyEnum.pick(
            len(self.difficultyBeatmapSets))

    def to_jsonable(self) -> dict:
        return {
            "_version": self.version,
            "_songName": self.songName,
            "_songSubName": self.songSubName,
            "_songAuthorName": self.songAuthorName,
            "_levelAuthorName": self.levelAuthorName,
            "_beatsPerMinute": self.beatsPerMinute,
            "_shuffle": self.shuffle,
            "_shufflePeriod": self.shufflePeriod,
            "_previewStartTime": self.previewStartTime,
            "_previewDuration": self.previewDuration,
            "_songFilename": self.songFilename,
            "_coverImageFilename": self.coverImageFilename,
            "_environmentName": self.environmentName,
            "_songTimeOffset": self.songTimeOffset,
            "_customData": {
                "_warnings": [],
                "_information": [],
                "_suggestions": [],
                "_requirements": [],
                **self.customData
            },
            "_difficultyBeatmapSets": [
                {
                    "_beatmapCharacteristicName": "Standard",
                    "_difficultyBeatmaps": [
                        {
                            "_difficulty": di.name,
                            "_difficultyRank": di.value,
                            "_beatmapFilename": f"Standard{di.name}.dat",
                            "_noteJumpMovementSpeed": 10,
                            "_noteJumpStartBeatOffset": 0,
                            "_customData": {
                                "_difficultyLabel": df.difficulty_label,
                                "_editorOffset": 0,
                                "_editorOldOffset": 0,
                                "_warnings": [],
                                "_information": [],
                                "_suggestions": [],
                                "_requirements": [],
                                ** df.customData
                            }
                        }
                        for di, df in zip(self._difficulty_internals, self.difficultyBeatmapSets)
                    ]
                }
            ]
        }

    def write_to(self, path: Path):
        path.joinpath('Info.dat').write_text(json.dumps(
            self.to_jsonable(), ensure_ascii=False, indent=2), encoding='utf-8')
        for di, df in zip(self._difficulty_internals, self.difficultyBeatmapSets):
            path.joinpath(f"Standard{di.name}.dat").write_text(json.dumps(
                df.to_jsonable(), ensure_ascii=False, separators=(',', ':')), encoding='utf-8')


class RagnarockDifficultyEnum(IntEnum):
    Easy = 3
    Normal = 5
    Hard = 8

    @classmethod
    def pick(cls, qtty: int) -> List['RagnarockDifficultyEnum']:
        if qtty < 1 or qtty > 3:
            raise ValueError(f'{qtty=} must be in the range [1, 3]')
        return [
            [cls.Normal],
            [cls.Normal, cls.Hard],
            [cls.Easy, cls.Normal, cls.Hard],
        ][qtty-1]


class RagnaRockDifficultyV1:
    def __init__(self,
                 ) -> None:
        self.version = '1'
        self.notes: List[RagnaRockDifficultyNote] = []
        self.difficulty_label: str = 'Normal'
        self.customData: Dict[str, Any] = {}

    def to_jsonable(self) -> dict:
        return {
            "_version": self.version,
            "_customData": {
                "_time": 0,
                "_BPMChanges": [],
                "_bookmarks": [],
                **self.customData,
            },
            "_events": [],
            "_notes": list(map(RagnaRockDifficultyNote.to_jsonable, self.notes)),
            "_obstacles": [],
        }


class RagnaRockDifficultyNote:
    def __init__(self,
                 time,
                 lineIndex,
                 ) -> None:
        self.time: float = time
        self.lineIndex: int = lineIndex

    def to_jsonable(self) -> dict:
        return {
            "_time": self.time,
            "_lineIndex": self.lineIndex,
            "_lineLayer": 1,
            "_type": 0,
            "_cutDirection": 1,
        }


class NoteLineIndexEnum(IntEnum):
    '''Bottom-left origin'''
    FarLeft = 0
    LightLeft = 1
    LightRight = 2
    FarRight = 3


class HandEnum(IntEnum):
    left = 0
    right = 1


class RagnaRockHandCoordinateHolder:
    def __init__(self,
                 line: NoteLineIndexEnum,
                 ) -> None:
        self.index: NoteLineIndexEnum = line

    def copy(self) -> 'RagnaRockHandCoordinateHolder':
        return type(self)(self.index)

    def goto(self, other: 'RagnaRockHandCoordinateHolder'):
        self.index = other.index


class RagnaRockCoreographyHintHolder:
    def __init__(self,
                 time: int,
                 coordinate: RagnaRockHandCoordinateHolder,
                 coordinate_end: RagnaRockHandCoordinateHolder,
                 ) -> None:
        self.time: int = time
        self.coordinate: RagnaRockHandCoordinateHolder = coordinate
        self.coordinate_end: RagnaRockHandCoordinateHolder = coordinate_end


class RagnaRockHandPositionSimulator:
    def __init__(self,
                 affinity: float,
                 timing: int,
                 coordinate: RagnaRockHandCoordinateHolder,
                 ) -> None:
        self.affinity: float = affinity
        self.timing: int = timing
        self.coordinate: RagnaRockHandCoordinateHolder = coordinate

    def copy(self):
        return type(self)(
            self.affinity,
            self.timing,
            self.coordinate.copy(),
        )

    def check_cut(self, time: int, beat_pos: NoteLineIndexEnum) -> RagnaRockCoreographyHintHolder:
        canvas_pos = RagnaRockHandCoordinateHolder(beat_pos)
        return RagnaRockCoreographyHintHolder(
            time,
            canvas_pos.copy(),
            canvas_pos.copy(),
        )

    def cut(self, now: int, coreography: RagnaRockCoreographyHintHolder):
        self.timing = now
        self.coordinate = coreography.coordinate.copy()


class RagnaRockHandsPositionsSimulator:
    def __init__(self, bsi: RagnaRockInfoButDifficulties) -> None:
        self.bsi = bsi
        self.hands = [
            RagnaRockHandPositionSimulator(
                0.5,
                0,
                RagnaRockHandCoordinateHolder(
                    NoteLineIndexEnum.LightLeft,
                ),
            ),
            RagnaRockHandPositionSimulator(
                2.5,
                0,
                RagnaRockHandCoordinateHolder(
                    NoteLineIndexEnum.LightRight,
                ),
            ),
        ]
        self.coreography_ids: List[int] = []
        self.coreography_contents: Dict[int,
                                        RagnaRockCoreographyHintHolder] = {}
        # -1 is at kernel's reserved space section; no pointer will be there
        self.coreography_hand_ids: List[int] = [-1, -1]
        self.last_coreography_hand_ids: List[int] = [-1, -1]

    @property
    def left(self) -> RagnaRockHandPositionSimulator:
        return self.hands[0]

    @left.setter
    def left(self, value):
        self.hands[0] = value

    @property
    def right(self) -> RagnaRockHandPositionSimulator:
        return self.hands[1]

    @right.setter
    def right(self, value):
        self.hands[1] = value

    def move_to(self,
                now: int,
                preferred_hands: Tuple[HandEnum, HandEnum],
                points_of_attention: List[Tuple[int, int, NoteLineIndexEnum]]) -> int:
        both_hands = False
        moves = 0
        for hand in range(2):  # free hand before assigning tasks
            coreography_id = self.coreography_hand_ids[hand]
            if coreography_id != -1 and now >= self.coreography_contents[coreography_id].time:
                self.last_coreography_hand_ids[hand] = self.coreography_hand_ids[hand]
                self.coreography_hand_ids[hand] = -1
            del coreography_id
            del hand
        sorted_poas = sorted(points_of_attention,
                             key=lambda poa: (poa[1], poa[2]))
        #                     1st: hit time
        #                     2nd: leftmost note
        for (beat_id, beat, beat_pos) in sorted_poas:
            if now != beat:
                # ignore notes not for the present
                continue
            if beat_id in self.coreography_ids:
                # ignore already processed notes
                continue
            # from now on, only notes and arc beginnings
            available_hands = [HandEnum(h)
                               for h, x in enumerate(self.coreography_hand_ids)
                               if x == -1]
            if len(available_hands) < 1:
                # print(' '*8+f'#> WARNING: busy hands dropped beat at {now}')
                continue
            active_hands = (
                list(preferred_hands) if both_hands else
                (
                    [next(h for h in preferred_hands if h in available_hands)]
                )
            )
            del available_hands
            moved = True
            for active_hand_enum in active_hands:
                # if both_hands:
                #     print(active_hands, now, active_hand_enum)
                last_coreography: Optional[RagnaRockCoreographyHintHolder] = self.coreography_contents.get(
                    self.last_coreography_hand_ids[active_hand_enum.value])
                active_hand: RagnaRockHandPositionSimulator = self.hands[active_hand_enum.value]
                coreography = active_hand.check_cut(beat, beat_pos)
                del last_coreography
                fixed_beat_id = beat_id if not both_hands else beat_id + active_hand_enum.value
                active_hand.cut(now, coreography)
                self.coreography_hand_ids[active_hand_enum.value] = fixed_beat_id
                self.coreography_contents[fixed_beat_id] = coreography
                self.coreography_ids.append(fixed_beat_id)
                del fixed_beat_id
                del coreography
                del active_hand
                del active_hand_enum
            if moved:
                moves += 1
            del moved
            del beat_id
            del beat
            del beat_pos
        del sorted_poas
        return moves

    def build_coreography(self) -> RagnaRockDifficultyV1:
        difficulty = RagnaRockDifficultyV1()
        for id_ in self.coreography_ids:
            coreography = self.coreography_contents[id_]
            difficulty.notes.append(
                RagnaRockDifficultyNote(
                    self.bsi.convert_to_beat(coreography.time),
                    coreography.coordinate.index,
                )
            )
            del coreography
            del id_
        return difficulty
