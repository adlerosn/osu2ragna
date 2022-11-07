#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import argparse
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

from osu2ragna.ragnatools import (EffortCalculator, RagnaRockDifficultyV1,
                                  RagnaRockInfo)
import osu2ragna.effort_constants


def write_effort_constants(
    hit_effort: float,
    move_effort: float,
    away_effort: float,
    doublehand_effort: float,
    relax_behavior: float,
    relax_rate: float,
    slope: float,
    intercept: float,
) -> str:
    s = ''
    s += '#!/usr/bin/env python3\n'
    s += '# -*- encoding: utf-8 -*-\n'
    s += '\n'
    s += 'from typing import Tuple\n'
    s += '\n'
    s += 'EFFORT_CALCULATOR_DEFAULTS: Tuple[float, float, float, float, float, float] = (\n'
    s += f'    {hit_effort},\n'
    s += f'    {move_effort},\n'
    s += f'    {away_effort},\n'
    s += f'    {doublehand_effort},\n'
    s += f'    {relax_behavior},\n'
    s += f'    {relax_rate},\n'
    s += ')\n'
    s += '\n'
    s += 'EFFORT_CALCULATOR_RELOCATOR_DEFAULTS: Tuple[float, float] = (\n'
    s += f'    {slope},\n'
    s += f'    {intercept},\n'
    s += ')\n'
    Path(osu2ragna.effort_constants.__file__).write_text(s, encoding='utf-8')
    return s


def get_parser():
    parser = argparse.ArgumentParser(
        prog=f'{Path(sys.executable).stem} -m {Path(__file__).parent.name}')
    parser.add_argument('--drylearn', action='store_const',
                        const=True, default=False, help='Learn and print coefficients')
    parser.add_argument('--learn', action='store_const',
                        const=True, default=False, help='Learn, print coefficients and write them into disk')
    parser.add_argument('--test', action='store_const',
                        const=True, default=False, help='Apply coefficients')
    parser.add_argument('ragna_customsongs_path', type=Path, default=None,
                        metavar='RAGNA_CUSTOMSONGS',
                        help='The CustomSongs folder for Raganarock')
    parser.add_argument('hit_effort', type=float,
                        metavar='HIT_EFFORT',
                        default=osu2ragna.effort_constants.EFFORT_CALCULATOR_DEFAULTS[0], nargs='?',
                        help=f'Hit Effort Coefficient (default: {osu2ragna.effort_constants.EFFORT_CALCULATOR_DEFAULTS[0]})')
    parser.add_argument('move_effort', type=float,
                        metavar='MOVE_EFFORT',
                        default=osu2ragna.effort_constants.EFFORT_CALCULATOR_DEFAULTS[1], nargs='?',
                        help=f'Movement Effort Coefficient (default: {osu2ragna.effort_constants.EFFORT_CALCULATOR_DEFAULTS[1]})')
    parser.add_argument('away_effort', type=float,
                        metavar='AWAY_EFFORT',
                        default=osu2ragna.effort_constants.EFFORT_CALCULATOR_DEFAULTS[2], nargs='?',
                        help=f'Away Effort Coefficient (default: {osu2ragna.effort_constants.EFFORT_CALCULATOR_DEFAULTS[2]})')
    parser.add_argument('doublehand_effort', type=float,
                        metavar='DOUBLEHAND_MLTPLR',
                        default=osu2ragna.effort_constants.EFFORT_CALCULATOR_DEFAULTS[3], nargs='?',
                        help=f'Double-handed Effort Multiplier Coefficient (default: {osu2ragna.effort_constants.EFFORT_CALCULATOR_DEFAULTS[3]})')
    parser.add_argument('relax_behavior', type=float,
                        metavar='RELAX_BEHAVIOR',
                        default=osu2ragna.effort_constants.EFFORT_CALCULATOR_DEFAULTS[4], nargs='?',
                        help=f'The Relax Rate Effort Coefficient (default: {osu2ragna.effort_constants.EFFORT_CALCULATOR_DEFAULTS[4]})')
    parser.add_argument('relax_rate', type=float,
                        metavar='RELAX_RATE',
                        default=osu2ragna.effort_constants.EFFORT_CALCULATOR_DEFAULTS[5], nargs='?',
                        help=f'The Relax Rate Effort Coefficient (default: {osu2ragna.effort_constants.EFFORT_CALCULATOR_DEFAULTS[5]})')
    return parser


def load_ragnasongs(ragna_customsongs_path: Path) -> List[RagnaRockInfo]:
    return [RagnaRockInfo.read_from(info_dat_path)
            for info_dat_path in ragna_customsongs_path.glob('*/info.dat')]


class ForeignProcessTester:
    def __init__(self,
                 songs,
                 hit_effort,
                 hit_effort_,
                 move_effort,
                 move_effort_,
                 away_effort,
                 away_effort_,
                 doublehand_effort,
                 doublehand_effort_,
                 relax_behavior,
                 relax_behavior_,
                 relax_rate,
                 relax_rate_,
                 magnitude,
                 ) -> None:
        self.songs: list = songs
        self.hit_effort: float = hit_effort
        self.move_effort: float = move_effort
        self.away_effort: float = away_effort
        self.doublehand_effort: float = doublehand_effort
        self.relax_behavior: float = relax_behavior
        self.relax_rate: float = relax_rate
        self.hit_effort_: int = hit_effort_
        self.move_effort_: int = move_effort_
        self.away_effort_: int = away_effort_
        self.doublehand_effort_: int = doublehand_effort_
        self.relax_behavior_: int = relax_behavior_
        self.relax_rate_: int = relax_rate_
        self.magnitude: int = magnitude

    def __call__(self) -> Tuple[Tuple[float, float, float, float, float, float, float, float, float], int, int, int, int, int, int]:
        return (do_test(self.songs,
                        self.hit_effort+self.hit_effort_*10**self.magnitude,
                        self.move_effort+self.move_effort_*10**self.magnitude,
                        self.away_effort+self.away_effort_*10**self.magnitude,
                        self.doublehand_effort+self.doublehand_effort_*10**self.magnitude,
                        self.relax_behavior+self.relax_behavior_*10**self.magnitude,
                        self.relax_rate+self.relax_rate_*10**self.magnitude,
                        ),
                self.hit_effort_, self.move_effort_, self.away_effort_,
                self.doublehand_effort_, self.relax_behavior_, self.relax_rate_)


def do_dry_learn(songs: List[RagnaRockDifficultyV1],
                 hit_effort: float,
                 move_effort: float,
                 away_effort: float,
                 doublehand_effort: float,
                 relax_behavior: float,
                 relax_rate: float,
                 printing: bool = False,
                 ) -> Tuple[float, float, float, float, float, float, float, float]:
    tests: List[Tuple[Tuple[float, float, float, float, float, float, float, float, float],
                      float, float, float, float, float, float]] = []
    for magnitude in range(0, -4, -1):
        while True:
            tests = []
            with ProcessPoolExecutor(cpu_count()) as pe:
                def callmeback(result: Future):
                    nonlocal tests
                    tpl = result.result()
                    tests.append(tpl)
                    if printing:
                        print(f'        {tpl}')
                for hit_effort_ in range(-1, 2, 1):
                    for move_effort_ in range(-1, 2, 1):
                        for away_effort_ in range(-1, 2, 1):
                            for doublehand_effort_ in range(-1, 2, 1):
                                for relax_behavior_ in range(-1, 2, 1):
                                    for relax_rate_ in range(-1, 2, 1):
                                        fut = pe.submit(ForeignProcessTester(
                                                        songs,
                                                        hit_effort,
                                                        hit_effort_,
                                                        move_effort,
                                                        move_effort_,
                                                        away_effort,
                                                        away_effort_,
                                                        doublehand_effort,
                                                        doublehand_effort_,
                                                        relax_behavior,
                                                        relax_behavior_,
                                                        relax_rate,
                                                        relax_rate_,
                                                        magnitude,
                                                        ))
                                        fut.add_done_callback(callmeback)
            tests.sort()
            hit_effort__, move_effort__, away_effort__, doublehand_effort__, relax_behavior__, relax_rate__ = (
                tests[0][1:])
            hit_effort += round(hit_effort__*10**magnitude, 6)
            move_effort += round(move_effort__*10**magnitude, 6)
            away_effort += round(away_effort__*10**magnitude, 6)
            doublehand_effort += round(doublehand_effort__*10**magnitude, 6)
            relax_behavior += round(relax_behavior__*10**magnitude, 6)
            relax_rate += round(relax_rate__*10**magnitude, 6)
            if printing:
                print(
                    f'    {magnitude=} corr={-tests[0][0][0]} cov={-tests[0][0][1]} abs(varx)={-tests[0][0][2]} varx={-tests[0][0][3]} max_err={tests[0][0][4]} avg_err={tests[0][0][5]} min_err={tests[0][0][6]} {hit_effort=} {move_effort=} {away_effort=} {doublehand_effort=} {relax_behavior=} {relax_rate=} slope={tests[0][0][7]} intercept={tests[0][0][8]}')
            if hit_effort__ == 0 and move_effort__ == 0 and away_effort__ == 0 and doublehand_effort__ == 0 and relax_rate__ == 0:
                break
    tests.sort()
    out = (
        hit_effort,
        move_effort,
        away_effort,
        doublehand_effort,
        relax_behavior,
        relax_rate,
        tests[0][0][7],
        tests[0][0][8],
    )
    if printing:
        print(
            f'corr={-tests[0][0][0]} cov={-tests[0][0][1]} abs(varx)={-tests[0][0][2]} varx={-tests[0][0][3]} max_err={tests[0][0][4]} avg_err={tests[0][0][5]} min_err={tests[0][0][6]} {hit_effort=} {move_effort=} {away_effort=} {doublehand_effort=} {relax_behavior=} {relax_rate=} slope={tests[0][0][7]} intercept={tests[0][0][8]}')
    return out


def do_learn(songs: List[RagnaRockDifficultyV1],
             hit_effort: float,
             move_effort: float,
             away_effort: float,
             doublehand_effort: float,
             relax_behavior: float,
             relax_rate: float,
             printing: bool = False,
             ) -> Tuple[float, float, float, float, float, float, float, float]:
    s = do_dry_learn(
        songs,
        hit_effort,
        move_effort,
        away_effort,
        doublehand_effort,
        relax_behavior,
        relax_rate,
        printing,
    )
    write_effort_constants(*s)
    return s


def avg(fs: Union[List[float], Tuple[float, ...]]) -> float:
    return sum(fs) / max(1, len(fs))


def do_test(songs: List[RagnaRockDifficultyV1],
            hit_effort: float,
            move_effort: float,
            away_effort: float,
            doublehand_effort: float,
            relax_behavior: float,
            relax_rate: float,
            printing: bool = False
            ) -> Tuple[float, float, float, float, float, float, float, float, float]:
    ec = EffortCalculator(hit_effort, move_effort, away_effort,
                          doublehand_effort, relax_behavior, relax_rate)
    esa = [*map(lambda a: (ec(a), a.difficulty_rank), songs)]
    x, y = [*zip(*esa)]
    avgx = avg(x)
    avgy = avg(y)
    varx = sum((xi-avgx)**2 for xi in x)/len(x)
    vary = sum((yi-avgy)**2 for yi in y)/len(y)
    cov = sum((x-avgx)*(y-avgy) for x, y in esa) / len(esa)
    sdx_sdy = ((varx ** .5) * (vary ** .5))
    corr = cov / max(0.000000001, sdx_sdy)
    ess = [abs(p-a) for p, a in esa]
    # linear regression: https://www.swiftutors.com/admin/photos/linear-regression-formula.png
    slope = ((len(x) * sum(xi*yi for xi, yi in esa) - sum(x) * sum(y)) /
             (len(x) * sum(xi**2 for xi in x) - sum(x)**2))
    intercept = (sum(y) - slope*sum(x)) / len(x)
    if printing:
        print(esa)
        print([(xi*slope + intercept, yi) for xi, yi in esa])
        print(
            f'{corr=} {cov=} {abs(varx)=} {varx=} {max(ess)=} {avg(ess)=} {min(ess)=} {slope=} {intercept=}')
    return (-corr, -cov, -abs(varx), -varx, max(ess), avg(ess), min(ess), slope, intercept)


def main():
    args = get_parser().parse_args()
    if len([*filter(bool, [args.drylearn, args.learn, args.test])]) != 1:
        raise ValueError(
            'At least one of --learn, --drylearn or --test must be used, but only one')
    songsets: List[RagnaRockInfo] = load_ragnasongs(
        args.ragna_customsongs_path)
    songs: List[RagnaRockDifficultyV1] = [song
                                          for songset in songsets
                                          for song in songset.difficultyBeatmapSets]
    cbk: Callable[[List[RagnaRockInfo], float, float, float, float],
                  Any] = do_learn if args.learn else (do_dry_learn if args.drylearn else do_test)
    cbk(
        songs,
        args.hit_effort,
        args.move_effort,
        args.away_effort,
        args.doublehand_effort,
        args.relax_behavior,
        args.relax_rate,
        printing=True,
    )


if __name__ == '__main__':
    main()
