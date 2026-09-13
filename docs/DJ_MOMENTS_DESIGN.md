# How a moment is made — design for a yes (2026-09-12)

*Written after "it seems a pale shadow… go read more on DJ theory about how moments are created" and
"it's a painful mishmash". Nothing below is built. This is the page to say yes or no to.*

## Why it is a mishmash

The conductor makes **one lane move per phrase**, each chosen by a dozen guards, forever. That is why it is
busy and shapeless at once: constant small motion, no shape inside the minute. Every control I added
(READY, LOUD/PHRASE, MOMENTS, FX, programs, DROP, BREAK) is a *knob on that machinery*. None of them is a
moment. A DJ's moment is not a lane flip; it is a **composed shape over 16–32 bars**, and the crowd feels
it because it was set up bars earlier.

## What the literature says a moment is

Solberg (2014, 2019) on the break routine, plus the practice sources (phrase mixing, tension/release):

1. **Take away** (contrasting breakdown): the beat and bass go; something melodic or a vocal stays. 8–16 bars.
2. **Build**: layers return in 2/4/8-bar steps, the frequency range opens (filter), density rises. 8–16 bars.
3. **The hole**: a beat to a bar of near silence, the downbeat *delayed*. The one thing listeners named
   unprompted. The longer the expectancy is held, the bigger the peak.
4. **Payoff**: bass drum and bass slam back **on the one of a phrase**, louder than before.

Everything lands on the phrase grid (the 32). The strongest single lever, measured, is the **removal and
reintroduction of bass and bass drum**, not effects. We have exactly that lever: stems.

## Five composed moments (each a script of bar-offsets the conductor executes)

| moment | needs | take away | build | hole | payoff (on the one) |
|---|---|---|---|---|---|
| **SLAM** | one song | bass + drums out for the last 2 bars of a phrase | — | last beat | bass + drums back, +30 % for a phrase |
| **BREAK ROUTINE** | one song | drums + bass out 8 bars, melody/vocal stays | drums back at −8, bass EQ opens over −8…−1 | last bar (only the vocal, or nothing) | all back on the one at the song's drop or hook if within 4 phrases, else the next phrase |
| **SING-ALONG** | a song that sings | at its hook: bass + drums out 2 bars, vocal alone | — | — | back on the third bar of the hook |
| **TAKEOVER** (exists as the bass moment) | two songs | old bass out | new bassline in loud at the payoff | 1 beat before | new drums follow at the next phrase |
| **ACAPELLA BRIDGE** | two songs | old bed out over 4 bars, old vocal alone 4–8 bars | new song's drums in under it at −4 | last beat | new bass + drums slam on the one; the old vocal rests when its line ends |

Two more build tools exist in practice and I want a **yes or no on each**, because they touch the "no
stutter builds" rule: **loop tightening** (the drums loop 4 → 2 → 1 bar over the last 8 bars = the
drum-roll effect, made from the real drum stem) and a **high-pass sweep on the mix bus** over the build
(a mixer move, not a synthesized effect). Without them the build is drums-return + bass-EQ-opens, which is
enough for the BREAK ROUTINE to read.

## Where it lands

Always on a phrase boundary of the receiving song, preferably its **measured drop or hook** (we have both).
Arming a moment picks the nearest such landing within 4 phrases and counts down to it. While a moment
runs, the arrangement's own per-phrase moves are **paused**: nothing else moves. Between moments the
policy gets calmer (one move per two phrases, voices stay longer). That alone addresses "busy".

## The surface (what changes on screen)

- **One row of five buttons**: SLAM · BREAK · SING · TAKEOVER · BRIDGE. Press = **arm**. The picture draws
  the choreography as a dashed plan across the four lanes (what leaves where, the hole, the slam) and the
  button reads its countdown: `SLAM in 6 bars`. Press again = **go now** (next bar). Right-click = cancel.
- The two-song moments use the newest voice or staged song; with none, the first song UP NEXT is staged.
- **MOMENTS dial** (rare / some / lots) arms these same five by itself, spaced as now; you see the same
  countdown and can cancel or change it. So the DJ's own moments and yours are the same thing on the same row.
- **Leaves the main surface**: the READY grid, LOUD/PHRASE, the FX dial, DROP and BREAK as separate buttons
  (they become SLAM-now and BREAK), the 2×4 program grid (one PROGRAM box). Stem-by-stem control stays one
  right-click away on a song in the picture or the list.
- Nothing on the surface rebuilds per bar (done today for READY and SONGS).

```
 story  │ ····strip····|·····build·····|hole| SLAM ▶ Song B drop
 drums  │ ████████                ▒▒▒▒▒▒▒▒     ██████████████
 bass   │ ████████                     ▒▒▒▒    ██████████████
 melody │ ████████████████████████████████████  ██████████████
 vocals │        ██████████████████████████     ██████
                        -16          -8   -1  0 (the one)
```

## What I need from you

1. Yes or no to the five moments, or which to drop or add.
2. Loop tightening: yes / no. Bus high-pass sweep in a build: yes / no.
3. Yes to taking READY, LOUD/PHRASE, FX and the program grid off the main surface.

Then the build order is: the moment script runner in the conductor with SLAM and BREAK ROUTINE first (one
song, no picker involved), heard by you, before the two-song ones.
