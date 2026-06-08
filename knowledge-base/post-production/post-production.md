# POST-PRODUCTION — Complete Knowledge Base
# TitanAI Reference: Editorial through Delivery

---

## EDITORIAL

### Workflow Overview
1. **Dailies** — footage ingested, synced, QC'd, LUT applied, sent to editor + director
2. **Assembly Cut** — editor's first pass, all usable material, no pacing concerns
3. **Rough Cut** — structured narrative, all scenes present, 2-3x final length typical
4. **Director's Cut** — director's creative vision locked (DGA: 10 weeks minimum for features)
5. **Producer's Cut** — producers give notes, address studio/financier concerns
6. **Picture Lock** — final cut approved, no further changes; VFX, sound, music begin conforming

### Editing Systems
- **Avid Media Composer** — industry standard for features/episodic; bin-based project structure, MXF media
- **Adobe Premiere Pro** — growing adoption, native format support, good for docs/indie
- **DaVinci Resolve** — free version capable, Studio version for collaboration; also used for colour
- **Frame.io** — cloud review and approval, deep Premiere/Resolve integration
- **Editorial Standards** — 24fps for cinema, 25fps for European broadcast, 29.97/23.976 for US broadcast

### Dailies Pipeline
- **DIT (Digital Imaging Technician)** — on-set data management, LUT application, colour QC
- **Data Wrangler** — copies media, verifies checksums, labels drives
- **Checksums** — MD5 or xxHash verification; never use a copy without verifying
- **Dailies Lab** — Technicolor, Deluxe, or in-house; applies show LUT, creates viewing copies
- **Deliverables** — camera originals (RAID + cloud backup), editorial proxies, audio sync

### Avid Best Practices
- **Project Settings** — match camera format; 4K projects with 1080p proxies common
- **Bins** — organised by scene, character, or type; linked clips not embedded
- **Media Management** — consolidate/transcode before offline; never move Avid media manually
- **ScriptSync** — phonetic sync of dialogue to script; dramatically speeds dailies review
- **MultiCamera** — sync via timecode or audio waveform; assign camera angles

---

## VISUAL EFFECTS (VFX)

### VFX Pipeline
1. **Tracking/Matchmove** — 2D/3D tracking of camera movement for CGI integration
2. **Roto/Paint** — rotoscoping (masking), wire/rig removal, beauty work
3. **FX Simulation** — fire, water, cloth, destruction (Houdini primary tool)
4. **Modelling/Rigging** — 3D asset creation and character rigging
5. **Animation** — keyframe or motion capture driven character/creature animation
6. **Lighting/Rendering** — CG elements lit to match live plate, rendered via farm
7. **Compositing** — combining all elements into final frame (Nuke primary tool)
8. **Colour/Finish** — VFX elements colour matched to DI before final grade

### Software
- **Houdini** (SideFX) — industry standard for FX simulation, procedural effects
- **Maya** (Autodesk) — industry standard for character animation and rigging
- **Nuke** (Foundry) — industry standard compositing; node-based
- **After Effects** (Adobe) — broadcast/lower budget compositing
- **Blender** — open source, growing feature film adoption for modelling/animation
- **Unreal Engine** — virtual production, real-time rendering, LED volume walls
- **Substance Painter** — 3D texturing
- **ZBrush** — sculpting characters and creatures

### Virtual Production
- **LED Volume** — large curved LED wall displays background footage in real-time
- **In-Camera VFX (ICVFX)** — final pixel VFX captured in camera, reduces post complexity
- **Mo-Sys** — camera tracking system for LED volume; transmits camera data to Unreal
- **Frustum** — inner LED volume area that camera sees; rest of wall provides ambient light
- **Benefits** — director sees final result on set, talent reacts to real environments, eliminates greenscreen

### VFX Bidding & Management
- **Turnover Package** — editorial exports for VFX houses to bid on
- **Bid** — per-shot or per-sequence pricing; fixed bid vs T&M
- **VFX Supervisor** — creative lead, client-side; manages vendor relationships
- **VFX Producer** — schedule, budget, delivery management
- **Vendor Review** — weekly client review of in-progress VFX work
- **Final Delivery** — EXR sequences at full resolution, multiple passes

---

## SOUND POST-PRODUCTION

### Sound Editorial
- **Supervising Sound Editor** — leads entire sound post team
- **Dialogue Editor** — cleans production dialogue, manages ADR integration
- **ADR (Automated Dialogue Replacement)** — re-recording dialogue in studio; looping
- **ADR Supervisor** — selects ADR lines, casts ADR voice talent, supervises sessions
- **Sound Effects Editor** — builds SFX editorial: cut, sync, layer
- **Foley** — custom-recorded practical sounds (footsteps, cloth, props)
- **Foley Artist** — performs foley to picture in specialised Foley stage
- **Ambience/BG** — background atmosphere tracks; room tone, environment sounds

### Music
- **Composer** — writes original score; delivered as stems (strings, brass, etc.)
- **Music Supervisor** — selects and licenses pre-existing music
- **Music Editor** — syncs score and songs to picture; manages music deliverables
- **Temp Track** — placeholder music used in editorial; often too close to final cut
- **Score Sessions** — live orchestral recording; typical 3 minutes scored per 3-hour session
- **Music Licensing** — sync license (right to use in film) + master license (specific recording)

### Mix
- **Re-recording Mixer** — performs the final mix of all audio elements
- **Premix** — individual elements mixed into submixes (dialogue, SFX, music)
- **Final Mix** — all submixes combined, balanced, and prepared for delivery
- **Loudness Standards** — cinema: no standard; broadcast: ATSC A/85 (-24 LKFS); streaming: -14 LUFS (Spotify), -14 LUFS (Apple Music), -13 LUFS (YouTube)
- **Dolby Atmos** — object-based audio; up to 128 audio objects + 10 bed channels
- **IMAX** — 12.0 configuration (12 speakers); separate IMAX mix often required

---

## COLOUR GRADING / DI (DIGITAL INTERMEDIATE)

### DI Process
1. **Conform** — picture lock cut rebuilt in DI suite using camera originals
2. **Grade** — primary (exposure/colour balance) and secondary (selective) grading
3. **VFX Integration** — VFX finals composited/checked within DI
4. **QC** — Quality Control; frame-by-frame check for defects
5. **Mastering** — creating distribution masters (DCP, HDR masters, SDR masters)
6. **Deliverables** — all required distribution formats generated

### Colour Grading Tools
- **DaVinci Resolve** — industry standard; node-based colour pipeline
- **Baselight** (Filmlight) — high-end facility tool; popular in Europe
- **Lustre** (Autodesk) — legacy tool, less common now

### DaVinci Resolve Workflow
- **Nodes** — serial (sequential), parallel (combined), layer (blend modes)
- **Colour Spaces** — input (log: ARRI LogC, REDlog, SLog3), working (scene-linear or DaVinci Wide Gamut), output (P3-D65 DCI, Rec.2020, Rec.709)
- **LUTs** — 1D (simple gamma), 3D (full colour transform); show LUT vs technical LUT
- **Scopes** — waveform (exposure), parade (colour balance), vectorscope (saturation/hue)
- **Colour Managed Workflow** — ACES or DaVinci colour management; consistent across pipeline

### HDR Standards
- **HDR10** — open standard, static metadata, 10-bit, Rec.2020, PQ curve, 1000-4000 nit mastering
- **Dolby Vision** — proprietary, dynamic metadata per shot, 12-bit capable, 4000-10000 nit mastering
- **HLG (Hybrid Log-Gamma)** — broadcast HDR, no metadata, backwards compatible
- **SDR to HDR** — re-grade required; HDR is not just boosted brightness

---

## DELIVERY

### Cinema (DCP)
- **DCP (Digital Cinema Package)** — industry standard cinema delivery format
- **Specs** — JPEG2000 video at 250 Mbps, PCM audio, MXF wrapped, XML CPL
- **Resolutions** — 2K (2048x858 scope, 2048x1080 flat), 4K (4096x1716 scope)
- **Frame Rates** — 24, 25, 48, 60 fps supported
- **KDM** — Key Delivery Message; encrypts DCP for specific theatre/projector
- **DCP Creation** — easyDCP, OpenDCP (free), Clipster, Fraunhofer

### Streaming Deliverables (Common Requirements)
| Platform | Video Codec | Resolution | Audio | HDR |
|----------|------------|------------|-------|-----|
| Netflix | HEVC/H.264 | Up to 4K | Dolby Atmos + 5.1 | Dolby Vision + HDR10 |
| Apple TV+ | HEVC/ProRes | Up to 4K | Dolby Atmos + 5.1 | Dolby Vision + HDR10 |
| Amazon | H.264/HEVC | Up to 4K | Dolby Atmos | HDR10+ + Dolby Vision |
| Disney+ | H.264/HEVC | Up to 4K | Dolby Atmos | Dolby Vision + HDR10 |
| HBO Max | H.264/HEVC | Up to 4K | Dolby Atmos | HDR10 + Dolby Vision |

### Broadcast Deliverables
- **AS-11** — UK broadcast standard (BBC, ITV); MXF wrapper, DNxHD or AVC-Intra
- **IMF (Interoperable Master Format)** — Netflix/Amazon required; MXF package, enables versioning
- **ProRes** — Apple codec; 4444, 422 HQ, 422, 422 LT, 422 Proxy
- **Closed Captions** — CEA-608 (SD), CEA-708 (HD); SRT, SCC, TTML formats
- **Audio Deliverables** — full mix stems: M&E (music + effects), dialogue only, music only

### E&O Insurance & Legal Deliverables
- **Chain of Title** — proves ownership from original source to current owner
- **Clearance Report** — confirms all script elements, music, locations are cleared
- **E&O Insurance** — Errors & Omissions; required by all distributors
- **Music Cue Sheet** — every piece of music, composer, publisher, duration, usage type
- **MPAA Rating** — G, PG, PG-13, R, NC-17; submitted via CARA
- **BBFC Rating** — UK classification; U, PG, 12A, 15, 18
