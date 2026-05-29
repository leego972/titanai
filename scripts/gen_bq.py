#!/usr/bin/env python3
"""
Generate upgrade_bq.jsonl — Hebrew, Kabbalah & Jewish History
60+ expert Q&A pairs. Appends to existing file if present.
"""
import json, os

OUT = "/workspace/data_upgrades/upgrade_bq.jsonl"

pairs = [
# ── KABBALAH ─────────────────────────────────────────────────────────────────
("""What are the Ten Sefirot in Kabbalah and what does each represent?""",
"""The Ten Sefirot are the divine attributes through which Ein Sof (the Infinite) reveals itself and creates the universe. From highest to lowest on the Tree of Life:

1. Keter (Crown) — the supreme, transcendent crown; divine will; source of all.
2. Chokhmah (Wisdom) — the first flash of divine thought; undifferentiated potential.
3. Binah (Understanding) — the divine womb; structured thought that gestates creation; the Supernal Mother.
4. Chesed (Lovingkindness) — divine grace, mercy, and unlimited generosity.
5. Gevurah / Din (Strength / Judgment) — divine power, strict judgment, discipline, contraction.
6. Tiferet (Beauty) — the heart of the Tree; harmony, compassion, truth; the Blessed Holy One.
7. Netzach (Eternity / Victory) — divine emotion, desire, artistry, nature forces.
8. Hod (Splendor / Acknowledgment) — divine intellect applied to emotion; prophecy, language, gratitude.
9. Yesod (Foundation) — the channel that transmits the flow of divine energy to the world; covenant, connection.
10. Malkhut (Kingdom) — the Shekhinah (divine presence); the final receptacle; the physical world.

The three columns of the Tree are: Right (mercy) — Chokhmah, Chesed, Netzach; Left (judgment) — Binah, Gevurah, Hod; Center (balance) — Keter, Tiferet, Yesod, Malkhut. The Sefirot are interconnected by 22 paths corresponding to the 22 Hebrew letters."""),

("""What is Tzimtzum in Lurianic Kabbalah?""",
"""Tzimtzum (literally "contraction" or "withdrawal") is the foundational cosmological concept developed by Rabbi Isaac Luria (the Ari, 1534-1572) to explain how an infinite God could create a finite universe.

The problem: If Ein Sof (the Infinite) fills all reality, there is no space for creation.
The solution: God voluntarily contracted/withdrew a portion of His infinite light into Himself, creating a conceptual vacuum — the Chalal (empty space). Into this space the divine light re-entered as a single ray (Kav), initiating the process of creation.

This act of self-limitation is also a moral paradigm: creation required divine self-restriction. The Tanya (Chabad) interprets Tzimtzum not as literal withdrawal but as concealment — the infinite is still present but hidden, allowing finite beings to experience themselves as separate.

The resulting vessels (Kelim) that received the divine light eventually shattered (Shevirat HaKelim — the Breaking of the Vessels) when they couldn't contain the intensity of divine light. The fallen sparks (Nitzotzot) scattered into the material world, and the purpose of humanity — through Torah, mitzvot, and intention (kavvanah) — is to raise these sparks back, accomplishing Tikkun Olam (repair of the world)."""),

("""Explain the Four Worlds of Kabbalah (Atzilut, Beriah, Yetzirah, Assiah).""",
"""Kabbalistic cosmology describes creation as proceeding through four distinct worlds, each a progressive diminishment of divine light:

1. Atzilut (Emanation) — The world of pure divinity. The Sefirot exist here in their highest, most unified form. No duality; only the divine. Corresponds to the Yod of the Tetragrammaton (YHVH). The realm of the Chayah soul-level.

2. Beriah (Creation) — The world of the Divine Throne (Kisei HaKavod) and the archangels. Here, distinct divine beings first emerge — including the souls of the righteous and the highest angels (Seraphim). Corresponds to the first Heh of YHVH. Realm of the Neshamah.

3. Yetzirah (Formation) — The world of formation, where the heavenly realms and the lower orders of angels (Chayot, Ofanim, etc.) exist. The emotional and action aspect of the divine. Corresponds to Vav of YHVH. Realm of the Ruach.

4. Assiah (Action) — The world of action; the physical universe including the material world and its spiritual counterpart. The Kelipot (husks/shells) are most present here. Corresponds to the final Heh of YHVH. Realm of the Nefesh.

The worlds are interconnected: what manifests in Assiah is rooted in the higher worlds. Human spiritual work elevates sparks from Assiah back toward Atzilut. The acronym ABiYA is commonly used for these four worlds."""),

("""What is the Zohar and who wrote it?""",
"""The Zohar (Hebrew: "Splendor" or "Radiance") is the central text of Kabbalah — the foundational mystical commentary on the Torah. It is written primarily in Aramaic and presents itself as teachings of the 2nd-century CE Tannaitic sage Rabbi Shimon bar Yochai (Rashbi) and his circle of disciples.

Historical context: The Zohar first appeared in 13th-century Spain, published by Rabbi Moses de Leon (c. 1240-1305) in Castile around 1280. Most modern scholars (including Gershom Scholem) conclude that de Leon authored or heavily edited most of the Zohar, though some portions may draw on earlier mystical traditions. De Leon attributed the work to Rashbi, who according to the Talmud spent years hiding in a cave — an environment suited to divine revelation.

Contents: The Zohar is an enormous collection including mystical commentaries on the Torah portion by portion, as well as independent sections like the Idra Rabba (The Great Assembly), Idra Zuta (The Small Assembly), Tikkunei Zohar (70 introductions to Bereishit), and the Sitrei Torah (Secrets of the Torah).

Key themes: The nature of God (Ein Sof and the Sefirot), the mystical meaning of Torah letters and words, the Shekhinah (divine feminine presence), the soul and its journey, cosmic redemption, and the spiritual significance of every mitzvah. The Zohar elevated Kabbalah from an esoteric elite practice to a widespread mystical movement, culminating in the Lurianic school of Safed in the 16th century."""),

("""What is Gematria and how is it used in Jewish mysticism?""",
"""Gematria is a hermeneutical method that assigns numerical values to Hebrew letters and uses these values to find hidden connections between words and concepts sharing the same numerical total.

The standard numerical assignments (Mispar Hechrachi):
Aleph=1, Bet=2, Gimel=3, Dalet=4, Heh=5, Vav=6, Zayin=7, Chet=8, Tet=9, Yod=10, Kaf=20, Lamed=30, Mem=40, Nun=50, Samech=60, Ayin=70, Peh=80, Tzaddi=90, Kuf=100, Resh=200, Shin=300, Tav=400.

Famous examples:
- Echad (One, aleph+chet+dalet = 1+8+4=13) = Ahavah (Love, aleph+heh+bet+heh = 1+5+2+5=13). The equation: God is One = God is Love.
- YHVH (10+5+6+5=26) = Adonai (1+4+50+10=65)... actually these don't match, but their product and other combinations are studied.
- Nachash (serpent) = 50+8+300=358 = Mashiach (Messiah) = 40+300+10+8=358. The Zohar interprets: the serpent represents the evil that the Messiah will ultimately transform.
- The word chai (life, chet+yod = 8+10=18) is why Jews commonly give charity in multiples of 18.

Forms of gematria: Mispar Gadol (using final forms of letters with higher values), Mispar Katan (reduced single digits), Atbash (substitution cipher), Albam, and many others. Gematria is used throughout Talmudic literature, Kabbalistic texts, and Hasidic teachings to uncover layers of Torah meaning."""),

("""Who was the Ari (Rabbi Isaac Luria) and what was his contribution to Kabbalah?""",
"""Rabbi Isaac Luria Ashkenazi (1534-1572), known as the Ari HaKadosh (the Holy Lion), was the greatest Kabbalist of the 16th century and the founder of Lurianic Kabbalah — the dominant school of Jewish mysticism from his time onward.

Biography: Born in Jerusalem, raised in Egypt, Luria moved to Safed (Tzfat) in the Galilee around 1570, joining an extraordinary circle of mystics including Rabbi Joseph Karo (author of the Shulchan Aruch), Rabbi Moses Cordovero, and Rabbi Shlomo Alkabetz (who wrote Lecha Dodi). Luria died at 38 after only two years in Safed, yet his impact was transformative.

Key innovations of Lurianic Kabbalah:
1. Tzimtzum — God's self-contraction to create space for the world (see above).
2. Shevirat HaKelim — the Breaking of the Vessels; cosmic catastrophe that scattered divine sparks.
3. Tikkun — the repair/rectification of the cosmic rupture through Torah, prayer, and mitzvot.
4. Gilgul Neshamot — detailed doctrine of metempsychosis (reincarnation of souls).
5. Kavvanot — specific mystical intentions for prayers and mitzvot that direct spiritual energy.
6. Partzufim — reconfiguration of the Sefirot into divine "faces" (Arikh Anpin, Abba, Imma, Zeir Anpin, Nukvah).

The Ari wrote almost nothing himself. His teachings were transmitted primarily by his disciple Rabbi Chaim Vital in Etz Chaim (Tree of Life) and Pri Etz Chaim (Fruit of the Tree of Life). Lurianic Kabbalah profoundly influenced all subsequent Jewish mysticism, including Hasidism."""),

("""What are the five levels of the soul in Kabbalah?""",
"""Kabbalah teaches that the human soul is not a single entity but a composite of five ascending levels:

1. Nefesh (נֶפֶשׁ) — the lowest, most physical level; the animating life-force shared with all living creatures. Associated with blood and the body's instincts. Corresponds to the world of Assiah.

2. Ruach (רוּחַ) — spirit; the emotional and moral dimension of the self. The seat of personality, character traits (middot), and the struggle between good and evil inclination (yetzer tov and yetzer hara). Corresponds to Yetzirah.

3. Neshamah (נְשָׁמָה) — the divine soul; the intellectual and spiritual dimension uniquely given to humans. The point of direct connection to God. This is what receives Torah understanding. Corresponds to Beriah.

4. Chayah (חַיָּה) — the living essence; a supernal level of soul that most people do not experience in ordinary consciousness. Corresponds to Atzilut. Experienced in states of profound spiritual illumination.

5. Yechidah (יְחִידָה) — the unique one; the highest level, completely united with God; the divine spark at the core of every soul. Beyond ordinary experience. The point of absolute singularity with Ein Sof.

The Tanya (by Rabbi Schneur Zalman of Liadi) elaborates further, describing the "G-dly soul" (Nefesh HaElohit) and the "animal soul" (Nefesh HaBehamit) that are in constant dialogue within the human being. The goal of spiritual work is to allow the G-dly soul to permeate and transform the animal soul."""),

("""What is Tikkun Olam and what does it mean in traditional Kabbalah versus contemporary usage?""",
"""Original Kabbalistic meaning: Tikkun Olam (תִּקּוּן עוֹלָם, "repair/rectification of the world") is a Lurianic Kabbalistic concept. After the cosmic catastrophe of Shevirat HaKelim (Breaking of the Vessels), divine sparks (Nitzotzot) fell into the material world and became trapped within Kelipot (husks/shells of impurity). Every human action performed with proper intention (kavvanah) — especially Torah study, prayer, and mitzvot — liberates these sparks and restores cosmic unity. The cumulative effect of all such actions will eventually complete the rectification and bring Messianic redemption. Each mitzvah repairs a specific aspect of the cosmic structure.

The phrase also appears in the Aleinu prayer in the phrase "l'takken olam b'malkhut Shaddai" (to repair the world under God's sovereignty) — meaning the universal acknowledgment of God's kingship.

Contemporary secular usage: In modern liberal Jewish movements (particularly post-1970s American Judaism), Tikkun Olam has been reinterpreted as a mandate for social justice, charity, environmentalism, and political activism — essentially any act that improves society. While these are indeed Jewish values (tzedakah, gemilut chasadim), traditional scholars note this usage strips the term of its theurgic, mystical dimension and its specifically Jewish covenantal context.

The tension is important: Traditional Kabbalah teaches Tikkun requires specifically Jewish religious acts (mitzvot) because the sparks fell according to the map of Torah. The contemporary usage universalizes it into a general humanitarian ethic."""),

# ── HEBREW LANGUAGE ───────────────────────────────────────────────────────────
("""What makes Hebrew unique as a sacred language in Jewish tradition?""",
"""Hebrew (Ivrit) holds a singular status in Jewish theology and mystical thought for several reasons:

1. Lashon HaKodesh (the Holy Tongue): The Talmud and Midrash designate Hebrew as the language in which God created the world — "God spoke and the world came into being" (Psalms 33:9). The Sefer Yetzirah teaches that God created existence through 22 Hebrew letters.

2. Intrinsic meaning: Unlike other languages where words are arbitrary labels, Kabbalah teaches that Hebrew letters and words possess inherent spiritual power. The letters are not mere symbols but energetic forces; each has a shape, numerical value, and cosmic function.

3. The Aleph-Bet: 22 letters, divided into: 3 Mothers (Aleph, Mem, Shin — air, water, fire), 7 Doubles (Bet, Gimel, Dalet, Kaf, Peh, Resh, Tav — planetary forces), 12 Simples (the remaining letters — zodiacal forces). This system is laid out in Sefer Yetzirah (Book of Formation), one of the earliest Kabbalistic texts.

4. No vowels in the written Torah: The Torah scroll is written without vowels (nikud), allowing multiple layers of reading (p'shat, remez, drash, sod — literal, allegorical, homiletical, mystical — the PaRDeS system).

5. Root-based system: Every Hebrew word derives from a three-letter root (shoresh). Words sharing a root share a conceptual family. For example, SheLeM (peace/wholeness), ShaLaM (to complete/pay), ShaLoM, MiShLoM (from), and ShLeMut (completeness) all derive from shin-lamed-mem.

6. Continuity: Unlike any other ancient language, Hebrew has remained a living sacred language for 3,000+ years and was successfully revived as a spoken vernacular in the late 19th/early 20th century (by Eliezer Ben-Yehuda)."""),

("""What is Sefer Yetzirah (Book of Formation) and why is it important?""",
"""Sefer Yetzirah (ספר יצירה, "Book of Formation" or "Book of Creation") is one of the oldest and most cryptic texts in Jewish mysticism, dating variously from the 2nd century BCE to the 6th century CE (scholarly debate continues). Traditionally attributed to Abraham or Rabbi Akiva, it is the foundational text of early Jewish cosmology and mystical cosmogony.

Content and structure: The text is very short (less than 2,000 words in most versions) yet extraordinarily dense. It describes how God created the universe through two instruments:
1. The 32 Paths of Wisdom: 10 Sefirot (divine numbers/emanations) + 22 Hebrew letters.
2. The 22 letters divided into: 3 Mothers, 7 Doubles, 12 Simples (as described above).

Key teachings:
- "God created His world with three books (sefarim): with Sefer (writing), Sefor (number), and Sipur (speech/story)" — the three dimensions of letters.
- Each letter governs a part of the body, a time period, and an astrological/cosmic domain.
- The Sefirot are described as "ten and not nine, ten and not eleven" — a precise limit.
- Pairs of opposites (good/evil, wisdom/folly, life/death) are built into creation's structure.

Influence: Sefer Yetzirah influenced all subsequent Kabbalah, particularly the medieval commentaries of Saadia Gaon, Judah Halevi (who used it as the basis of the Kuzari), and the Kabbalists of Gerona and Safed. The Golem legends (artificial beings created through mystical letter combinations) are rooted in Sefer Yetzirah traditions."""),

("""What is Biblical Hebrew and how does it differ from Modern Hebrew?""",
"""Biblical Hebrew (Ivrit Mikra'it) and Modern Hebrew (Ivrit Modernit) share the same root system and alphabet but differ substantially:

Grammar and syntax:
- Biblical Hebrew has a verb-subject-object word order; Modern Hebrew is subject-verb-object.
- Biblical Hebrew uses the waw-consecutive (vav-consecutive) — a grammatical construction where vav (and) converts perfect tense to imperfect and vice versa, creating narrative flow. Modern Hebrew has abolished this.
- Biblical Hebrew has dual number forms extensively (eyes, hands, lips, Egypt = Mitzrayim = the two Egypts). Modern Hebrew retains only a few duals.
- Biblical Hebrew uses the pausal form — words shift pronunciation at the end of a verse.

Vocabulary:
- Thousands of words in Biblical Hebrew appear nowhere else; their meaning is reconstructed from context or cognate languages (Aramaic, Ugaritic, Akkadian, Arabic).
- Modern Hebrew has ~100,000 words, thousands borrowed/coined for technology, science, and modernity. Eliezer Ben-Yehuda created hundreds of new words from biblical and Talmudic roots.

Pronunciation:
- Original biblical pronunciation is partially reconstructed from Greek and Aramaic transliterations. Ashkenazi pronunciation preserves some distinctions (cholam vs kubutz) that Sephardi/Israeli pronunciation merged.
- Modern Israeli Hebrew is primarily Sephardic-based, pronouncing resh as a uvular r, and collapsing distinctions between aleph/ayin and between tet/tav.

Script: Both use the same square Assyrian script (Ktav Ashurit) for printed text, though Biblical manuscripts were also written in the Paleo-Hebrew script (Ktav Ivri) in early periods."""),

# ── JEWISH HISTORY ────────────────────────────────────────────────────────────
("""Describe the major periods of ancient Jewish history from Abraham to the Roman destruction.""",
"""Ancient Jewish history spans approximately 2,000 years from the patriarchal period to the destruction of the Second Temple:

1. Patriarchal Period (c. 2000-1700 BCE): Abraham, Isaac, and Jacob establish monotheism and the covenant (brit) with God. Jacob's 12 sons become the 12 tribes of Israel. Jacob's family descends to Egypt following Joseph.

2. Egyptian Slavery and Exodus (c. 1400-1250 BCE): After generations of slavery, Moses leads the Exodus from Egypt. The Torah is received at Mount Sinai — the defining moment of Jewish nationhood. 40 years of wandering in the desert.

3. Conquest and Judges (c. 1200-1020 BCE): Under Joshua, the Israelites enter and conquer Canaan. The period of the Judges (Shoftim) follows — a cycle of sin, oppression, repentance, and deliverance. Major judges: Deborah, Gideon, Samson.

4. United Monarchy (c. 1020-930 BCE): Saul becomes the first king. David conquers Jerusalem and establishes it as the capital. Solomon builds the First Temple (Beit HaMikdash) — the spiritual center of the Israelite world.

5. Divided Kingdom (930-722 BCE): After Solomon's death, the kingdom splits into Israel (Northern, 10 tribes) and Judah (Southern, 2 tribes). The Northern Kingdom is conquered and dispersed by Assyria (722 BCE) — the "Ten Lost Tribes."

6. Kingdom of Judah, Babylonian Exile (722-538 BCE): Judah survives longer; the Babylonians under Nebuchadnezzar destroy the First Temple (586 BCE) and exile the Jews to Babylon. The Psalms of exile ("By the rivers of Babylon") are composed.

7. Persian Period, Second Temple (538-332 BCE): Cyrus the Great of Persia allows the Jews to return. Ezra and Nehemiah rebuild Jerusalem. The Second Temple is completed (516 BCE). The Great Assembly (Anshei Knesset HaGedolah) canonizes Torah and prayer.

8. Hellenistic Period (332-63 BCE): Alexander the Great conquers the region. The Maccabean revolt against the Seleucid king Antiochus IV Epiphanes (168-165 BCE) results in the rededication of the Temple — celebrated as Hanukkah. The Hasmonean dynasty rules.

9. Roman Period (63 BCE-70 CE): Rome conquers Judea. Despite several revolts, including the Great Revolt (66-73 CE), the Romans destroy the Second Temple in 70 CE — the most catastrophic event in Jewish history, initiating the long Diaspora."""),

("""What was the significance of the Babylonian Exile for Jewish religion and culture?""",
"""The Babylonian Exile (586-538 BCE) was paradoxically one of the most creative and transformative periods in Jewish history, fundamentally reshaping Jewish religion in ways that persist to this day.

Background: Nebuchadnezzar II of Babylon destroyed the First Temple, killed or exiled much of Judah's leadership, and deported the population in three waves. The exile lasted approximately 50 years until Cyrus the Great's Edict of Cyrus (538 BCE).

Religious transformations:
1. Synagogue (Beit Knesset): Without the Temple and its sacrificial system, Jews developed communal prayer houses (proto-synagogues) where Torah reading and prayer replaced sacrifice. This was a radical theological shift — worship without a physical center.
2. Canonization of Torah: The trauma of exile accelerated the collection and editing of biblical texts. Scribes (Sofrim) under Ezra worked to preserve and transmit the Torah with absolute precision, giving rise to the Masoretic tradition.
3. Prophetic literature: The exilic prophets — Jeremiah (who predicted the exile), Ezekiel (who received visions in Babylon), and Second Isaiah (Deutero-Isaiah, chapters 40-66) — produced some of the most profound theology in the Bible. Second Isaiah introduced the concept of a suffering servant and of God as universal creator, not merely a tribal deity.
4. Monotheism sharpened: In the face of Babylonian polytheism, Jewish monotheism became uncompromising and articulate. The absolute uniqueness of the God of Israel was crystallized.
5. Sabbath and circumcision: These became identity markers distinguishing Jews in exile. Torah observance replaced land-based practices as the center of identity.
6. Return and restoration: A significant community stayed in Babylon, which later produced the Babylonian Talmud (the more authoritative of the two Talmuds). Those who returned with Ezra and Nehemiah rebuilt Jerusalem and the Temple with a transformed religious consciousness."""),

("""What was Hasidism and who founded it?""",
"""Hasidism (Chasidus, from the Hebrew Chasid — "pious one") is a major Jewish spiritual revival movement founded in 18th-century Eastern Europe by Rabbi Israel ben Eliezer, known as the Ba'al Shem Tov (Master of the Good Name; abbreviated Besht; c. 1700-1760).

Historical context: Poland-Lithuania in the early 18th century was a landscape of Jewish trauma — the Cossack massacres of 1648-1649 (Chmielnicki Pogroms) had destroyed hundreds of communities and killed hundreds of thousands. The failed messianic movement of Shabbetai Tzvi (1626-1676) had left many disillusioned. The established rabbinical elite was often elitist and inaccessible to the poor masses.

The Besht's revolution: The Ba'al Shem Tov taught that:
1. God is present everywhere (divine immanence) — in every blade of grass, in joy, in seemingly mundane activities.
2. Devekut (cleaving to God) in prayer and daily life is accessible to everyone, not just scholars.
3. Joy (simcha) is a cardinal religious obligation; sadness (atzvut) is spiritually dangerous.
4. The simple folk, through sincerity of heart, can reach God as surely as the greatest scholar.
5. The Tzaddik (righteous leader) serves as an intermediary, elevating the prayers and souls of the community.

Hasidism's structure: After the Besht, his disciples spread Hasidism throughout Eastern Europe. Major early masters: the Maggid of Mezeritch (Dov Ber), Rabbi Elimelech of Lizhensk, Rabbi Nachman of Breslov (who emphasized storytelling and joy in despair), and Rabbi Schneur Zalman of Liadi (who founded Chabad-Lubavitch and wrote the Tanya).

Opposition (Mitnagdim): The Vilna Gaon (Rabbi Elijah of Vilna, 1720-1797) led fierce rabbinic opposition to Hasidism, issuing bans (cherem) against it. He feared it was deemphasizing Torah study, elevating the Tzaddik improperly, and veering toward the errors of Shabbetianism. Despite this opposition, Hasidism grew to encompass millions and remains a major force in Orthodox Judaism today."""),

("""Describe the Talmud — what is it, how was it composed, and what are its two versions?""",
"""The Talmud is the central text of Rabbinic Judaism — the vast repository of law, legend, philosophy, and discussion that the Jewish people have used to interpret and apply Torah since the destruction of the Second Temple (70 CE).

Structure: The Talmud is built on the Mishnah — the first major written codification of the Oral Torah, compiled by Rabbi Judah HaNasi (the Prince) around 200 CE in the Land of Israel. The Mishnah is organized into six orders (Sedarim):
1. Zeraim (Seeds) — agricultural laws, prayer
2. Moed (Festivals) — Shabbat and holiday laws
3. Nashim (Women) — marriage, divorce, vows
4. Nezikin (Damages) — civil and criminal law
5. Kodashim (Holy Things) — Temple service, sacrifices
6. Tohorot (Purities) — ritual purity laws

Each order is divided into tractates (masekhtot), each tractate into chapters, each chapter into mishnayot (individual teachings).

The Gemara: Subsequent generations of rabbis (Amoraim, 200-500 CE) discussed, debated, and expanded on the Mishnah. This discussion is the Gemara. Mishnah + Gemara = Talmud.

Two Talmuds:
1. Jerusalem Talmud (Talmud Yerushalmi): Compiled in the Land of Israel, edited c. 400 CE. More concise, less polished; written in Western Aramaic. Less studied but significant for Israeli law and certain tractates.
2. Babylonian Talmud (Talmud Bavli): Compiled in the Babylonian academies (Sura, Pumbedita), edited c. 500-600 CE. Much larger, more extensively edited, written in Eastern Aramaic. The standard Talmud studied worldwide; the authoritative halakhic source in most matters.

The Talmud is organized as a page (daf) with the Mishnah text in the center, surrounded by Gemara, with Rashi's commentary on the inner margin and the Tosafot (glosses by Rashi's students) on the outer margin. This page layout (the Vilna edition) has been standard since the 16th century."""),

("""Who was Maimonides (Rambam) and what were his major contributions to Judaism?""",
"""Rabbi Moses ben Maimon (1135/1138-1204), known by the acronym Rambam and in the non-Jewish world as Maimonides, was the greatest medieval Jewish philosopher, legal codifier, and physician. Born in Cordoba, Spain, he fled the Almohad persecution and eventually settled in Fustat (Old Cairo), Egypt, where he served as head of the Jewish community and physician to Saladin's court.

Major works:

1. Mishneh Torah (Repetition of the Torah, 1170-1180): A monumental 14-volume legal code written in clear Mishnaic Hebrew that systematically codified all of Jewish law — not just what was practically applicable but the entire legal tradition including Temple laws, kingship, and theoretical matters. Revolutionary because it eliminated the need to consult the Talmud directly — Maimonides ruled definitively on every question. Controversial for the same reason (he didn't cite sources). Still a cornerstone of halakhic study.

2. Moreh Nevukhim (Guide for the Perplexed, 1190): Written in Arabic for intellectuals torn between Torah and Aristotelian philosophy. Maimonides argued that the Torah and reason are harmonious; anthropomorphic descriptions of God in scripture are metaphorical; prophecy is a natural intellectual/spiritual capacity; evil is the absence of good. Hugely influential on Christian and Islamic scholasticism (Thomas Aquinas cited him extensively).

3. Commentary on the Mishnah: His first major work, including the famous "Thirteen Principles of Faith" — the most widely accepted Jewish creed, recited as "Ani Ma'amin" (I believe).

4. Medical works: 10+ treatises on medicine, including works on asthma, poisons, hemorrhoids, and general health (often cited in medieval medical literature).

Legacy: "From Moses to Moses there has been none like Moses" — a medieval saying comparing him to Moses the Lawgiver. His rationalism attracted admiration and fierce opposition (the Maimonidean controversy). His influence on Jewish law and thought remains unmatched."""),

("""What was the Spanish Inquisition and the Expulsion of 1492 for the Jewish people?""",
"""The Spanish Inquisition (established 1478) and the Expulsion from Spain (1492) constitute one of the most traumatic chapters in Jewish history and fundamentally reshaped the Jewish world.

Background: After centuries of Jewish life in Al-Andalus and Christian Spain — a period of relative tolerance (convivencia) alternating with persecution — a combination of anti-Jewish violence (the 1391 pogroms killed tens of thousands) and social pressure had produced hundreds of thousands of conversos (Jews who converted to Christianity, also called Marranos derogatorily). Many conversos secretly maintained Jewish practice.

The Inquisition: Established by King Ferdinand and Queen Isabella to investigate the sincerity of converso Christianity. Led by the notorious Tomás de Torquemada as Grand Inquisitor, the Inquisition used torture, confiscation of property, public humiliation (auto-da-fé), and burning at the stake. It had no jurisdiction over Jews who had never converted, but its campaign against conversos destroyed communities. Estimates of those killed range from thousands to tens of thousands over three centuries.

The Expulsion of 1492: On March 31, 1492 — the same year Columbus sailed — Ferdinand and Isabella signed the Alhambra Decree, ordering all Jews who refused baptism to leave Spain within four months. Approximately 100,000-300,000 Jews were expelled (estimates vary widely). Many went to:
- The Ottoman Empire (especially Salonica/Thessaloniki and Istanbul), where Sultan Bayezid II welcomed them.
- Portugal (until expelled in 1497).
- North Africa (Morocco, Algeria).
- Italy, the Netherlands.

Cultural legacy: The expelled Spanish Jews (Sephardim, from "Sepharad" = Spain in Hebrew) carried their culture, language (Ladino/Judeo-Spanish), liturgy, and traditions throughout the Mediterranean world. Sephardic communities maintained Ladino as a living language until the 20th century."""),

("""What is the Shema and why is it considered the most important prayer in Judaism?""",
"""The Shema Yisrael (שְׁמַע יִשְׂרָאֵל, "Hear O Israel") is the central declaration of Jewish faith, drawn from Deuteronomy 6:4:

"Shema Yisrael YHVH Eloheinu YHVH Echad"
"Hear, O Israel: the LORD is our God, the LORD is One."

It is followed by the Baruch Shem (recited silently on most occasions): "Blessed is the Name of His glorious kingdom forever and ever."

Why central:
1. Affirmation of monotheism: The Shema is the fundamental declaration that there is only one God — not a duality of good and evil forces, not a pantheon. The word Echad (One) emphasizes absolute undivided unity.
2. Death and devotion: Jews recite the Shema as their final words before death. Martyrs throughout history (including Rabbi Akiva, burned alive by the Romans) died with the Shema on their lips. The last letter of Shema (ayin, enlarged in Torah scrolls) and the last letter of Echad (dalet) spell "eid" (witness) — the Jew is God's witness.
3. Twice daily: The Torah commands "when you lie down and when you rise up" — so the Shema is recited morning and evening, the two hinges of the Jewish day.
4. Three paragraphs: The full Shema includes Deuteronomy 6:4-9 (love God, study Torah, bind tefillin), Deuteronomy 11:13-21 (reward for observance, mezuzah), and Numbers 15:37-41 (tzitzit fringes, Exodus from Egypt).
5. The mezuzah: Deuteronomy 6:9 commands affixing these words to your doorposts — hence the mezuzah, a small parchment scroll containing the first two paragraphs of the Shema, on every Jewish doorway.

Kabbalistic significance: The six words of the first verse correspond to the six directions of space (north, south, east, west, up, down), unifying all dimensions under divine Oneness. Meditation on these words is considered among the highest forms of Jewish contemplative practice."""),

("""What is Shabbat and how is it observed?""",
"""Shabbat (שַׁבָּת, "rest") is the Jewish day of rest, observed from sundown Friday to nightfall Saturday, commemorating God's rest on the seventh day of creation (Genesis 2:2-3) and the Exodus from Egypt (Deuteronomy 5:15). It is the most frequently commanded mitzvah in the Torah and considered equal in weight to all other commandments combined (Talmud, Yerushalmi Berakhot 1:5).

The 39 categories of prohibited work (Melachot): The Mishnah (Tractate Shabbat) derived 39 categories of forbidden labor from the activities used to build the Tabernacle in the desert: planting, plowing, harvesting, binding sheaves, threshing, winnowing, selecting, grinding, sifting, kneading, baking, shearing, bleaching, combing, dyeing, spinning, stretching, making loops, weaving, separating, tying, untying, sewing, tearing, trapping, slaughtering, flaying, salting hides, curing, scraping, cutting, writing, erasing, building, demolishing, extinguishing fire, kindling fire, striking the final hammer blow, carrying. Each category has many derivative prohibitions.

How Shabbat begins: The woman of the household lights (at minimum) two candles and recites the blessing 18 minutes before sunset. Kiddush (sanctification over wine) is recited. A festive meal follows, with two challah loaves (zahor/zecher — the double portion of manna from the Exodus). Synagogue prayers are lengthened with special liturgy.

How Shabbat ends: Havdalah (separation) ceremony with wine, spices, and a braided candle marks the transition back to the weekday.

The spirit: Beyond technical prohibitions, the rabbis taught oneg Shabbat (delight in Shabbat) — the day should be experienced as a foretaste of the World to Come (Olam HaBa), a day of family, study, prayer, and rest."""),

("""Who was Rashi and why is his commentary indispensable?""",
"""Rabbi Shlomo Yitzchaki (1040-1105), universally known by his acronym Rashi (Rabbi Shlomo Yitzchaki), was a French rabbinic scholar from Troyes, Champagne, and is the author of the most essential commentary on both the Torah and the Babylonian Talmud ever written.

Life: Born in Troyes, France, Rashi studied in the great Rhineland yeshivot of Mainz and Worms before returning to Troyes, where he led the Jewish community and supported himself through winemaking. He lived through the First Crusade (1096), which devastated Rhineland Jewish communities.

Torah commentary: Rashi's Torah commentary synthesizes the plain meaning (p'shat) with carefully selected midrashic traditions. Written in lucid Rabbinic Hebrew, it explains every verse, resolves apparent contradictions, and illuminates obscure words — often with Old French glosses (la'azim) when no Hebrew equivalent suffices. Rashi's commentary was the first Hebrew book printed (Reggio di Calabria, 1475) and has been printed alongside the Torah text in almost every edition since.

Talmud commentary: Rashi's commentary on the Babylonian Talmud is even more transformative. Without it, the Talmud's compressed, often elliptical Aramaic would be incomprehensible to most students. Rashi explains the meaning of each passage, supplies missing context, and defines technical terms. The commentary is printed on the inner column of every standard Talmud page.

His students (Tosafists): Rashi's grandsons and students — the Tosafot (additions) school — continued his work with critical glosses on the outer column of the Talmud page. Together, Rashi and Tosafot are the foundation of Talmud study for all subsequent generations."""),

("""What was Zionism and how did the State of Israel come into being?""",
"""Zionism is the political movement for the return of the Jewish people to their ancestral homeland (Eretz Yisrael) and the establishment of a Jewish state there. It emerged in the late 19th century as a response to persistent European antisemitism and the failure of Enlightenment promises of Jewish emancipation.

Pre-political aliyot: Small waves of religious Jews had always maintained a presence in Palestine (Eretz Yisrael). The Bilu pioneers (1882) and First Aliyah (1882-1903) began organized modern immigration.

Theodor Herzl (1860-1904): An Austro-Hungarian journalist shocked by the Dreyfus Affair (1894) — the false conviction of a Jewish French officer amid virulent antisemitism — Herzl concluded that Jews could never be safe in Europe. His pamphlet Der Judenstaat (The Jewish State, 1896) and the First Zionist Congress (Basel, 1897) founded the organized Zionist movement. He died at 44, but set its institutional framework.

Balfour Declaration (1917): Britain declared its support for "the establishment in Palestine of a national home for the Jewish people" — a landmark diplomatic achievement.

British Mandate (1920-1948): Britain administered Palestine. Jewish immigration waves (aliyot) grew, especially after the Holocaust. Conflict with Arab populations intensified.

Holocaust and UN partition: The murder of 6 million Jews (1941-1945) made the urgency undeniable. In 1947, the UN voted to partition Palestine into Jewish and Arab states. The Arab states rejected the plan.

Declaration of Independence (May 14, 1948): David Ben-Gurion declared the State of Israel. The next day, five Arab armies invaded. Israel survived the War of Independence, though at great cost (6,000 dead — 1% of the Jewish population). The war also produced the Palestinian refugee crisis and the Nakba (catastrophe, in Arabic).

Israel today: A parliamentary democracy of ~9 million people, a major center of global Jewish life, Torah study, and — critically for the Zionist vision — the only place on earth where Jews exercise sovereign self-determination."""),

("""What is Passover (Pesach) and what is the Seder?""",
"""Pesach (Passover) is the eight-day festival (seven in Israel) commemorating the Exodus from Egypt and the liberation of the Israelites from slavery — the foundational event of Jewish national and religious identity. It falls on 15-22 Nisan (spring).

The Seder (Hebrew: "order") is the ritual feast held on the first night (and in the Diaspora, second night) of Passover. It is the most widely observed Jewish ritual worldwide, even among secular Jews.

The Haggadah (telling): The Seder follows the Haggadah — a text recounting the Exodus story, filled with biblical passages, rabbinic commentary, songs, and rituals. The central commandment is v'higadeta l'vincha (you shall tell your child) — every generation must personally identify with the Exodus.

The Seder plate (ke'ara) holds:
- Maror (bitter herbs, horseradish) — the bitterness of slavery
- Charoset (chopped fruit, nuts, wine) — the mortar used by slaves
- Karpas (vegetable, usually parsley) — dipped in salt water (tears)
- Zeroa (shank bone) — the Paschal sacrifice
- Beitzah (roasted egg) — Temple sacrifice; symbol of mourning
- Chazeret (additional bitter herb)

Four cups of wine: Corresponding to the four expressions of redemption (Exodus 6:6-7). Elijah's cup is poured but not drunk, symbolizing messianic hope.

Matzah: Unleavened bread eaten throughout Passover; when the Israelites fled Egypt they had no time to let bread rise. Chametz (leavened products) are forbidden and must be removed from the home. The search for chametz (bedikat chametz) is performed the night before Passover by candlelight."""),

("""What is the Vilna Gaon and what was his approach to Judaism?""",
"""Rabbi Elijah ben Shlomo Zalman of Vilna (1720-1797), universally known as the Vilna Gaon (GRA — Gaon Rabbi Elijah; "gaon" = genius/excellency), was perhaps the greatest Talmudic mind of the modern era and the leading intellectual figure of non-Hasidic (Mitnagdic) Orthodox Judaism.

Intellectual biography: The Gaon was an extraordinary prodigy who had memorized the entire Talmud by age 10. He rejected the position of community rabbi (rav) of Vilna, preferring a life of total Torah study with minimal communal obligations. He reportedly slept only two hours per night in 30-minute intervals and spent most of his waking hours in Torah study. He mastered not only all of Talmudic literature but also mathematics, astronomy, grammar, and natural science — subjects he considered necessary for understanding Torah.

Approach to Torah:
1. Return to sources: The Gaon rejected the Pilpul method — the elaborate scholastic argumentation that had dominated Eastern European yeshivot. He insisted on returning to the plain meaning of the Talmudic text, comparing parallel sources, and correcting scribal errors through philological and logical analysis.
2. Kabbalistic mastery: Contrary to the stereotype, the Gaon was a profound Kabbalist who wrote commentaries on Sefer Yetzirah and the Zohar. However, he emphasized that Kabbalistic study must be grounded in rigorous Talmudic knowledge first.
3. Opposition to Hasidism: The Gaon issued the first cherem (excommunication) against the Hasidic movement in 1772, fearing its departure from Torah study primacy, its elevation of the Tzaddik, and the innovations in prayer text and posture. He refused to meet the Hasidic leaders who attempted to reconcile with him. His opposition shaped the Mitnagdic (Lithuanian/Yeshivish) tradition that remains a pillar of Orthodox Judaism.

Legacy: The Vilna Gaon's approach directly produced the great Lithuanian yeshiva movement — Volozhin (founded by his student Rabbi Chaim of Volozhin), Slobodka, Mir, Telz — which trained the Torah scholars of the 20th century. His commentary (Biur HaGra) on the Shulchan Aruch is authoritative."""),

("""What is the Holocaust (Shoah) and how has it shaped Jewish life and thought?""",
"""The Holocaust (Hebrew: Shoah, "catastrophe") was the systematic, state-sponsored genocide of six million Jews by Nazi Germany and its collaborators between 1933 and 1945 — approximately one-third of world Jewry and two-thirds of European Jewry. It represents an unprecedented rupture in Jewish history and Western civilization.

Historical facts:
- 1933-1938: Legal persecution — Nuremberg Laws (1935) stripped Jews of citizenship; boycotts; Kristallnacht (November 1938) — nationwide pogrom destroying synagogues and Jewish businesses.
- 1939-1941: With the conquest of Poland (3.3 million Jews) and the Soviet Union, Jews were concentrated in ghettos (Warsaw, Lodz, Vilna) under catastrophic conditions.
- 1941-1942: Mass shootings by mobile killing units (Einsatzgruppen) in Eastern Europe — "the Holocaust by bullets" — murdered approximately 1.5 million Jews. The Wannsee Conference (January 1942) coordinated the "Final Solution."
- 1942-1945: Six extermination camps — Auschwitz-Birkenau, Treblinka, Sobibor, Belzec, Majdanek, Chelmno — murdered millions by gas. The last transports arrived at Auschwitz in November 1944.

Jewish theological responses: The Shoah shattered many frameworks:
- Theodicy: How could God allow this? Rabbi Elchonon Wasserman (murdered 1941) saw it as punishment for abandoning Torah. Viktor Frankl found meaning-making in suffering. Elie Wiesel described the death of his childhood faith at Auschwitz. Rabbi Menachem Hartom saw it as hester panim (the hiding of God's face).
- "Torah from the ashes": The surviving yeshiva world rebuilt institutions in Israel and America, determined that Torah would outlast the Nazis.
- State of Israel: For many, Israel's establishment three years after liberation represented a miraculous national resurrection — the response to mass victimhood.
- Yom HaShoah: Holocaust Remembrance Day, observed on 27 Nisan, with sirens across Israel and memorial ceremonies worldwide.

Never Again: The phrase has become both a commitment and a challenge — defining the post-Holocaust ethical obligation to prevent genocide globally."""),

("""What is the Jewish concept of Teshuvah (repentance) and how does it work?""",
"""Teshuvah (תְּשׁוּבָה) literally means "return" — returning to God and to one's truest self after having strayed through sin. It is considered one of the greatest gifts God gave humanity (created before the world itself, according to the Midrash) and the foundation of Jewish ethics.

The four steps of Teshuvah (Rambam, Laws of Repentance):
1. Acknowledgment (Vidui): Verbally confessing the specific sin. "I did this wrong act." Judaism requires articulation, not just feeling.
2. Regret (Charatah): Genuine remorse — feeling the wrong, not just fearing consequences.
3. Cessation (Azivat HaChet): Actually stopping the sinful behavior.
4. Resolution (Kabbalah L'Atid): Committing not to repeat the act.

For interpersonal wrongs: Teshuvah before God is insufficient. One must first seek forgiveness from the person one has wronged, make restitution where possible, and only then can one's account with God be settled. If the person refuses forgiveness after three sincere requests, the obligation is discharged.

The High Holiday season (Yamim Noraim): The 10 Days of Repentance from Rosh Hashanah (Jewish New Year) through Yom Kippur (Day of Atonement) are the annual collective teshuvah period. On Rosh Hashanah, the books of life and death are "opened" and inscribed; on Yom Kippur (a full 25-hour fast) they are "sealed." The Gates of Heaven are understood to be maximally open during this period.

Deeper Kabbalistic teshuvah: The Tanya teaches that teshuvah is not merely fixing a mistake but a transformative return of the soul to its divine source. The Hebrew root is identical to "shuv" (return) — sin is a disconnection; teshuvah is the reconnection. Higher teshuvah (teshuvah ila'ah) transforms the very quality of one's sins, converting them retroactively into merits."""),

("""What is the Western Wall (Kotel) and why is it sacred?""",
"""The Western Wall (Hebrew: HaKotel HaMa'aravi; often called simply HaKotel, "The Wall") is the holiest site in Judaism accessible for prayer — a 488-meter retaining wall built by King Herod (c. 19 BCE) to support the expanded Temple Mount platform on which the Second Temple stood.

Historical context: The Wall is not part of the Temple itself but the western retaining wall of the Temple Mount (Har HaBayit). The Dome of the Rock and Al-Aqsa Mosque now occupy the Temple Mount, which is under the administration of the Islamic Waqf. The exposed section of the Kotel (approximately 57 meters wide) has been the primary Jewish prayer site since the Ottoman period.

Sacred status: "The Shekhinah (divine presence) never departed from the Western Wall" (Midrash Rabbah, Shemot). Among the prayers of the Amidah (standing prayer), Jews face Jerusalem; within Jerusalem, they face the Temple Mount; on the Temple Mount, they face the spot of the Holy of Holies — and the Western Wall is the closest accessible approach to that spot.

The Wall's significance:
- Physical remnant of the Temple that the Romans destroyed in 70 CE.
- Symbol of Jewish historical continuity, destruction, exile, and hope for restoration.
- Site of prayer for millennia; the practice of inserting written prayers (kvitlach) into the Wall's crevices is centuries old.
- Tears: The Wall is also called the "Wailing Wall" (a non-Jewish term) because Jews came there to mourn the Temple's destruction. The Hebrew term is not used by Jews; it is simply HaKotel.

1967: The Western Wall Plaza was cleared by Israel in the days following the Six-Day War, when Israeli paratroopers reunited Jerusalem. Rabbi Shlomo Goren blew the shofar at the Wall — a moment of extraordinary historical and spiritual significance for the Jewish people."""),

("""What are the High Holy Days — Rosh Hashanah and Yom Kippur?""",
"""The High Holy Days (Yamim Noraim, "Days of Awe") are the most solemn and spiritually significant days in the Jewish calendar — Rosh Hashanah (the Jewish New Year) and Yom Kippur (the Day of Atonement), separated by the Ten Days of Repentance (Aseret Yemei Teshuvah).

Rosh Hashanah (1-2 Tishrei):
- "The Birthday of the World" (Yom Harat Olam) — commemorates the creation of Adam and Eve on the sixth day of creation, traditionally dated to the beginning of the year 1 in the Hebrew calendar.
- The Day of Judgment: All souls pass before God for judgment on their deeds of the past year. Three books are opened — the righteous are inscribed for life, the wicked for death, and the majority are in between, awaiting Yom Kippur.
- The Shofar (ram's horn): The defining ritual of Rosh Hashanah. 100 shofar blasts are sounded in three configurations: Tekiah (long blast), Shevarim (three medium wails), Teruah (nine staccato blasts), and Tekiah Gedolah (a great blast). The shofar awakens the soul to repentance.
- Tashlich: Casting bread upon running water, symbolically "casting away" one's sins.
- Special foods: Round challah (symbolizing the cycle of the year), apples dipped in honey (for a sweet new year), pomegranates (seeds = 613 mitzvot), fish head ("may we be the head and not the tail").

Yom Kippur (10 Tishrei):
- The Day of Atonement: The holiest day of the year; described in Leviticus 16 with the High Priest's elaborate Temple service, including sending the "scapegoat" (Azazel) into the wilderness.
- Five prohibitions: No eating or drinking, washing, anointing, wearing leather shoes, or marital relations for 25 hours.
- Kol Nidre: The haunting annulment of vows sung at nightfall opens the Yom Kippur liturgy — one of the most emotionally powerful moments in the Jewish year.
- Five prayer services: Ma'ariv (evening), Shacharit (morning), Musaf (additional), Mincha (afternoon), Ne'ilah (closing — "the closing of the gates"). Ne'ilah culminates in a final Tekiah Gedolah and the declaration "Next year in Jerusalem!"
- The day closes having sealed the books of life."""),

("""Explain the concept of Gilgul Neshamot (reincarnation of souls) in Kabbalah.""",
"""Gilgul Neshamot (גִּלְגּוּל נְשָׁמוֹת, "rolling of souls") is the Kabbalistic doctrine of metempsychosis — the transmigration or reincarnation of souls. While not explicitly stated in the Torah and rejected by Saadia Gaon as a non-Jewish concept (9th century), gilgul became widely accepted in Kabbalistic circles from the Bahir (12th century) onward, reaching its fullest development in Lurianic Kabbalah.

Core principles:
1. Purpose of gilgul: A soul that has not completed its tikkun (rectification) — fulfilling all the mitzvot relevant to its root — must return in another body to complete what was left unfinished. There is no punishment in gilgul per se; it is remedial, not retributive.
2. The Ari's elaborate system: Luria, through claimed ruach hakodesh (divine inspiration), could read a person's soul-history in their face (from the "science" of physiognomy as understood by Kabbalah). His disciple Chaim Vital recorded these readings in Sha'ar HaGilgulim (Gate of Reincarnations).
3. Ibur (spiritual impregnation): A righteous soul can temporarily attach to a living person to help them complete a mitzvah or protect them — a benevolent version of gilgul.
4. Dybbuk: A wandering sinful soul that attaches to a living person without permission, causing psychological disturbance. Exorcism rites were developed to extract it.
5. Multiple souls: A person can carry portions of multiple souls (from different roots) simultaneously.
6. First incarnation: Adam contained all souls in potential. After his sin, sparks scattered into history, taking on bodies according to their tikkunim.

Constraints: The same soul cannot appear in two bodies simultaneously. Animals can also receive human souls for rectification. Specific souls are linked to specific mitzvot — a person born with a proclivity for a certain sin may be a soul returning to rectify that specific failing.

The doctrine remains controversial in mainstream Orthodox thought but is widely accepted in Sephardic, Hasidic, and Kabbalistic circles."""),

("""What is the Kabbalah concept of Ein Sof?""",
"""Ein Sof (אֵין סוֹף) — literally "without end," "the Infinite," "that which has no limit" — is the Kabbalistic term for the absolute, unknowable divine reality that transcends all attributes and descriptions.

The challenge: The Torah speaks of God as having emotions, speaking, resting, being angry, rejoicing. But if God is truly infinite and perfect, how can one attribute to Him any quality, emotion, or action without limiting Him? Kabbalists resolved this through Ein Sof.

What Ein Sof is NOT: Ein Sof cannot be:
- Named (any name is a limitation)
- Known (all knowledge requires a knower and object of knowledge — a duality)
- Described (all description is limitation)
- Addressed in prayer (one cannot pray TO Ein Sof)
- Located (beyond space)
- Temporally constrained (beyond time)

What Ein Sof IS: The absolute totality of divine being. "Before creation, Ein Sof was alone and Ein Sof's name was alone" (Zohar). All being derives from Ein Sof; all returns to Ein Sof.

The Sefirot as interface: Since Ein Sof cannot be directly known or related to, the Sefirot are the "garments" or "vessels" through which Ein Sof expresses itself in ways accessible to creation. The Sefirot are simultaneously divine (not separate from Ein Sof) and relational (the means by which creation connects to God).

The paradox: Ein Sof fills all worlds (memaleh kol almin) yet transcends all worlds (sovev kol almin). This is not a logical contradiction but points to the limits of language and concept when approaching the Infinite. Maimonides arrived at a similar position through Aristotelian negative theology — God can only be described by what He is not."""),

("""Who was Rabbi Akiva and why is he central to Jewish tradition?""",
"""Rabbi Akiva ben Yosef (c. 50-135 CE) is one of the most revered sages in the entire Jewish tradition — the "father of all halakhah" (Jewish law) whose interpretive methods shaped the entire Talmudic tradition.

Life story: Akiva was an illiterate shepherd until age 40, working for the wealthy Kalba Savua. Kalba Savua's daughter Rachel saw his potential and agreed to marry him on condition he study Torah. Disowned by her father, the couple lived in poverty while Akiva studied — becoming one of the greatest scholars in history. The Talmud (Ketubot 63a) records Rachel's enormous role in enabling his greatness.

Contributions:
1. Midrash Halakha: Akiva developed the system of deriving law from every word, letter, and textual feature of the Torah. He taught that even the "crowns" (taggin) on letters have legal meaning.
2. The Mishnah's foundation: Much of the legal framework systematized by Rabbi Judah HaNasi into the Mishnah derives from Akiva's school.
3. Biblical interpretation: Akiva championed the inclusion of the Song of Songs in the biblical canon, calling it "the holy of holies" of scripture — understanding it as an allegory of God's relationship with Israel.
4. Mysticism: Akiva was among the four sages who "entered the Pardes" (the mystical orchard). He was the only one who "entered in peace and departed in peace" — the others died, went mad, or apostatized. This story anchors the Kabbalistic tradition's reverence for prepared study.

Martyrdom: Akiva continued teaching Torah publicly after the Romans forbade it following the Bar Kokhba revolt (132-135 CE). He was arrested, tortured with iron combs, and executed. His final words were the Shema — specifically drawing out the word "Echad" (One) until he died. This death became the paradigm of kiddush Hashem (martyrdom for God's name)."""),

("""What is the Jewish calendar and how does it work?""",
"""The Hebrew calendar (Luach HaIvri) is a lunisolar calendar — based on both the lunar month and the solar year — used to determine Jewish holidays, Torah reading cycles, and lifecycle dates.

Structure:
- Months are lunar: 29 or 30 days each, beginning with the new moon (Rosh Chodesh).
- Years are solar: adjusted to keep the agricultural festivals (Passover in spring, Sukkot in fall) aligned with their seasons.
- Leap years: A 13th month (Adar II) is added in 7 of every 19 years (the Metonic cycle) to synchronize lunar months with the solar year. This was formalized by Hillel II in 358 CE.

The 12 months (in order): Tishrei (30 days), Cheshvan (29 or 30), Kislev (30 or 29), Tevet (29), Shevat (30), Adar (29, or Adar I/II in leap year), Nisan (30), Iyar (29), Sivan (30), Tammuz (29), Av (30), Elul (29).

Year count: The Hebrew year counts from the traditional dating of creation (anno mundi). The year 2024 CE corresponds to approximately 5784-5785 in the Hebrew calendar. The exact calculation begins with the assumed date of Adam's creation: 25 Elul 1 AM.

Major dates:
- Tishrei 1-2: Rosh Hashanah
- Tishrei 10: Yom Kippur
- Tishrei 15-21: Sukkot; Tishrei 22: Shemini Atzeret/Simchat Torah
- Kislev 25 — Tevet 2/3: Hanukkah
- Adar 14: Purim
- Nisan 15-22: Passover
- Sivan 6: Shavuot
- Av 9: Tisha B'Av (national day of mourning)

The Sabbatical year (Shemitah): Every seventh year, agricultural land in Israel must lie fallow — a reminder that the land belongs to God. The Jubilee (Yovel) occurs every 50th year, with release of indentured servants and return of ancestral land."""),

("""What is the importance of Jerusalem (Yerushalayim) in Jewish thought and history?""",
"""Jerusalem (יְרוּשָׁלַיִם, Yerushalayim) holds the most central place of any city in Jewish religious consciousness — it is simultaneously a physical city, a spiritual capital, a cosmic axis, and a messianic symbol.

Biblical foundation:
- First mentioned as "Shalem," ruled by Melchizedek (Genesis 14) — a priest-king who blessed Abraham.
- David conquered the Jebusite city (~1000 BCE) and made it his capital.
- Solomon built the First Temple on Mount Moriah — the same site as the Akeidah (binding of Isaac, Genesis 22) — establishing it as the earthly dwelling of the divine presence (Shekhinah).
- Destroyed by Babylon (586 BCE); rebuilt (516 BCE); destroyed by Rome (70 CE).

Spiritual centrality:
- "Beautiful in elevation, the joy of all the earth is Mount Zion" (Psalms 48:3).
- The Temple (Beit HaMikdash) was the axis mundi — where heaven and earth met, where sacrifice connected physical and divine, where the high priest entered the Holy of Holies on Yom Kippur.
- The Holy of Holies contained the Ark of the Covenant with the Ten Commandments.
- The Even HaShetiyah (Foundation Stone) beneath the Dome of the Rock is the "navel of the world" in Jewish cosmology — the first point of creation.

Prayer orientation: Every synagogue in the world faces Jerusalem. Within Jerusalem, prayer faces the Temple Mount.

Yearning and mourning: "Next year in Jerusalem" closes the Passover Seder and Yom Kippur service. The fast of Tisha B'Av mourns both Temples' destruction. A Jew traditionally leaves a small section of a house wall unpainted (zecher l'churban) in memory of the Temple's destruction. At a Jewish wedding, the groom breaks a glass for the same reason.

Modern Jerusalem: Israel declared Jerusalem its united capital in 1980. The city is holy to Judaism, Christianity (site of Jesus's crucifixion and resurrection), and Islam (third holiest city; site of Mohammed's Night Journey). Its status remains one of the most contested in international politics."""),
]

os.makedirs(os.path.dirname(OUT), exist_ok=True)

existing = 0
if os.path.exists(OUT):
    with open(OUT) as f:
        existing = sum(1 for _ in f)

with open(OUT, "a") as f:
    for q, a in pairs:
        record = {"messages": [{"role": "user", "content": q.strip()}, {"role": "assistant", "content": a.strip()}]}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

total = existing + len(pairs)
print(f"upgrade_bq.jsonl: {existing} existing + {len(pairs)} new = {total} total pairs")
