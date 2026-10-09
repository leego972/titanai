# Titan Open-Source Provenance Registry v1

## Policy

Titan datasets should use as many high-quality open/public sources as practical, but only where commercial ML use and redistribution of derived training material are compatible with the source terms.

The dataset must not copy long passages from source documents. Sources are used to ground facts, control principles, taxonomies, public statistics and scenario design. Training prompts and answers are independently authored.

Each generated record should store `source_family`, source version/date where material, and an `as_of` date for volatile facts.

## Source status labels

- `APPROVED_OPEN`: suitable for commercial derived training use subject to stated attribution/license terms.
- `APPROVED_PUBLIC_DOMAIN`: public-domain/CC0 material suitable for derived training use.
- `CONDITIONAL`: usable only after item/series-level license verification.
- `EXCLUDE_FOR_TRAINING`: terms prohibit or materially restrict ML training or commercial reuse.

## Cybersecurity

### NIST
Status: APPROVED_PUBLIC_DOMAIN for U.S. government-authored material, with source attribution retained.
Use for: CSF 2.0, SP 800-series, SSDF, identity, incident response, risk management, cryptographic and security-control concepts.
Record source IDs and publication revisions.

### CISA
Status: APPROVED_PUBLIC_DOMAIN for U.S. government-authored material unless an item indicates third-party rights.
Use for: Cybersecurity Performance Goals, Secure by Design, incident response, phishing, ransomware resilience, vulnerability prioritisation, logging and operational defensive guidance.

### OWASP
Status: APPROVED_OPEN where project license permits commercial derivative use; preserve required attribution/share-alike obligations at the source/derived-artifact level as applicable.
Use for: ASVS, Top 10, API Security, SAMM, testing and secure-development concepts.
ASVS 5.0 is CC BY-SA 4.0.

### MITRE ATT&CK
Status: APPROVED_OPEN under MITRE ATT&CK terms permitting research, development and commercial use with the required copyright designation/license notice.
Use for: adversary-behaviour taxonomy, detection reasoning, defensive coverage and threat-informed scenarios.
Do not turn ATT&CK-derived records into procedural criminal intrusion instructions.

### MITRE CWE
Status: CONDITIONAL until the applicable CWE terms/version are recorded in the build manifest.
Use for: software weakness taxonomy and secure-development/root-cause examples.

### CVE / NVD
Status: CONDITIONAL.
Use for: vulnerability identifiers, public vulnerability metadata and remediation/evidence exercises only after feed/API terms and any third-party text rights are checked for the specific ingestion method.

### FIRST EPSS
Status: CONDITIONAL until current data/license terms are recorded.
Use for: probabilistic vulnerability prioritisation and risk-ranking exercises.

## General knowledge / structured facts

### Wikidata
Status: APPROVED_PUBLIC_DOMAIN.
Structured data in main/property/lexeme namespaces is CC0.
Use for: entities, geography, dates, relationships, identifiers and cross-domain factual scaffolding.
Volatile facts require an as-of date and should often be reserved for retrieval/evaluation rather than static weight training.

### Smithsonian Open Access
Status: APPROVED_PUBLIC_DOMAIN for items explicitly marked CC0.
Use for: history, art, culture, natural history, science and object metadata.
Only ingest assets/metadata with the CC0 designation; item-level third-party/privacy/publicity restrictions must still be respected.

### Europeana metadata
Status: APPROVED_PUBLIC_DOMAIN for Europeana metadata released under CC0.
Use for: cultural heritage/history metadata and entity relationships.
Underlying digital objects have separate rights statements and must not be assumed open merely because metadata is CC0.

### UNdata
Status: APPROVED_OPEN under stated terms permitting copying, duplication and further distribution with UNdata citation.
Use for: demographic, economic and international statistical reasoning where the contributing database's terms do not impose an additional restriction.
Record the original database/source when available.

### World Bank Open Data
Status: APPROVED_OPEN for World Bank-produced datasets explicitly licensed CC BY 4.0; CONDITIONAL for datasets carrying another license.
Use for: economics, development, demographics, infrastructure, education and cross-country quantitative reasoning.
Attribution and changes must be recorded.

### Our World in Data
Status: CONDITIONAL.
OWID-produced data may be CC BY, but much of the site republishes third-party data under the original provider's terms.
Only ingest rows whose underlying source/license has been individually approved.

### OpenStreetMap
Status: CONDITIONAL due ODbL/share-alike/database obligations.
Use only through a separately reviewed geospatial pipeline where attribution and derived-database obligations are satisfied.
Do not casually mix OSM database extracts into a commercially redistributed corpus without license review.

## Science / earth / environment / space

### USGS
Status: APPROVED_PUBLIC_DOMAIN for USGS-authored government data/information; inspect individual assets for third-party rights.
Use for: geology, earthquakes, hydrology, geography, natural hazards, maps and earth-science quantitative tasks.

### NOAA
Status: APPROVED_PUBLIC_DOMAIN for qualifying U.S. federal data/products, with item-level rights verification.
Use for: weather, climate, oceans, atmospheric science and measurement reasoning.

### NASA
Status: APPROVED_PUBLIC_DOMAIN for qualifying U.S. government works, with item-level checks for third-party material, logos and imagery restrictions.
Use for: astronomy, spaceflight, earth observation, planetary science and engineering context.

### data.gov / agency open-data portals
Status: CONDITIONAL by dataset.
Use when the individual dataset is explicitly public domain, CC0, CC BY or another commercial-compatible open license.

## Markets / finance / trading / crypto

### SEC EDGAR and SEC structured filings/data
Status: CONDITIONAL pending endpoint/data-specific terms and issuer filing-content rights review.
Preferred use: facts/structured calculations derived from public filings rather than reproducing narrative filing text.

### CFTC public market data
Status: CONDITIONAL pending dataset/API-specific terms.
Use for: derivatives-market structure, positioning/statistical exercises and regulatory concepts where permitted.

### World Bank open financial/economic indicators
Status: APPROVED_OPEN when the individual dataset is World Bank-produced CC BY 4.0.

### UNdata economic/statistical series
Status: APPROVED_OPEN subject to cited source/database terms.

### FRED
Status: EXCLUDE_FOR_TRAINING via FRED services/API under current terms.
Reason: current FRED terms expressly prohibit using FRED Services/Content in development or training of machine-learning/AI systems. Do not ingest FRED data into Titan training datasets even where an underlying series may be public-domain. If a needed series is available directly from its original public-domain agency, source it from that agency under that agency's terms instead.

### Public blockchain data
Status: CONDITIONAL by chain, indexer and API provider.
Facts recorded on a public blockchain are not equivalent to blanket permission to scrape a commercial indexer's API. Prefer protocol-native/open datasets or independently operated nodes; record chain height/date and ingestion method.

### Protocol documentation / open-source clients
Status: CONDITIONAL by repository/document license.
Use permissively licensed protocol specifications and open-source code for architecture/technical concepts; do not redistribute copyrighted prose unnecessarily.

## History

Priority open source families:
- Smithsonian CC0 metadata/assets
- Europeana CC0 metadata
- Wikidata CC0 structured data
- U.S. National Archives/public-domain U.S. government records where applicable
- Library/museum collections only where item rights are explicitly public domain/CC0/open
- government statistical archives under compatible terms

Historical training records must distinguish established facts from interpretation and record sources for contested claims.

## Ingestion rules

1. A source does not enter the production corpus until its license/status is in this registry or a versioned successor.
2. Item-level licensing overrides source-family assumptions.
3. `CONDITIONAL` sources require an automated or human license check before ingestion.
4. `EXCLUDE_FOR_TRAINING` sources must be blocked by the ingestion pipeline.
5. Source text is evidence, not the final answer. Generate original prompt/answer material and retain provenance.
6. Use multiple independent source families for important factual claims where practical.
7. For volatile facts, record `as_of`; prefer retrieval-backed use over permanent weight training when staleness risk is material.
8. Never fabricate a citation or provenance identifier.
9. Dataset package must include source-family counts and license summary.
10. A factual competency cannot pass release review if its source distribution is materially dependent on one provider when independent corroboration is reasonably available.
