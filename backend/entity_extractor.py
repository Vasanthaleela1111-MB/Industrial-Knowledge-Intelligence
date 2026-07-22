import json
import re
from collections import defaultdict

from backend.config import GROQ_API_KEY, GROQ_MODEL


# ============================================================
# REGEX PATTERNS
# ============================================================

TAG_PATTERN = re.compile(
    r"\b(?:"
    r"[A-Z]{1,5}[-/]?\d{2,5}[A-Z]?"
    r"|"
    r"[A-Z]{2,}-[A-Z]{1,4}-\d{2,5}"
    r")\b"
)

DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
    r"|"
    r"\d{4}-\d{2}-\d{2}"
    r")\b"
)

PARAMETER_PATTERN = re.compile(
    r"\b(?:"
    r"pressure|temperature|flow|level|vibration|"
    r"current|voltage|speed|torque|ph|humidity"
    r")\s*[:=]?\s*"
    r"[-+]?\d+(?:\.\d+)?\s*"
    r"(?:"
    r"bar|psi|kpa|c|degc|rpm|mm/s|a|v|%|m3/h|lpm"
    r")?",
    re.IGNORECASE,
)

REGULATORY_PATTERN = re.compile(
    r"\b(?:"
    r"Factory Act|"
    r"OISD[- ]?\d*|"
    r"PESO|"
    r"ISO\s?\d{3,5}|"
    r"BIS|"
    r"OSHA|"
    r"CPCB|"
    r"SPCB|"
    r"EPA|"
    r"NFPA\s?\d*"
    r")\b",
    re.IGNORECASE,
)

FAILURE_PATTERN = re.compile(
    r"\b(?:"
    r"leak|"
    r"corrosion|"
    r"crack|"
    r"overheat|"
    r"trip|"
    r"failure|"
    r"vibration|"
    r"seizure|"
    r"cavitation|"
    r"blockage|"
    r"contamination|"
    # r"deviation|"
    r"non[- ]?conformance"
    r")\b",
    re.IGNORECASE,
)

INDUSTRIAL_STOPWORDS = {

    "date",
    "document",
    "report",
    "page",
    "table",
    "figure",

    "maintenance",
    "inspection",
    "failure",
    "system",
    "equipment",
    "component",
    "procedure",
    "manual",
    "software",
    "hardware",
    "document number",
    "reference",
    "record",

    "process",
    "device",
    "machine",
    "assembly",
    "unit",
    "operation",
    "test",
    "analysis",
    "service"
}

KNOWLEDGE_EXTRACTION_PROMPT = """
You are an industrial knowledge extraction system.

Analyze the industrial document text provided below.

Your job is to extract important industrial entities and the
meaningful relationships explicitly supported by the text.

The uploaded document can be any type of industrial document,
including:

- Engineering documents
- Maintenance reports
- Inspection reports
- SOPs
- OEM manuals
- P&IDs
- Safety documents
- Quality documents
- Compliance documents
- Audit reports
- Incident reports
- Work orders
- Process documents
- Technical manuals

ENTITY EXTRACTION

Identify meaningful entities.

Possible entity types include, but are not limited to:

Equipment
Subsystem
Component
SparePart
Instrument
Valve
Pump
Motor
Compressor
Tank
Pipeline

FailureMode
FailurePattern
FailureEvent
RootCause
CorrectiveAction
PreventiveAction

MaintenanceActivity
MaintenanceTask
MaintenanceSchedule
Inspection
InspectionFinding
WorkOrder

ProcessParameter
Sensor
Measurement
Alarm
OperatingCondition

Hazard
Risk
Incident
NearMiss

Regulation
Standard
ComplianceRequirement
QualityRequirement

SOP
Procedure
Checklist

Material
Chemical
Utility

Location
Plant
Unit
Area

Organization
OEM
Vendor
Person

If an important industrial entity does not fit these types,
you may create an appropriate entity type.

Do not extract meaningless generic words as entities.

For example:

Bad:
failure
maintenance
inspection
pump
motor
system
component
equipment
software
manual
report
procedure

Never classify these words alone as a FailureMode:

Deviation
Configuration
Procedure
Manual
Software
Maintenance
Inspection
Documentation
Requirement
Update
Version
Change Control

These are only FailureMode entities if the document explicitly states they are an actual failure, fault, defect, incident, alarm, or abnormal condition.

Good:
Pump P-101
Main Hydraulic Pump
GCID Processor
Ground Software Maintenance Facility
Simulation Control Station
Bearing Failure
Mechanical Seal Leakage
Pressure Relief Valve
Compressor C-203
Lubrication Pump
Operating System Upgrade
Preventive Maintenance Schedule

RELATIONSHIP EXTRACTION

Extract relationships only when they are supported by the text.

Do not create generic relationships.

BAD

Equipment -> Maintenance

Pump -> Failure

GOOD

Pump P-101 -> HAS_FAILURE_MODE -> Bearing Failure

Bearing Failure -> CAUSED_BY -> Poor Lubrication

Poor Lubrication -> HAS_CORRECTIVE_ACTION -> Bearing Replacement

Possible relationship types include, but are not limited to:

HAS_COMPONENT
PART_OF

CONNECTED_TO

HAS_FAILURE_MODE
FAILED_DUE_TO
CAUSED_BY

HAS_ROOT_CAUSE

RESOLVED_BY

HAS_CORRECTIVE_ACTION
HAS_PREVENTIVE_ACTION

INSPECTED_BY

HAS_FINDING

REQUIRES_MAINTENANCE
UNDERWENT_MAINTENANCE

HAS_PARAMETER

OPERATES_AT

LOCATED_AT

MONITORED_BY

REFERENCES

GOVERNED_BY

COMPLIES_WITH

HAS_HAZARD

USES

MANUFACTURED_BY

REPLACED_WITH

SIMILAR_TO

You may create another meaningful relationship type when
required by the document.

IMPORTANT RULES

IMPORTANT RULES

1. Extract ONLY entities explicitly mentioned.

2. Never invent entities.

3. Never invent relationships.

4. Never connect two entities unless the document explicitly states the relationship.

5. Do NOT extract generic nouns such as:

failure
maintenance
inspection
system
equipment
component
software
document
manual
procedure
report
plant

unless they are part of a larger entity.

GOOD

Bearing Failure

Mechanical Seal Leakage

Simulation Control Station

Ground Software Maintenance Facility

Pump P-101

Hydraulic Motor HM-201

BAD

Failure

Maintenance

Inspection

Pump

Motor

System

6. Preserve equipment tags exactly.

7. Prefer complete entity names.

8. Merge abbreviations with their full names whenever the document clearly defines them.

9. Return ONLY valid JSON.

Do not include explanations.

Do not include markdown.

Do not include ```json code fences.

Required JSON format:

{
    "entities": [
        {
            "name": "P-101",
            "type": "Equipment"
        },
        {
            "name": "mechanical seal leakage",
            "type": "FailureMode"
        }
    ],
    "relationships": [
        {
            "source": "P-101",
            "target": "mechanical seal leakage",
            "type": "HAS_FAILURE_MODE"
        }
    ]
}

If no meaningful entities or relationships exist, return:

{
    "entities": [],
    "relationships": []
}

Example

Input:

Pump P101 developed excessive vibration.

Inspection found bearing wear.

Root cause was poor lubrication.

Bearing was replaced.

Maintenance followed ISO 9001.

Output:

{
  "entities":[
    {
      "name":"Pump P101",
      "type":"Equipment"
    },
    {
      "name":"Bearing",
      "type":"Component"
    },
    {
      "name":"Excessive Vibration",
      "type":"FailureMode"
    },
    {
      "name":"Poor Lubrication",
      "type":"RootCause"
    },
    {
      "name":"Bearing Replacement",
      "type":"CorrectiveAction"
    },
    {
      "name":"ISO 9001",
      "type":"Standard"
    }
  ],

  "relationships":[
    {
      "source":"Pump P101",
      "target":"Bearing",
      "type":"HAS_COMPONENT"
    },
    {
      "source":"Pump P101",
      "target":"Excessive Vibration",
      "type":"HAS_FAILURE_MODE"
    },
    {
      "source":"Excessive Vibration",
      "target":"Poor Lubrication",
      "type":"CAUSED_BY"
    },
    {
      "source":"Poor Lubrication",
      "target":"Bearing Replacement",
      "type":"HAS_CORRECTIVE_ACTION"
    },
    {
      "source":"Bearing Replacement",
      "target":"ISO 9001",
      "type":"COMPLIES_WITH"
    }
  ]
}

Now analyze the following industrial document.

Industrial Document:

INDUSTRIAL DOCUMENT TEXT:
"""


class EntityExtractor:

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        if not GROQ_API_KEY:
            return None

        from langchain_groq import ChatGroq

        self._llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
            max_tokens=1200,
        )
        return self._llm

    # ========================================================
    # FAST REGEX ENTITY EXTRACTION
    # ========================================================

    def extract(
        self,
        text: str
    ) -> dict[str, list[str]]:

        entities = defaultdict(set)

        # Equipment tags
        for match in TAG_PATTERN.findall(text):

            entities[
                "equipment_tags"
            ].add(
                match.upper()
            )

        # Dates
        for match in DATE_PATTERN.findall(text):

            entities[
                "dates"
            ].add(
                match
            )

        # Process parameters
        for match in PARAMETER_PATTERN.findall(text):

            entities[
                "process_parameters"
            ].add(
                match.strip()
            )

        # Regulatory references
        for match in REGULATORY_PATTERN.findall(text):

            entities[
                "regulatory_references"
            ].add(
                match.upper()
            )

        # Basic failure keywords
        for match in FAILURE_PATTERN.findall(text):

            entities[
                "failure_modes"
            ].add(
                match.lower()
            )

        return {
            key: sorted(values)
            for key, values
            in entities.items()
        }
    
    def filter_entities(
        self,
        entities: list[dict]
    ) -> list[dict]:
        """
        Remove generic, duplicate and low-value entities.
        """

        filtered = []

        seen = set()

        for entity in entities:

            if not isinstance(entity, dict):
                continue

            name = str(
                entity.get("name", "")
            ).strip()

            name = re.sub(r"\s+", " ", name)
            name = name.strip(".,:;-")

            entity_type = str(
                entity.get("type", "")
            ).strip()

            if not name:
                continue

            # Remove one-letter entities
            if len(name) <= 1:
                continue

            # Remove pure numbers
            if name.isdigit():
                continue

            # Remove dates
            if DATE_PATTERN.fullmatch(name):
                continue

            # Remove generic words
            if name.lower() in INDUSTRIAL_STOPWORDS:
                continue

            key = (
                entity_type.lower(),
                name.lower()
            )

            if key in seen:
                continue

            seen.add(key)

            filtered.append(
                {
                    "name": name,
                    "type": entity_type
                }
            )

        return filtered

    # ========================================================
    # CONTEXTUAL ENTITY + RELATIONSHIP EXTRACTION
    # ========================================================

    def extract_knowledge(
        self,
        text: str
    ) -> dict:

        llm = self._get_llm()
        if llm is None:

            print(
                "[EntityExtractor] "
                "GROQ_API_KEY not configured. "
                "Skipping contextual extraction."
            )

            return {
                "entities": [],
                "relationships": []
            }

        # Avoid sending empty chunks
        if not text or not text.strip():
            # entities = self.filter_entities(
            #     entities
            # )

            # print("\n" + "=" * 80)
            # print("FINAL ENTITIES")
            # print(json.dumps(entities, indent=2))

            # print("\nFINAL RELATIONSHIPS")
            # print(json.dumps(relationships, indent=2))

            # print("=" * 80 + "\n")


            return {
                "entities": [],
                "relationships": []
            }

        try:
            from langchain_core.messages import HumanMessage

            prompt = KNOWLEDGE_EXTRACTION_PROMPT + text

            response = llm.invoke(
                [HumanMessage(content=prompt)]
            )

            print("\n===== RAW LLM OUTPUT =====")
            print(response.content)
            print("==========================\n")

            if not response:

            #     entities = self.filter_entities(
            #     entities
            # )

                return {
                    "entities": [],
                    "relationships": []
                }

            if not hasattr(
                response,
                "content"
            ):
                
            #     entities = self.filter_entities(
            #     entities
            # )

                return {
                    "entities": [],
                    "relationships": []
                }

            content = (
                response
                .content
                .strip()
            )

            if not content:

            #     entities = self.filter_entities(
            #     entities
            # )

                return {
                    "entities": [],
                    "relationships": []
                }

            # ------------------------------------------------
            # Remove markdown fences if Groq adds them
            # ------------------------------------------------

            content = re.sub(
                r"^```(?:json)?\s*",
                "",
                content,
                flags=re.IGNORECASE,
            )

            content = re.sub(
                r"\s*```$",
                "",
                content,
            )

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            data = json.loads(
                content
            )

            entities = data.get(
                "entities",
                []
            )
            seen = set()
            clean_entities = []

            for entity in entities:

                if not isinstance(entity, dict):
                    continue

                name = str(entity.get("name", "")).strip()
                name = re.sub(r"\s+", " ", name)
                name = name.strip(".,:;-")
                # Ignore entities with fewer than 3 letters unless they contain digits
                if len(name) < 3 and not re.search(r"\d", name):
                    continue
                entity_type = str(entity.get("type", "Entity")).strip()

                if not name:
                    continue

                key = (
                    entity_type.lower(),
                    name.lower()
                )

                if key in seen:
                    continue

                seen.add(key)

                clean_entities.append(
                    {
                        "name": name,
                        "type": entity_type
                    }
                )

            entities = clean_entities

            relationships = data.get(
                "relationships",
                []
            )

            clean_relationships = []

            for rel in relationships:

                if (
                    isinstance(rel, dict)
                    and rel.get("source")
                    and rel.get("target")
                    and rel.get("type")
                ):
                    clean_relationships.append(
                        {
                            "source": str(rel["source"]).strip(),
                            "target": str(rel["target"]).strip(),
                            "type": str(rel["type"]).upper().replace(" ", "_")
                        }
                    )

            relationships = clean_relationships

            # Basic type validation
            if not isinstance(
                entities,
                list
            ):

                entities = []

            if not isinstance(
                relationships,
                list
            ):

                relationships = []

            entities = self.filter_entities(
                entities
            )    

            return {
                "entities": entities,
                "relationships": relationships,
            }

        except json.JSONDecodeError as e:

            print(
                "[EntityExtractor] "
                f"Invalid JSON from Groq: {e}"
            )

            # entities = self.filter_entities(
            #     entities
            # )

            return {
                "entities": [],
                "relationships": []
            }

        except Exception as e:

            print(
                "[EntityExtractor] "
                f"Knowledge extraction failed: {e}"
            )

            return {
                "entities": [],
                "relationships": []
            }

    # ========================================================
    # MERGE REGEX ENTITY RESULTS
    # ========================================================

    def merge(
        self,
        entity_sets: list[
            dict[str, list[str]]
        ]
    ) -> dict[str, list[str]]:

        merged = defaultdict(set)

        for entity_set in entity_sets:

            for key, values in entity_set.items():

                merged[
                    key
                ].update(
                    values
                )

        return {
            key: sorted(values)
            for key, values
            in merged.items()
        }