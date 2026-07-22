from collections import Counter, defaultdict
import re
import json
from langchain_groq import ChatGroq
from backend.config import GROQ_API_KEY, GROQ_MODEL

FAILURE_KEYWORDS = {
    "failure": ["failure", "failed", "fault", "breakdown", "rca", "root cause"],
    "leak": ["leak", "leakage", "seal leak"],
    "vibration": ["vibration", "vibrating", "high vibration"],
    "overheating": ["overheat", "overheating", "high temperature", "hot bearing"],
    "corrosion": ["corrosion", "rust", "corroded"],
    "trip": ["trip", "tripped", "shutdown"],
    "wear": ["wear", "worn", "erosion"],
    "blockage": ["blockage", "blocked", "clog", "choking"],
    "misalignment": ["misalignment", "misaligned", "alignment"],
}


def _excerpt(text: str, size: int = 450) -> str:
    text = " ".join((text or "").split())
    return text[:size] + ("..." if len(text) > size else "")


def _entity_values(chunk: dict, *keys: str) -> list[str]:
    entities = chunk.get("entities", {}) or {}
    values = []
    for key in keys:
        item = entities.get(key, [])
        if isinstance(item, list):
            values.extend(str(value) for value in item if value)
    return values


def _detect_failure_modes(text: str) -> list[str]:
    lowered = (text or "").lower()
    found = []
    for mode, keywords in FAILURE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            found.append(mode.title())
    return found

def _detect_asset_type(text: str) -> str:
    text = (text or "").lower()

    if any(word in text for word in [
        "software", "application", "system manual",
        "host computer", "interface control"
    ]):
        return "Software"

    if any(word in text for word in [
        "pump", "motor", "compressor", "valve",
        "bearing", "gearbox"
    ]):
        return "Mechanical"

    if any(word in text for word in [
        "facility", "plant", "station", "building"
    ]):
        return "Facility"

    return "General"

class MaintenanceAgent:
    def analyze(self, equipment_tag: str, chunks: list[dict]) -> dict:
        tag = (equipment_tag or "").upper().strip()
        tag_pattern = re.compile(rf"\b{re.escape(tag)}\b", re.IGNORECASE) if tag else None

        relevant = []

        for chunk in chunks:
            text = chunk.get("text", "")
            equipment_tags = [value.upper() for value in _entity_values(chunk, "equipment_tags", "equipment", "assets")]
            if tag and ((tag_pattern and tag_pattern.search(text)) or tag in equipment_tags):
                relevant.append(chunk)

        combined_text = "\n".join(
            chunk.get("text", "")
            for chunk in relevant
        )
        asset_type = _detect_asset_type(combined_text)

        modes = Counter()
        evidence = []
        for chunk in relevant:
            extracted_modes = _entity_values(chunk, "failure_modes")
            detected_modes = _detect_failure_modes(chunk.get("text", ""))
            # modes.update(str(mode).title() for mode in (extracted_modes or detected_modes))
            all_modes = extracted_modes or detected_modes
            if all_modes:
                modes.update(str(mode).title() for mode in all_modes)
            evidence.append({
                "file_name": chunk.get("file_name", "Document"),
                "chunk_id": chunk.get("id", ""),
                "page": chunk.get("metadata", {}).get("page", "Unknown"),
                "score": 1.0,
                "excerpt": _excerpt(chunk.get("text", "")),
            })
            if extracted_modes:
                modes.update(str(mode).title() for mode in extracted_modes)

        failure_modes = [
            {
                "name": mode,
                "description": f"Found in {count} supporting record(s) for {tag}.",
                "occurrences": count,
            }
            for mode, count in modes.most_common()
        ]

        # recommendations = []
        # if failure_modes:
        #     recommendations.append(f"Prioritise RCA around repeated '{failure_modes[0]['name']}' signals for {tag}.")
        # if relevant:
            
        #        recommendations.extend([ "Review the cited work orders, inspection notes, and maintenance history before releasing the asset.",
        #         "Compare recurring symptoms against OEM preventive maintenance intervals and operating conditions.",
        #         "Schedule condition-based inspection for repeated vibration, overheating, leakage, trips, corrosion, or blockage signals.",
        #     ])
        # else:
        #     recommendations.append(f"No direct records found for {tag}. Upload maintenance history or verify the equipment tag spelling.")

        # if len(relevant) >= 5 or sum(modes.values()) >= 3:
        #     risk = "High"

        recommendations = []

        if asset_type == "Software":
            recommendations = [
                "Follow the documented software maintenance procedures.",
                "Apply configuration management before deploying updates.",
                "Perform software testing and validation after changes.",
                "Maintain version control and change records."
            ]

        elif asset_type == "Mechanical":
            recommendations = [
                "Follow the maintenance procedures described in the document."
            ]

        elif asset_type == "Electrical":
            recommendations = [
                "Follow the inspection and maintenance guidance described in the document."
            ]

        elif not relevant:
            recommendations = [
                f"No direct records found for {tag}. Upload maintenance history or verify the equipment tag spelling."
            ]

        # risk = "Unknown"

        # if failure_modes:
        #     risk = "Medium"

        risk = "Low"

        if len(failure_modes) >= 3:
            risk = "High"
        elif len(failure_modes) >= 1:
            risk = "Medium"

        if not relevant:
            risk = "Unknown"

        return {
            "equipment_tag": tag,
            "asset_type": asset_type,
            "risk_level": risk,
            "failure_modes": failure_modes,
            "recommendations": recommendations,
            "evidence": evidence[:8],
        }


class ComplianceAgent:
    REQUIRED_CONTROLS = {
        "Inspection Records": ["inspection", "audit", "checklist", "test certificate", "report"],
        "Safety Procedure": ["sop", "procedure", "permit", "lockout", "tagout", "ppe", "safety"],
        "Maintenance Evidence": ["work order", "preventive maintenance", "calibration", "repair", "maintenance"],
        "Incident Learning": ["incident", "near miss", "root cause", "corrective action", "failure"],
        "Regulatory Reference": ["factory act", "oisd", "peso", "iso", "bis", "cpcb", "spcb", "standard"],
    }

    def assess(self, chunks: list[dict], standard: str) -> dict:
        corpus = "\n".join(chunk.get("text", "").lower() for chunk in chunks)
        coverage = []
        gaps = []

        for control, keywords in self.REQUIRED_CONTROLS.items():
            hits = [keyword for keyword in keywords if keyword in corpus]
            evidence = []
            if hits:
                for chunk in chunks:
                    text_lower = chunk.get("text", "").lower()
                    if any(keyword in text_lower for keyword in hits):
                        evidence.append({
                            "file_name": chunk.get("file_name", "Document"),
                            "chunk_id": chunk.get("id", ""),
                            "excerpt": _excerpt(chunk.get("text", ""), 260),
                        })
                        if len(evidence) == 3:
                            break

            if hits:
                coverage.append({
                    "title": control,
                    "description": f"Evidence found for: {', '.join(sorted(set(hits)))}.",
                    "status": "Covered",
                    "evidence": evidence,
                })
            else:
                gaps.append({
                    "title": control,
                    "description": f"No clear uploaded evidence found for {control.lower()} against {standard}.",
                    "recommendation": f"Upload or link current {control.lower()} records and map them to {standard}.",
                    "status": "Gap",
                })

        # score = round(100 * len(coverage) / len(self.REQUIRED_CONTROLS), 1)
        score = round(100 * len(coverage) / len(self.REQUIRED_CONTROLS))

        status = (
            "Compliant"
            if score >= 85 else
            "Partially Compliant"
            if score >= 50 else
            "Non-Compliant"
        )

        audit_readiness = (
            "Ready"
            if score >= 85 else
            "Needs Improvement"
            if score >= 50 else
            "Not Ready"
        )

        recommendations = []

        for gap in gaps:
            recommendations.append(gap["recommendation"])

        return {
            "standard": standard,
            "compliance_score": score,
            "status": status,
            "requirements_covered": [
                item["title"] for item in coverage
            ],
            "compliance_gaps": [
                item["description"] for item in gaps
            ],
            "audit_readiness": audit_readiness,
            "recommendations": recommendations,
            "supporting_evidence": [
                evidence
                for item in coverage
                for evidence in item["evidence"]
            ],
        }
        
# class LessonsAgent:
#     def summarize(self, chunks: list[dict]) -> dict:
#         by_mode = defaultdict(list)
#         for chunk in chunks:
#             modes = _entity_values(chunk, "failure_modes") or _detect_failure_modes(chunk.get("text", ""))
#             for mode in modes:
#                 by_mode[str(mode).title()].append(chunk)

#         patterns = []
#         for mode, evidence_chunks in sorted(by_mode.items(), key=lambda item: len(item[1]), reverse=True):
#             patterns.append({
#                 "pattern": mode,
#                 "occurrences": len(evidence_chunks),
#                 "warning": f"Repeated {mode} references found. Push this as a field alert where similar equipment or operating context appears.",
#                 "evidence": [
#                     {
#                         "file_name": chunk.get("file_name", "Document"),
#                         "chunk_id": chunk.get("id", ""),
#                         "excerpt": _excerpt(chunk.get("text", ""), 300),
#                     }
#                     for chunk in evidence_chunks[:4]
#                 ],
#             })

#         return {"patterns": patterns[:10], "total_patterns": len(patterns)}

class LessonsAgent:
    """
    LLM-powered Lessons Learned Agent.

    Generates meaningful operational lessons from the uploaded
    industrial documents instead of simply counting keywords.
    """

    def summarize(self, chunks: list[dict]) -> dict:

        if not chunks:
            return {
                "patterns": [],
                "total_patterns": 0
            }

        # ----------------------------------------------------
        # If Groq is unavailable, use simple fallback
        # ----------------------------------------------------

        if not GROQ_API_KEY:

            patterns = []

            for chunk in chunks[:5]:

                patterns.append({
                    "pattern": "General Observation",
                    "occurrences": 1,
                    "warning": "LLM unavailable. Showing document excerpt.",
                    "recommendation": "Review this document manually.",
                    "evidence": [
                        {
                            "file_name": chunk.get("file_name", "Document"),
                            "chunk_id": chunk.get("id", ""),
                            "excerpt": _excerpt(chunk.get("text", ""), 300),
                        }
                    ]
                })

            return {
                "patterns": patterns,
                "total_patterns": len(patterns)
            }

        # ----------------------------------------------------
        # Build Context
        # ----------------------------------------------------

        context = []

        evidence = []

        # Limit context size
        for chunk in chunks[:4]:

            context.append(f"""
SOURCE DOCUMENT : {chunk.get("file_name","Document")}

CONTENT
-------
{chunk.get("text","")}
""")

            evidence.append({
                "file_name": chunk.get("file_name", "Document"),
                "chunk_id": chunk.get("id", ""),
                "excerpt": _excerpt(chunk.get("text", ""), 300),
            })

        context = "\n\n".join(context)

        prompt = f"""
You are a Senior Industrial Reliability Engineer.

Analyze ONLY the supplied industrial documents.

Your job is to identify meaningful lessons learned.

Rules

1. Group similar observations together.

2. Ignore repeated keywords.

3. Extract operational lessons.

4. Extract maintenance lessons.

5. Extract safety lessons.

6. Extract quality lessons.

7. Extract recurring problems.

8. Extract best practices.

9. Never invent information.

10. If no lesson exists return an empty list.

Return ONLY valid JSON.

JSON format

{{
    "lessons":[
        {{
            "title":"Short title",
            "lesson":"What was learned",
            "recommendation":"Recommended action",
            "confidence":"High"
        }}
    ]
}}

Documents

{context}
"""

        try:

            llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0,
                max_tokens=900,
            )

            response = llm.invoke(prompt)

            answer = response.content.strip()

            answer = answer.replace("```json", "")
            answer = answer.replace("```", "")
            answer = answer.strip()

            match = re.search(r"\{.*\}", answer, re.DOTALL)

            if not match:
                raise ValueError("Invalid JSON returned.")

            data = json.loads(match.group(0))

            lessons = data.get("lessons", [])

            patterns = []

            for lesson in lessons:
                patterns.append({

                "pattern": lesson.get(
                    "title",
                    "Lesson"
                ),

                "occurrences": 1,

                "warning": lesson.get(
                    "lesson",
                    ""
                ),

                "recommendation": lesson.get(
                    "recommendation",
                    ""
                ),

                "confidence": lesson.get(
                    "confidence",
                    "Medium"
                ),

                "evidence": list({
                    item["file_name"]: item
                    for item in evidence
                }.values())[:3],

            })

            return {

                "patterns": patterns,

                "total_patterns": len(patterns)

            }

        except Exception as e:

                print("=" * 80)
                print("LESSONS LLM ERROR")
                print(e)
                print("=" * 80)

                return {
                    "patterns": [],
                    "total_patterns": 0
                }