"""
scripts/facts.py
==================
ELI5: This file is the "answer key" the whole project is built from. Every
single fact here becomes BOTH a sentence buried inside a fake company
document AND a question-and-answer pair in the evaluation set. Building
both from the same source means we can be 100% sure every "correct
answer" in the exam actually exists, word-for-word, somewhere in the
documents — otherwise we'd be grading against answers that were never
actually retrievable in the first place, which would make the whole
evaluation meaningless.

75 facts total, deliberately spread across 5 categories so the eval can
show which retrieval method is strong or weak on which KIND of question,
not just an overall score.
"""

from dataclasses import dataclass


@dataclass
class Fact:
    fact_id: str
    category: str          # product | policy | procedure | decision | glossary
    doc_id: str             # which document file this fact lives in
    doc_title: str
    fact_sentence: str      # the exact sentence embedded in the document
    question: str           # the natural-language question a user might ask
    expected_answer: str    # short answer
    answer_keywords: list[str]  # words that must appear in a retrieved chunk to count as a "hit"
    difficulty: str          # exact | paraphrase | distractor


COMPANY_NAME = "Solara Robotics Inc."

# ---------------------------------------------------------------------------
# 1. PRODUCT SPECS — 8 products x 3 facts = 24 questions
# ---------------------------------------------------------------------------
PRODUCTS = [
    {"name": "Solara Rover X1", "id": "rover-x1", "battery_kwh": 18.5, "range_km": 42,
     "price_usd": 24500, "warranty_years": 2, "release_year": 2024},
    {"name": "Solara Rover X2", "id": "rover-x2", "battery_kwh": 26.0, "range_km": 68,
     "price_usd": 31900, "warranty_years": 3, "release_year": 2025},
    {"name": "Solara Drone Falcon-3", "id": "falcon-3", "battery_kwh": 2.1, "range_km": 15,
     "price_usd": 3800, "warranty_years": 1, "release_year": 2023},
    {"name": "Solara Drone Falcon-5", "id": "falcon-5", "battery_kwh": 3.4, "range_km": 27,
     "price_usd": 6200, "warranty_years": 2, "release_year": 2025},
    {"name": "Solara SensorPod LX", "id": "sensorpod-lx", "battery_kwh": 0.8, "range_km": 0,
     "price_usd": 1450, "warranty_years": 1, "release_year": 2024},
    {"name": "Solara SensorPod LX-Pro", "id": "sensorpod-lx-pro", "battery_kwh": 1.2, "range_km": 0,
     "price_usd": 2300, "warranty_years": 2, "release_year": 2025},
    {"name": "Solara Cargo Hauler H1", "id": "hauler-h1", "battery_kwh": 45.0, "range_km": 85,
     "price_usd": 58000, "warranty_years": 3, "release_year": 2024},
    {"name": "Solara Cargo Hauler H2", "id": "hauler-h2", "battery_kwh": 62.0, "range_km": 120,
     "price_usd": 79500, "warranty_years": 3, "release_year": 2026},
]

# ---------------------------------------------------------------------------
# 2. HR POLICIES — 15 facts
# ---------------------------------------------------------------------------
POLICIES = [
    {"id": "vacation-policy", "title": "Annual Leave Policy",
     "sentence": "All full-time employees at Solara Robotics Inc. receive 22 days of paid annual leave per calendar year.",
     "question": "How many days of paid annual leave do full-time employees get?",
     "answer": "22 days", "keywords": ["22", "days", "annual leave"], "difficulty": "exact"},
    {"id": "remote-work-policy", "title": "Remote Work Policy",
     "sentence": "Employees may work remotely up to 3 days per week with manager approval.",
     "question": "How many days per week can employees work remotely?",
     "answer": "3 days", "keywords": ["3", "days", "remote"], "difficulty": "exact"},
    {"id": "parental-leave-policy", "title": "Parental Leave Policy",
     "sentence": "Primary caregivers are entitled to 16 weeks of fully paid parental leave.",
     "question": "How many weeks of fully paid parental leave do primary caregivers receive?",
     "answer": "16 weeks", "keywords": ["16", "weeks", "parental"], "difficulty": "exact"},
    {"id": "probation-policy", "title": "Probation Period Policy",
     "sentence": "New hires complete a probation period of 90 days before their role is confirmed permanent.",
     "question": "How long is the probation period for new hires?",
     "answer": "90 days", "keywords": ["90", "days", "probation"], "difficulty": "exact"},
    {"id": "expense-policy", "title": "Travel Expense Policy",
     "sentence": "Employees can claim up to $75 per day for meals while traveling on company business.",
     "question": "What is the daily meal expense limit for business travel?",
     "answer": "$75 per day", "keywords": ["75", "meal", "day"], "difficulty": "exact"},
    {"id": "equipment-policy", "title": "Equipment Allowance Policy",
     "sentence": "Each employee receives an annual equipment allowance of $1,200 for home office setup.",
     "question": "What is the annual home office equipment allowance?",
     "answer": "$1,200", "keywords": ["1,200", "equipment", "allowance"], "difficulty": "exact"},
    {"id": "sick-leave-policy", "title": "Sick Leave Policy",
     "sentence": "Employees accrue 10 paid sick days per year, which do not roll over to the next year.",
     "question": "How many paid sick days do employees accrue per year?",
     "answer": "10 days", "keywords": ["10", "sick", "days"], "difficulty": "exact"},
    {"id": "notice-policy", "title": "Resignation Notice Policy",
     "sentence": "Employees are required to give 4 weeks of written notice before resigning.",
     "question": "How much notice must employees give before resigning?",
     "answer": "4 weeks", "keywords": ["4", "weeks", "notice"], "difficulty": "exact"},
    {"id": "overtime-policy", "title": "Overtime Compensation Policy",
     "sentence": "Overtime hours are compensated at 1.5 times the employee's standard hourly rate.",
     "question": "What rate is overtime paid at, relative to standard hourly pay?",
     "answer": "1.5 times", "keywords": ["1.5", "overtime", "rate"], "difficulty": "paraphrase"},
    {"id": "referral-policy", "title": "Employee Referral Policy",
     "sentence": "A successful employee referral that results in a hire earns a $2,000 bonus, paid after the new hire's 90-day mark.",
     "question": "How much is the employee referral bonus?",
     "answer": "$2,000", "keywords": ["2,000", "referral", "bonus"], "difficulty": "exact"},
    {"id": "training-policy", "title": "Professional Development Policy",
     "sentence": "Each employee has an annual learning and development budget of $1,500 for courses, books, or conferences.",
     "question": "What is the annual professional development budget per employee?",
     "answer": "$1,500", "keywords": ["1,500", "development", "budget"], "difficulty": "exact"},
    {"id": "conduct-policy", "title": "Code of Conduct",
     "sentence": "Any conflict of interest must be disclosed to HR in writing within 5 business days of becoming aware of it.",
     "question": "Within how many business days must a conflict of interest be disclosed?",
     "answer": "5 business days", "keywords": ["5", "business days", "conflict"], "difficulty": "paraphrase"},
    {"id": "data-policy", "title": "Data Retention Policy",
     "sentence": "Customer support records are retained for 3 years before being permanently deleted.",
     "question": "How long are customer support records retained before deletion?",
     "answer": "3 years", "keywords": ["3", "years", "retained"], "difficulty": "exact"},
    {"id": "security-policy", "title": "Password Security Policy",
     "sentence": "All employee passwords must be changed at least every 180 days.",
     "question": "How often must employee passwords be changed?",
     "answer": "every 180 days", "keywords": ["180", "days", "password"], "difficulty": "exact"},
    {"id": "wfh-equipment-policy", "title": "Home Office Safety Policy",
     "sentence": "Employees working from home must complete a home office safety checklist within 30 days of their start date.",
     "question": "Within how many days of starting must a new employee complete the home office safety checklist?",
     "answer": "30 days", "keywords": ["30", "days", "safety checklist"], "difficulty": "paraphrase"},
]

# ---------------------------------------------------------------------------
# 3. ENGINEERING PROCEDURES — 10 facts
# ---------------------------------------------------------------------------
PROCEDURES = [
    {"id": "code-review-proc", "title": "Code Review Procedure",
     "sentence": "Every pull request requires approval from at least 2 reviewers before it can be merged.",
     "question": "How many reviewer approvals does a pull request need before merging?",
     "answer": "2 reviewers", "keywords": ["2", "reviewers", "approval"], "difficulty": "exact"},
    {"id": "test-coverage-proc", "title": "Test Coverage Standard",
     "sentence": "All new backend code must maintain a minimum of 85% automated test coverage.",
     "question": "What is the minimum required test coverage for new backend code?",
     "answer": "85%", "keywords": ["85", "coverage", "test"], "difficulty": "exact"},
    {"id": "incident-response-proc", "title": "Incident Response SLA",
     "sentence": "Critical production incidents must receive an initial response within 15 minutes of being reported.",
     "question": "Within how many minutes must a critical incident receive an initial response?",
     "answer": "15 minutes", "keywords": ["15", "minutes", "incident"], "difficulty": "exact"},
    {"id": "deployment-proc", "title": "Production Deployment Procedure",
     "sentence": "Deployments to production require sign-off from the on-call engineering lead and are only permitted between 9am and 4pm on weekdays.",
     "question": "What hours are production deployments permitted?",
     "answer": "9am to 4pm on weekdays", "keywords": ["9am", "4pm", "weekdays"], "difficulty": "paraphrase"},
    {"id": "firmware-update-proc", "title": "Firmware Update Rollout Procedure",
     "sentence": "New firmware is rolled out to 5% of the fleet first, and only expanded fleet-wide after 72 hours with no reported faults.",
     "question": "What percentage of the fleet receives new firmware in the initial rollout stage?",
     "answer": "5%", "keywords": ["5%", "fleet", "firmware"], "difficulty": "exact"},
    {"id": "battery-safety-proc", "title": "Battery Safety Testing Procedure",
     "sentence": "Every battery pack must pass a thermal stress test at 60 degrees Celsius before shipping.",
     "question": "What temperature is used for the battery pack thermal stress test?",
     "answer": "60 degrees Celsius", "keywords": ["60", "degrees", "thermal"], "difficulty": "exact"},
    {"id": "sensor-calibration-proc", "title": "Sensor Calibration Procedure",
     "sentence": "LIDAR sensors must be recalibrated every 500 operating hours or every 6 months, whichever comes first.",
     "question": "How often must LIDAR sensors be recalibrated, in terms of operating hours?",
     "answer": "every 500 operating hours", "keywords": ["500", "hours", "lidar", "recalibrat"], "difficulty": "paraphrase"},
    {"id": "data-pipeline-proc", "title": "Telemetry Data Pipeline Procedure",
     "sentence": "Raw telemetry data from all field units is uploaded to central storage every 4 hours when connectivity is available.",
     "question": "How often is raw telemetry data uploaded to central storage?",
     "answer": "every 4 hours", "keywords": ["4", "hours", "telemetry"], "difficulty": "exact"},
    {"id": "bug-triage-proc", "title": "Bug Triage Procedure",
     "sentence": "Bugs labeled as severity-1 must be triaged by the engineering team within 2 hours of being filed.",
     "question": "Within how many hours must a severity-1 bug be triaged?",
     "answer": "2 hours", "keywords": ["2", "hours", "severity-1", "triag"], "difficulty": "paraphrase"},
    {"id": "release-cadence-proc", "title": "Software Release Cadence",
     "sentence": "The core navigation software follows a 2-week release cadence, with hotfixes deployed as needed outside that cycle.",
     "question": "What is the release cadence for the core navigation software?",
     "answer": "2 weeks", "keywords": ["2", "week", "release", "cadence"], "difficulty": "exact"},
]

# ---------------------------------------------------------------------------
# 4. MEETING DECISIONS — 8 facts
# ---------------------------------------------------------------------------
DECISIONS = [
    {"id": "decision-q3-roadmap", "title": "Q3 Roadmap Planning Meeting Notes",
     "sentence": "The team decided to prioritize the Rover X2 firmware upgrade over the new SensorPod integration for Q3.",
     "question": "What did the team decide to prioritize for Q3, the Rover X2 firmware upgrade or the SensorPod integration?",
     "answer": "Rover X2 firmware upgrade", "keywords": ["Rover X2", "firmware", "prioritize"], "difficulty": "paraphrase"},
    {"id": "decision-vendor-switch", "title": "Supply Chain Review Meeting Notes",
     "sentence": "The group approved switching the LIDAR sensor supplier to ClearSight Optics starting next quarter.",
     "question": "Which new supplier was approved for LIDAR sensors?",
     "answer": "ClearSight Optics", "keywords": ["ClearSight", "Optics", "LIDAR", "supplier"], "difficulty": "exact"},
    {"id": "decision-office-move", "title": "Facilities Planning Meeting Notes",
     "sentence": "The company confirmed the Austin engineering office will relocate to a larger facility in March 2027.",
     "question": "When will the Austin engineering office relocate?",
     "answer": "March 2027", "keywords": ["March", "2027", "Austin", "relocate"], "difficulty": "exact"},
    {"id": "decision-pricing-change", "title": "Pricing Strategy Meeting Notes",
     "sentence": "Leadership approved a 6% price increase on the Cargo Hauler line effective the next fiscal year.",
     "question": "What price increase was approved for the Cargo Hauler line?",
     "answer": "6%", "keywords": ["6%", "price increase", "Cargo Hauler"], "difficulty": "exact"},
    {"id": "decision-hiring-freeze", "title": "Headcount Planning Meeting Notes",
     "sentence": "A hiring freeze was approved for all departments except engineering, effective immediately.",
     "question": "Which department was excluded from the hiring freeze?",
     "answer": "engineering", "keywords": ["engineering", "hiring freeze", "exclud"], "difficulty": "paraphrase"},
    {"id": "decision-safety-recall", "title": "Product Safety Review Meeting Notes",
     "sentence": "The committee voted to issue a voluntary recall of Falcon-3 units manufactured before March 2023 due to a battery connector defect.",
     "question": "Which product had a voluntary recall due to a battery connector defect?",
     "answer": "Falcon-3", "keywords": ["Falcon-3", "recall", "battery connector"], "difficulty": "paraphrase"},
    {"id": "decision-partnership", "title": "Strategic Partnership Meeting Notes",
     "sentence": "The board approved a strategic partnership with Meridian Logistics to pilot the Cargo Hauler H2 in 3 distribution centers.",
     "question": "How many distribution centers will pilot the Cargo Hauler H2 under the Meridian Logistics partnership?",
     "answer": "3", "keywords": ["3", "distribution centers", "Meridian"], "difficulty": "exact"},
    {"id": "decision-open-source", "title": "Engineering Strategy Meeting Notes",
     "sentence": "The team agreed to open-source the SensorPod calibration toolkit under an MIT license by the end of the year.",
     "question": "Under what license will the SensorPod calibration toolkit be open-sourced?",
     "answer": "MIT license", "keywords": ["MIT", "license", "open-source"], "difficulty": "exact"},
]

# ---------------------------------------------------------------------------
# 5. GLOSSARY TERMS — 18 facts (deliberately paraphrase-heavy: this category
#    is designed to challenge keyword-only search, since the question often
#    doesn't share exact words with the definition)
# ---------------------------------------------------------------------------
GLOSSARY = [
    {"id": "glossary-telemetry", "term": "Telemetry",
     "sentence": "Telemetry refers to the automated collection and transmission of sensor and performance data from a field unit back to central systems.",
     "question": "What does the term 'telemetry' mean at Solara Robotics?",
     "answer": "automated collection and transmission of sensor and performance data",
     "keywords": ["automated", "collection", "transmission", "sensor", "data"], "difficulty": "paraphrase"},
    {"id": "glossary-fleet", "term": "Fleet",
     "sentence": "A fleet refers to the complete set of deployed units of a single product model currently active in the field.",
     "question": "How is the term 'fleet' defined internally?",
     "answer": "the complete set of deployed units of a single product model",
     "keywords": ["deployed", "units", "product model"], "difficulty": "paraphrase"},
    {"id": "glossary-payload", "term": "Payload",
     "sentence": "Payload is the maximum additional weight a unit can carry or deploy beyond its own base weight.",
     "question": "What does 'payload' mean for a Solara unit?",
     "answer": "the maximum additional weight a unit can carry beyond its own base weight",
     "keywords": ["maximum", "additional weight", "carry"], "difficulty": "paraphrase"},
    {"id": "glossary-uptime", "term": "Uptime",
     "sentence": "Uptime is the percentage of scheduled operating hours during which a unit is fully functional and available for tasking.",
     "question": "How is 'uptime' defined for Solara units?",
     "answer": "the percentage of scheduled operating hours a unit is fully functional",
     "keywords": ["percentage", "operating hours", "functional"], "difficulty": "paraphrase"},
    {"id": "glossary-geofence", "term": "Geofence",
     "sentence": "A geofence is a virtual perimeter defined by GPS coordinates that restricts where an autonomous unit is permitted to operate.",
     "question": "What is a 'geofence' in the context of Solara's autonomous units?",
     "answer": "a virtual perimeter defined by GPS coordinates restricting where a unit can operate",
     "keywords": ["virtual perimeter", "GPS", "restrict"], "difficulty": "paraphrase"},
    {"id": "glossary-failsafe", "term": "Failsafe mode",
     "sentence": "Failsafe mode is an automatic operating state a unit enters when it loses connectivity, causing it to stop and await manual override.",
     "question": "What happens when a Solara unit enters 'failsafe mode'?",
     "answer": "it stops and awaits manual override", "keywords": ["stop", "manual override", "connectivity"], "difficulty": "paraphrase"},
    {"id": "glossary-otau", "term": "OTA update",
     "sentence": "An OTA (over-the-air) update is a firmware or software update delivered wirelessly to a deployed unit without requiring physical access.",
     "question": "What does 'OTA update' stand for and mean?",
     "answer": "over-the-air update, delivered wirelessly without physical access",
     "keywords": ["over-the-air", "wirelessly", "firmware"], "difficulty": "paraphrase"},
    {"id": "glossary-mtbf", "term": "MTBF",
     "sentence": "MTBF (Mean Time Between Failures) is the average operating time between one hardware failure and the next for a given unit type.",
     "question": "What does MTBF measure?",
     "answer": "the average operating time between hardware failures",
     "keywords": ["average", "time", "failures"], "difficulty": "exact"},
    {"id": "glossary-payload-bay", "term": "Payload bay",
     "sentence": "The payload bay is the physical compartment on a Cargo Hauler unit designed to hold cargo during transport.",
     "question": "What is the 'payload bay' on a Cargo Hauler unit?",
     "answer": "the compartment that holds cargo during transport", "keywords": ["compartment", "cargo", "transport"], "difficulty": "paraphrase"},
    {"id": "glossary-degraded-mode", "term": "Degraded mode",
     "sentence": "Degraded mode is a reduced-capability operating state a unit enters automatically when one or more non-critical sensors fail.",
     "question": "When does a unit enter 'degraded mode'?",
     "answer": "when one or more non-critical sensors fail", "keywords": ["non-critical", "sensors", "fail"], "difficulty": "paraphrase"},
    {"id": "glossary-hot-swap", "term": "Hot-swap",
     "sentence": "Hot-swap refers to replacing a battery pack or sensor module while the unit remains powered on.",
     "question": "What does 'hot-swap' mean for Solara hardware?",
     "answer": "replacing a component while the unit remains powered on",
     "keywords": ["replacing", "powered on", "battery"], "difficulty": "paraphrase"},
    {"id": "glossary-edge-compute", "term": "Edge compute",
     "sentence": "Edge compute refers to processing sensor data directly on the unit itself, rather than sending it to the cloud for processing.",
     "question": "What is 'edge compute' at Solara Robotics?",
     "answer": "processing sensor data directly on the unit instead of the cloud",
     "keywords": ["processing", "on the unit", "cloud"], "difficulty": "paraphrase"},
    {"id": "glossary-tele-op", "term": "Tele-operation",
     "sentence": "Tele-operation is the direct remote control of a unit by a human operator, used when autonomous mode is unsafe or unavailable.",
     "question": "What is 'tele-operation'?",
     "answer": "direct remote control of a unit by a human operator",
     "keywords": ["remote control", "human operator"], "difficulty": "paraphrase"},
    {"id": "glossary-mission-profile", "term": "Mission profile",
     "sentence": "A mission profile is a pre-configured set of tasks, waypoints, and operating parameters assigned to a unit before deployment.",
     "question": "What is a 'mission profile'?",
     "answer": "a pre-configured set of tasks and waypoints assigned before deployment",
     "keywords": ["pre-configured", "tasks", "waypoints"], "difficulty": "paraphrase"},
    {"id": "glossary-battery-cycle", "term": "Battery cycle",
     "sentence": "A battery cycle is counted each time a battery pack is discharged and recharged from roughly full to roughly empty.",
     "question": "What counts as one 'battery cycle'?",
     "answer": "one full discharge and recharge of the battery pack",
     "keywords": ["discharge", "recharge", "battery"], "difficulty": "paraphrase"},
    {"id": "glossary-swarm-mode", "term": "Swarm mode",
     "sentence": "Swarm mode allows multiple units to coordinate tasks and share sensor data with each other in real time.",
     "question": "What does 'swarm mode' enable?",
     "answer": "multiple units coordinating tasks and sharing sensor data in real time",
     "keywords": ["multiple units", "coordinate", "sensor data"], "difficulty": "paraphrase"},
    {"id": "glossary-calibration-drift", "term": "Calibration drift",
     "sentence": "Calibration drift is the gradual loss of sensor accuracy over time due to wear, temperature changes, or vibration.",
     "question": "What causes 'calibration drift'?",
     "answer": "wear, temperature changes, or vibration over time",
     "keywords": ["wear", "temperature", "vibration"], "difficulty": "paraphrase"},
    {"id": "glossary-checkpoint-sync", "term": "Checkpoint sync",
     "sentence": "Checkpoint sync is the process of periodically saving a unit's navigation state so it can resume safely after a restart.",
     "question": "What is 'checkpoint sync' used for?",
     "answer": "saving navigation state so a unit can resume safely after a restart",
     "keywords": ["saving", "navigation state", "restart"], "difficulty": "paraphrase"},
]
