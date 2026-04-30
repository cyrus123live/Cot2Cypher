"""ZOGRASCOPE-specific few-shot exemplars.

These exemplars use the Pole crime knowledge graph schema and Neo4j 5+ inline WHERE syntax
that ZOGRASCOPE uses (e.g., MATCH (x:Person WHERE x.name = "John") rather than separate WHERE clauses).

Coverage matrix:
| # | num_nodes | Query Type   | Pattern                      |
|---|-----------|--------------|------------------------------|
| 1 | 2         | count        | Single hop with two filters  |
| 2 | 2         | argmax       | Latest by date with filter   |
| 3 | 3         | entity_set   | Two-hop via shared friend    |
| 4 | 3         | attribute_set| Multi-hop email/phone lookup |
| 5 | 4         | min          | Earliest call duration       |
| 6 | 4         | entity_set   | Crime + officer + location   |
"""

EXEMPLARS_ZOG = [
    # 1. 2-node, count, two property filters (template 138)
    {
        "question": (
            "[Person] = Stephanie\n"
            "[Person] = Mary\n"
            "How many Stephanies share a home with someone named Mary?"
        ),
        "cypher": (
            'MATCH (x0:Person WHERE x0.name = "Stephanie")-[:KNOWS_LW]-(x1:Person WHERE x1.name = "Mary")\n'
            "RETURN COUNT(DISTINCT x0)"
        ),
        "reasoning": (
            "1. Sub-questions: (a) Find Person nodes named Stephanie. "
            "(b) Find Person nodes named Mary they share a home with via KNOWS_LW. "
            "(c) Count distinct Stephanies.\n"
            "2. Schema elements: Person nodes with name property; KNOWS_LW relationship "
            "(Person to Person, indicates living together).\n"
            "3. Pattern: Single-hop traversal between two Person nodes via KNOWS_LW with "
            "name filters on each end, undirected match (KNOWS_LW is symmetric).\n"
            "4. Construction:\n"
            "   - MATCH (x0:Person WHERE x0.name = \"Stephanie\")-[:KNOWS_LW]-(x1:Person WHERE x1.name = \"Mary\")\n"
            "   - RETURN COUNT(DISTINCT x0) for the count of Stephanies"
        ),
    },
    # 2. 2-node, argmax/max, single hop with filter
    {
        "question": (
            "[Vehicle] = Toyota\n"
            "What is the most recent date of a crime involving a Toyota?"
        ),
        "cypher": (
            'MATCH (x0:Crime)-[:INVOLVED_IN]-(x1:Vehicle WHERE x1.make = "Toyota")\n'
            "RETURN x0.date\n"
            "ORDER BY x0.date DESC\n"
            "LIMIT 1"
        ),
        "reasoning": (
            "1. Sub-questions: (a) Find Crimes that have a Vehicle involved. "
            "(b) Filter to Toyota vehicles. (c) Return the most recent crime date.\n"
            "2. Schema elements: Crime with date property; Vehicle with make property; "
            "INVOLVED_IN relationship from Vehicle to Crime.\n"
            "3. Pattern: Single-hop undirected traversal Crime—Vehicle with property filter "
            "on Vehicle, then ORDER BY date DESC LIMIT 1 to get the most recent.\n"
            "4. Construction:\n"
            "   - MATCH (x0:Crime)-[:INVOLVED_IN]-(x1:Vehicle WHERE x1.make = \"Toyota\")\n"
            "   - RETURN x0.date\n"
            "   - ORDER BY x0.date DESC LIMIT 1"
        ),
    },
    # 3. 3-node, entity_set, two-hop via shared friend
    {
        "question": (
            "[Person] = Alexander\n"
            "What is the latest crime connected to friends of those with the surname Alexander?"
        ),
        "cypher": (
            'MATCH (x0:Crime)-[:PARTY_TO]-(x1:Person)-[:KNOWS_SN]-(x2:Person WHERE x2.surname = "Alexander")\n'
            "RETURN x0\n"
            "ORDER BY x0.date DESC\n"
            "LIMIT 1"
        ),
        "reasoning": (
            "1. Sub-questions: (a) Find Persons with surname Alexander. "
            "(b) Find their friends via KNOWS_SN. "
            "(c) Find Crimes those friends are connected to via PARTY_TO. "
            "(d) Return the latest one.\n"
            "2. Schema elements: Person with surname property; Crime with date; "
            "KNOWS_SN (Person—Person friendship); PARTY_TO (Person—Crime).\n"
            "3. Pattern: Two-hop chain Crime—Person—Person with filter on terminal Person, "
            "undirected (friendship and party-to are symmetric in matching).\n"
            "4. Construction:\n"
            "   - MATCH (x0:Crime)-[:PARTY_TO]-(x1:Person)-[:KNOWS_SN]-(x2:Person WHERE x2.surname = \"Alexander\")\n"
            "   - RETURN x0 ORDER BY x0.date DESC LIMIT 1"
        ),
    },
    # 4. 3-node, attribute_set, two-hop with two filters
    {
        "question": (
            "[Person] = Philip\n"
            "[Crime] = Investigation complete; no suspect identified\n"
            "Can you provide the NHS numbers for people residing with Philip "
            "who were involved in incidents where investigation was completed without identifying a suspect?"
        ),
        "cypher": (
            'MATCH (x0:Person)-[:KNOWS_LW]-(x1:Person WHERE x1.name = "Philip")\n'
            'MATCH (x0:Person)-[:PARTY_TO]-(x2:Crime WHERE x2.last_outcome = "Investigation complete; no suspect identified")\n'
            "RETURN x0.nhs_no"
        ),
        "reasoning": (
            "1. Sub-questions: (a) Find Persons living with Philip via KNOWS_LW. "
            "(b) Filter those who are also PARTY_TO a Crime with the given last_outcome. "
            "(c) Return their nhs_no.\n"
            "2. Schema elements: Person with name and nhs_no; Crime with last_outcome; "
            "KNOWS_LW and PARTY_TO relationships.\n"
            "3. Pattern: Two MATCH clauses sharing the central Person x0, "
            "expressing the conjunction of two single-hop conditions.\n"
            "4. Construction:\n"
            "   - First MATCH establishes x0 lives with Philip\n"
            "   - Second MATCH establishes x0 is PARTY_TO a matching Crime\n"
            "   - RETURN x0.nhs_no for the answer"
        ),
    },
    # 5. 4-node, min, multi-hop
    {
        "question": (
            "[Person] = Hansen\n"
            "What is the date of the last call received by a resident living with a Hansen?"
        ),
        "cypher": (
            'MATCH (x0:PhoneCall)-[:CALLED]-(x1:Phone)-[:HAS_PHONE]-(x2:Person)-[:KNOWS_LW]-(x3:Person WHERE x3.surname = "Hansen")\n'
            "RETURN x0.call_date\n"
            "ORDER BY x0.call_date DESC\n"
            "LIMIT 1"
        ),
        "reasoning": (
            "1. Sub-questions: (a) Find a Person with surname Hansen. "
            "(b) Find a Person living with Hansen via KNOWS_LW. "
            "(c) Find their Phone via HAS_PHONE. "
            "(d) Find PhoneCalls received on that Phone via CALLED. "
            "(e) Return the latest call_date.\n"
            "2. Schema elements: Person (surname), Phone, PhoneCall (call_date); "
            "KNOWS_LW, HAS_PHONE, CALLED relationships.\n"
            "3. Pattern: Linear 4-hop chain PhoneCall—Phone—Person—Person, "
            "with undirected matches and a filter at the end of the chain.\n"
            "4. Construction:\n"
            "   - MATCH the full chain in one statement\n"
            "   - WHERE clause inline on the terminal Person (x3.surname = \"Hansen\")\n"
            "   - RETURN call_date ORDER BY DESC LIMIT 1"
        ),
    },
    # 6. 4-node, count, crime/officer/location
    {
        "question": (
            "[Officer] = Brister\n"
            "[Location] = 194 Garth Road\n"
            "How many crimes happened at 194 Garth Road and were investigated by an officer named Brister?"
        ),
        "cypher": (
            'MATCH (x0:Crime)-[:INVESTIGATED_BY]-(x2:Officer WHERE x2.surname = "Brister")\n'
            'MATCH (x0:Crime)-[:OCCURRED_AT]-(x1:Location WHERE x1.address = "194 Garth Road")\n'
            "RETURN COUNT(DISTINCT x0)"
        ),
        "reasoning": (
            "1. Sub-questions: (a) Find Crimes investigated by an Officer with surname Brister. "
            "(b) Restrict those Crimes to ones occurring at 194 Garth Road. "
            "(c) Count distinct Crimes.\n"
            "2. Schema elements: Crime, Officer (surname), Location (address); "
            "INVESTIGATED_BY (Crime—Officer), OCCURRED_AT (Crime—Location).\n"
            "3. Pattern: Two MATCH clauses sharing the central Crime x0, "
            "each adding one filter. Equivalent to a conjunction of two single-hop conditions.\n"
            "4. Construction:\n"
            "   - First MATCH: Crime—INVESTIGATED_BY—Officer with surname filter\n"
            "   - Second MATCH: same Crime—OCCURRED_AT—Location with address filter\n"
            "   - RETURN COUNT(DISTINCT x0) for the count"
        ),
    },
]


# Pole schema text (formatted as in eval_zograscope.py)
POLE_SCHEMA_TEXT = """Node properties:
- **Person** (People)
  - `name`: STRING
  - `surname`: STRING
  - `nhs_no`: STRING
  - `age`: INTEGER
- **Location** (Locations)
  - `address`: STRING
- **Phone** (Phone)
  - `phoneNo`: STRING
- **Email** (Email)
  - `email_address`: STRING
- **Officer** (Officers)
  - `name`: STRING
  - `surname`: STRING
  - `badge_no`: STRING
  - `rank`: STRING
- **PostCode** (PostCode)
  - `postcode`: STRING
- **Area** (Areas)
  - `areaCode`: STRING
- **PhoneCall** (Phone calls)
  - `call_time`: TIME
  - `call_date`: DATE
  - `call_duration`: INTEGER
- **Crime** (Crimes)
  - `date`: DATE
  - `type`: STRING
  - `last_outcome`: STRING
- **Object** (Criminal Objects)
  - `type`: STRING
- **Vehicle** (Vehicles)
  - `make`: STRING
  - `model`: STRING
  - `year`: INTEGER

Relationship types:
- **Person** -[:CURRENT_ADDRESS]-> **Location** (that lives in)
- **Person** -[:HAS_PHONE]-> **Phone** (which has)
- **Person** -[:HAS_EMAIL]-> **Email** (which has)
- **Person** -[:KNOWS_SN]-> **Person** (that is friends with)
- **Person** -[:KNOWS]-> **Person** (who knows)
- **Location** -[:HAS_POSTCODE]-> **PostCode** (which has)
- **PostCode** -[:POSTCODE_IN_AREA]-> **Area** (that is in)
- **Vehicle** -[:INVOLVED_IN]-> **Crime** (that is involved in)
- **PhoneCall** -[:CALLER]-> **Phone** (that were made to)
- **PhoneCall** -[:CALLED]-> **Phone** (that were received a)
- **Person** -[:KNOWS_PHONE]-> **Person** (knows the phone of)
- **Crime** -[:OCCURRED_AT]-> **Location** (that occurred at)
- **Crime** -[:INVESTIGATED_BY]-> **Officer** (that is investigated by)
- **Person** -[:PARTY_TO]-> **Crime** (which is involved in)
- **Person** -[:FAMILY_REL]-> **Person** (that has a family relation with)
- **Person** -[:KNOWS_LW]-> **Person** (that lives with)
- **Location** -[:LOCATION_IN_AREA]-> **Area** (that is included)"""
