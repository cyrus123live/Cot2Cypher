"""Nine hand-crafted few-shot exemplars covering diverse schema formats and query patterns.

Coverage matrix:
| # | Schema Format | Query Pattern            | Source                   |
|---|--------------|--------------------------|--------------------------|
| 1 | Minimal      | Single-hop + WHERE       | functional_cypher        |
| 2 | Minimal      | Aggregation + ORDER/LIMIT| functional_cypher        |
| 3 | Minimal      | Variable-length path     | functional_cypher        |
| 4 | Verbose      | Multi-hop + WITH pipeline| synthetic_gpt4o          |
| 5 | Verbose      | Single-hop + range WHERE | synthetic_claudeopus     |
| 6 | Verbose      | Multi-hop + WHERE filter | synthetic_claudeopus     |
| 7 | Verbose      | OPTIONAL MATCH / complex | synthetic_gpt4turbo      |
| 8 | JSON         | Simple COUNT             | text2cypher2023_train    |
| 9 | JSON         | Aggregation + WITH       | text2cypher2023_train    |
"""

EXEMPLARS = [
    # 1. Minimal schema, single-hop + WHERE (functional_cypher, idx 4)
    {
        "schema": (
            "Graph schema: Relevant node labels and their properties (with datatypes) are:\n"
            "Author {first_name: STRING}"
        ),
        "question": "Find the Author for which first_name starts with Jea!",
        "cypher": "MATCH (n:Author) WHERE n.first_name STARTS WITH 'Jea' RETURN n",
        "reasoning": (
            "1. The question asks to find Author nodes whose first_name starts with \"Jea\".\n"
            "2. Schema elements: Node label Author with property first_name (STRING).\n"
            "3. Pattern: Single node match with a string prefix filter — use STARTS WITH.\n"
            "4. Construction:\n"
            "   - MATCH (n:Author) to select all Author nodes\n"
            "   - WHERE n.first_name STARTS WITH 'Jea' to filter by prefix\n"
            "   - RETURN n to return matching authors"
        ),
    },
    # 2. Minimal schema, aggregation + ORDER BY + LIMIT (functional_cypher, idx 3)
    {
        "schema": (
            "Graph schema: Relevant node labels and their properties (with datatypes) are:\n"
            "Article {abstract: STRING}\n"
            "Keyword {}\n\n"
            "Relevant relationships are:\n"
            "{'start': Article, 'type': HAS_KEY, 'end': Keyword }"
        ),
        "question": (
            "For each Article find its abstract and the count of Keyword linked via HAS_KEY, "
            "and retrieve seven results in desc order of the counts!"
        ),
        "cypher": (
            "MATCH (n:Article) -[:HAS_KEY]->(m:Keyword) "
            "WITH DISTINCT n, m "
            "RETURN n.abstract AS abstract, count(m) AS count "
            "ORDER BY count DESC LIMIT 7"
        ),
        "reasoning": (
            "1. Sub-questions: (a) Which Articles have Keywords linked via HAS_KEY? "
            "(b) What is each Article's abstract? (c) How many Keywords per Article? "
            "(d) Return the top 7 by count descending.\n"
            "2. Schema elements: Node Article (abstract property), Node Keyword (no properties), "
            "Relationship HAS_KEY from Article to Keyword.\n"
            "3. Pattern: Single-hop traversal Article→HAS_KEY→Keyword with aggregation (COUNT), ordering, and limit.\n"
            "4. Construction:\n"
            "   - MATCH (n:Article)-[:HAS_KEY]->(m:Keyword) to traverse the relationship\n"
            "   - WITH DISTINCT n, m to deduplicate before counting\n"
            "   - RETURN n.abstract AS abstract, count(m) AS count for the aggregation\n"
            "   - ORDER BY count DESC LIMIT 7 to get top 7"
        ),
    },
    # 3. Minimal schema, variable-length path (functional_cypher, idx 6)
    {
        "schema": (
            "Graph schema: Relevant node labels and their properties (with datatypes) are:\n"
            "Article {title: STRING}"
        ),
        "question": (
            "List nodes that are 3 hops away from Article for which "
            "title=Maslov class and minimality in Calabi-Yau manifolds!"
        ),
        "cypher": (
            "MATCH (a:Article{title:'Maslov class and minimality in Calabi-Yau manifolds'})"
            "-[*3]->(n) RETURN labels(n) AS FarNodes"
        ),
        "reasoning": (
            "1. Sub-questions: (a) Which Article has the given title? "
            "(b) What nodes are exactly 3 hops away from it?\n"
            "2. Schema elements: Node Article with property title (STRING). "
            "No specific relationships listed, so we use a variable-length pattern.\n"
            "3. Pattern: Variable-length path of exactly 3 hops from a specific Article node.\n"
            "4. Construction:\n"
            "   - MATCH (a:Article{title:'Maslov class and minimality in Calabi-Yau manifolds'}) "
            "to find the starting node by inline property match\n"
            "   - -[*3]->(n) for exactly 3 hops in any relationship type\n"
            "   - RETURN labels(n) AS FarNodes to get the node labels at the end of the path"
        ),
    },
    # 4. Verbose schema, multi-hop + WITH pipeline (synthetic_gpt4o, idx 0)
    {
        "schema": (
            "Node properties:\n"
            "- **Country**\n"
            "  - `location`: POINT \n"
            "  - `code`: STRING Example: \"AFG\"\n"
            "  - `name`: STRING Example: \"Afghanistan\"\n"
            "  - `tld`: STRING Example: \"AF\"\n"
            "- **Filing**\n"
            "  - `begin`: DATE_TIME Min: 2000-02-08T00:00:00Z, Max: 2017-09-05T00:00:00Z\n"
            "  - `end`: DATE_TIME Min: 2000-02-08T00:00:00Z, Max: 2017-11-03T00:00:00Z\n"
            "  - `sar_id`: STRING Example: \"3297\"\n"
            "  - `amount`: INTEGER Min: 1.18, Max: 2721000000\n"
            "  - `number`: INTEGER Min: 1, Max: 174\n"
            "- **Entity**\n"
            "  - `id`: STRING Example: \"the-bank-of-new-york-mellon-corp\"\n"
            "  - `name`: STRING Example: \"The Bank of New York Mellon Corp.\"\n"
            "  - `country`: STRING Example: \"CHN\"\n"
            "Relationship properties:\n\n"
            "The relationships:\n"
            "(:Filing)-[:BENEFITS]->(:Entity)\n"
            "(:Filing)-[:CONCERNS]->(:Entity)\n"
            "(:Filing)-[:ORIGINATOR]->(:Entity)\n"
            "(:Entity)-[:FILED]->(:Filing)\n"
            "(:Entity)-[:COUNTRY]->(:Country)"
        ),
        "question": "Which 3 countries have the most entities linked as beneficiaries in filings?",
        "cypher": (
            "MATCH (f:Filing)-[:BENEFITS]->(e:Entity)-[:COUNTRY]->(c:Country) "
            "WITH c.name AS country, COUNT(e) AS entityCount "
            "ORDER BY entityCount DESC LIMIT 3 "
            "RETURN country, entityCount"
        ),
        "reasoning": (
            "1. Sub-questions: (a) Which entities are beneficiaries of filings? "
            "(b) What country is each entity in? (c) Which 3 countries have the most such entities?\n"
            "2. Schema elements: Filing→BENEFITS→Entity (beneficiary link), "
            "Entity→COUNTRY→Country (country link), Country.name for the country name.\n"
            "3. Pattern: Two-hop traversal Filing→BENEFITS→Entity→COUNTRY→Country, "
            "then aggregate by country with COUNT, ORDER BY DESC, and LIMIT 3.\n"
            "4. Construction:\n"
            "   - MATCH (f:Filing)-[:BENEFITS]->(e:Entity)-[:COUNTRY]->(c:Country) "
            "for the two-hop traversal\n"
            "   - WITH c.name AS country, COUNT(e) AS entityCount to aggregate per country\n"
            "   - ORDER BY entityCount DESC LIMIT 3 to get top 3\n"
            "   - RETURN country, entityCount"
        ),
    },
    # 5. Verbose schema, single-hop + range WHERE (synthetic_claudeopus, idx 7)
    {
        "schema": (
            "Node properties:\n"
            "- **Movie**\n"
            "  - `title`: STRING Example: \"The Matrix\"\n"
            "  - `votes`: INTEGER Min: 1, Max: 5259\n"
            "  - `tagline`: STRING Example: \"Welcome to the Real World\"\n"
            "  - `released`: INTEGER Min: 1975, Max: 2012\n"
            "- **Person**\n"
            "  - `born`: INTEGER Min: 1929, Max: 1996\n"
            "  - `name`: STRING Example: \"Keanu Reeves\"\n"
            "Relationship properties:\n"
            "- **ACTED_IN**\n"
            "  - `roles: LIST` Min Size: 1, Max Size: 6\n"
            "- **REVIEWED**\n"
            "  - `summary: STRING`\n"
            "  - `rating: INTEGER` Min: 45, Max: 100\n"
            "The relationships:\n"
            "(:Person)-[:ACTED_IN]->(:Movie)\n"
            "(:Person)-[:DIRECTED]->(:Movie)\n"
            "(:Person)-[:PRODUCED]->(:Movie)\n"
            "(:Person)-[:WROTE]->(:Movie)\n"
            "(:Person)-[:FOLLOWS]->(:Person)\n"
            "(:Person)-[:REVIEWED]->(:Movie)"
        ),
        "question": "Which movies released between 1990 and 2000 have more than 5000 votes?",
        "cypher": (
            "MATCH (m:Movie) "
            "WHERE m.released >= 1990 AND m.released <= 2000 AND m.votes > 5000 "
            "RETURN m.title, m.released, m.votes"
        ),
        "reasoning": (
            "1. Sub-questions: (a) Which movies were released between 1990 and 2000? "
            "(b) Of those, which have more than 5000 votes?\n"
            "2. Schema elements: Node Movie with properties released (INTEGER) and votes (INTEGER). "
            "Also title (STRING) for the output.\n"
            "3. Pattern: Single node match with multiple property filters (range on released, threshold on votes).\n"
            "4. Construction:\n"
            "   - MATCH (m:Movie) to select all movies\n"
            "   - WHERE m.released >= 1990 AND m.released <= 2000 for the date range\n"
            "   - AND m.votes > 5000 for the vote threshold\n"
            "   - RETURN m.title, m.released, m.votes to show relevant properties"
        ),
    },
    # 6. Verbose schema, multi-hop + WHERE filter (synthetic_claudeopus, idx 30)
    {
        "schema": (
            "Node properties:\n"
            "- **Article**\n"
            "  - `id`: STRING Example: \"ART176872705964\"\n"
            "  - `sentiment`: FLOAT Example: \"0.856\"\n"
            "  - `author`: STRING\n"
            "  - `title`: STRING\n"
            "- **Organization**\n"
            "  - `name`: STRING Example: \"New Energy Group\"\n"
            "  - `isPublic`: BOOLEAN\n"
            "Relationship properties:\n\n"
            "The relationships:\n"
            "(:Article)-[:MENTIONS]->(:Organization)"
        ),
        "question": (
            "List the first 3 organizations that were mentioned in articles "
            "with a sentiment less than 0.5."
        ),
        "cypher": (
            "MATCH (a:Article)-[:MENTIONS]->(o:Organization) "
            "WHERE a.sentiment < 0.5 "
            "RETURN o.name LIMIT 3"
        ),
        "reasoning": (
            "1. Sub-questions: (a) Which articles have sentiment less than 0.5? "
            "(b) What organizations do those articles mention? (c) Return the first 3.\n"
            "2. Schema elements: Node Article (sentiment: FLOAT), "
            "Relationship MENTIONS from Article to Organization, "
            "Node Organization (name: STRING).\n"
            "3. Pattern: Single-hop traversal Article→MENTIONS→Organization with a WHERE filter "
            "on the source node's property, plus LIMIT.\n"
            "4. Construction:\n"
            "   - MATCH (a:Article)-[:MENTIONS]->(o:Organization) for the traversal\n"
            "   - WHERE a.sentiment < 0.5 to filter articles by sentiment\n"
            "   - RETURN o.name LIMIT 3 to get the first 3 organization names"
        ),
    },
    # 7. Verbose schema, OPTIONAL MATCH / complex pattern (synthetic_gpt4turbo, idx 9229)
    {
        "schema": (
            "Node properties:\n"
            "- **Person**\n"
            "  - `name`: STRING Example: \"Julie Spellman Sweet\"\n"
            "  - `summary`: STRING Example: \"CEO at Accenture\"\n"
            "- **Organization**\n"
            "  - `name`: STRING Example: \"New Energy Group\"\n"
            "  - `isPublic`: BOOLEAN\n"
            "Relationship properties:\n\n"
            "The relationships:\n"
            "(:Organization)-[:HAS_CEO]->(:Person)\n"
            "(:Organization)-[:HAS_BOARD_MEMBER]->(:Person)\n"
            "(:Organization)-[:HAS_INVESTOR]->(:Person)\n"
            "(:Organization)-[:HAS_SUBSIDIARY]->(:Organization)\n"
            "(:Organization)-[:HAS_SUPPLIER]->(:Organization)\n"
            "(:Organization)-[:HAS_INVESTOR]->(:Organization)\n"
            "(:Organization)-[:HAS_COMPETITOR]->(:Organization)"
        ),
        "question": (
            "What are the relationships of 'Julie Spellman Sweet' within the context "
            "of the organization she is associated with?"
        ),
        "cypher": (
            "MATCH (p:Person {name: \"Julie Spellman Sweet\"})"
            "-[:HAS_CEO|HAS_BOARD_MEMBER|HAS_INVESTOR]-(o:Organization) "
            "OPTIONAL MATCH (o)-[r:HAS_SUBSIDIARY|HAS_SUPPLIER|HAS_INVESTOR|HAS_COMPETITOR]->"
            "(relatedOrg:Organization) "
            "RETURN p, o, r, relatedOrg"
        ),
        "reasoning": (
            "1. Sub-questions: (a) Which organization is Julie Spellman Sweet associated with? "
            "(b) Through what role — CEO, board member, or investor? "
            "(c) What other organizations are related to that organization?\n"
            "2. Schema elements: Person (name property), Organization node. "
            "Person-to-Org relationships: HAS_CEO, HAS_BOARD_MEMBER, HAS_INVESTOR (all go Org→Person). "
            "Org-to-Org relationships: HAS_SUBSIDIARY, HAS_SUPPLIER, HAS_INVESTOR, HAS_COMPETITOR.\n"
            "3. Pattern: First, find the Person and their Organization via any of the person-role "
            "relationships (undirected since schema has Org→Person but we match from Person). "
            "Then OPTIONAL MATCH for related organizations.\n"
            "4. Construction:\n"
            "   - MATCH (p:Person {name: \"Julie Spellman Sweet\"})"
            "-[:HAS_CEO|HAS_BOARD_MEMBER|HAS_INVESTOR]-(o:Organization) "
            "using multi-type relationship and undirected pattern\n"
            "   - OPTIONAL MATCH (o)-[r:HAS_SUBSIDIARY|HAS_SUPPLIER|HAS_INVESTOR|HAS_COMPETITOR]->"
            "(relatedOrg:Organization) to find connected organizations (may not exist)\n"
            "   - RETURN p, o, r, relatedOrg"
        ),
    },
    # 8. JSON schema, simple COUNT (text2cypher2023_train, idx 20)
    {
        "schema": (
            '{"Driver": {"count": 12, "labels": [], "properties": '
            '{"Name": {"unique": false, "indexed": false, "type": "STRING", "existence": false}, '
            '"Age": {"unique": false, "indexed": false, "type": "INTEGER", "existence": false}, '
            '"Home_city": {"unique": false, "indexed": false, "type": "STRING", "existence": false}, '
            '"Driver_ID": {"unique": false, "indexed": false, "type": "INTEGER", "existence": false}}, '
            '"type": "node", "relationships": {"ATTENDS": {"count": 0, "direction": "out", '
            '"labels": ["School"], "properties": {}}}}}'
        ),
        "question": "How many drivers are there?",
        "cypher": "MATCH (d:Driver) RETURN COUNT(d)",
        "reasoning": (
            "1. The question asks for a count of all Driver nodes.\n"
            "2. Schema elements: Node label Driver (properties: Name, Age, Home_city, Driver_ID). "
            "No filtering needed.\n"
            "3. Pattern: Simple node match with COUNT aggregation.\n"
            "4. Construction:\n"
            "   - MATCH (d:Driver) to select all Driver nodes\n"
            "   - RETURN COUNT(d) to count them"
        ),
    },
    # 9. JSON schema, aggregation + WITH pipeline (text2cypher2023_train, idx 271)
    {
        "schema": (
            '{"Pilot": {"count": 5, "labels": [], "properties": '
            '{"Age": {"type": "INTEGER"}, '
            '"Pilot_name": {"type": "STRING"}, '
            '"Rank": {"type": "INTEGER"}, '
            '"Nationality": {"type": "STRING"}, '
            '"Team": {"type": "STRING"}, '
            '"Join_Year": {"type": "INTEGER"}}, '
            '"type": "node", "relationships": {"OPERATED_BY": {"direction": "in", '
            '"labels": ["Aircraft"], "properties": {"Date": {"type": "STRING"}}}}}}'
        ),
        "question": "Show the most common nationality of pilots.",
        "cypher": (
            "MATCH (p:Pilot) "
            "WITH p.Nationality AS Nationality, COUNT(p) as count "
            "ORDER BY count DESC "
            "RETURN Nationality LIMIT 1"
        ),
        "reasoning": (
            "1. Sub-questions: (a) What nationalities do pilots have? "
            "(b) How many pilots per nationality? (c) Which nationality is most common?\n"
            "2. Schema elements: Node Pilot with property Nationality (STRING).\n"
            "3. Pattern: Single node match, group by Nationality with COUNT, "
            "order descending, take top 1.\n"
            "4. Construction:\n"
            "   - MATCH (p:Pilot) to select all pilots\n"
            "   - WITH p.Nationality AS Nationality, COUNT(p) as count to aggregate by nationality\n"
            "   - ORDER BY count DESC to sort most common first\n"
            "   - RETURN Nationality LIMIT 1 to get the single most common nationality"
        ),
    },
]
