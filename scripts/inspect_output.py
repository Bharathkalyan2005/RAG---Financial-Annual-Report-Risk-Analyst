import json
with open('data/processed/AAPL_2025.json', encoding='utf-8') as f:
    doc = json.load(f)

company = doc['company']
year = doc['year']
sections = doc['sections']
print('Company:', company, ' Year:', year)
print('Total sections:', len(sections))
print()
for s in sections:
    nt = len(s['tables'])
    tl = len(s['text'])
    name = s['section_name'][:68]
    pid = s['position_id']
    print(f'  [{pid:02d}] {name:<68}  text={tl:>6}c  tables={nt}')
