import json, re, collections
import sys
s = open(sys.argv[1] if len(sys.argv) > 1 else 'runs/long_run/playback.html').read()
d = json.loads(re.search(r'const DATA = (\{.*?\});\n', s, re.S).group(1))
F, G = d['frames'], d['grid']
terr = d['terrain']
N = len(d['agents'])
print("ticks", len(F), "first", F[0]['time'], "last", F[-1]['time'], "agents", N)
acts = collections.Counter(); outs = collections.Counter(); pair = collections.Counter()
for f in F:
    for a in f['agents']:
        if not a['was_alive']: continue
        acts[a['action'].split('_')[0]] += 1; outs[a['act']] += 1; pair[(a['action'], a['act'])] += 1
print("\nINTENDED ACTION FAMILIES", acts.most_common())
print("OUTCOMES", outs.most_common())
# mismatches: intended non-rest action resolved to rest (masked/invalid) 
mism = {k: v for k, v in pair.items() if k[1] == 'rest' and k[0] != 'rest'}
print("ACTION->REST (invalid/converted)", mism)
# speech
spk = [(f['t'], i, a['msg'], a['heard_by']) for f in F for i, a in enumerate(f['agents']) if a['said']]
alive_ticks = sum(a['alive'] for f in F for a in f['agents'])
print("\nSPEECH", len(spk), "of", alive_ticks, "alive agent-ticks; heard by anyone:", sum(1 for x in spk if x[3]), "; distinct:", len(set(tuple(x[2]) for x in spk)))
# needs trajectories per agent
print("\nPER AGENT: start/end food water energy health weight; deaths; positions on water")
for i in range(N):
    tr = [f['agents'][i] for f in F]
    alive_end = tr[-1]['alive']
    on_water = sum(1 for a in tr if terr[a['r']*G+a['c']] == '3')
    eats = sum(1 for a in tr if a['act']=='eat'); drinks = sum(1 for a in tr if a['act']=='drink'); gathers = sum(1 for a in tr if a['act']=='gather')
    sel = sum(1 for a in tr if a['act']=='select'); places = sum(1 for a in tr if a['act']=='place'); strikes = sum(1 for a in tr if a['act'].startswith('strike'))
    print(f" #{i} age {tr[0]['age']}->{tr[-1]['age']} h{tr[0]['height']} w {tr[0]['weight']}->{tr[-1]['weight']} food {tr[0]['food']}->{tr[-1]['food']} water {tr[0]['water']}->{tr[-1]['water']} energy {tr[0]['energy']}->{tr[-1]['energy']} hp {tr[0]['health']}->{tr[-1]['health']} alive_end {alive_end} | eat {eats} drink {drinks} gather {gathers} select {sel} place {places} strike {strikes} on_water {on_water}")
# selections: what do they select, and do they eat non-food? (eat masked) 
sel_hands = collections.Counter(a['hand'] for f in F for a in f['agents'] if a['act']=='select')
print("\nSELECTED MATERIALS", sel_hands)
eat_hands = collections.Counter(a['hand'] for f in F for a in f['agents'] if a['act']=='eat')
print("EATEN MATERIALS", eat_hands)
place_hands = collections.Counter(a['hand'] for f in F for a in f['agents'] if a['act']=='place')
print("PLACED MATERIALS", place_hands)
# events
ev = collections.Counter(e['type'] for e in d['events'])
print("\nEVENTS", ev)
deaths = [e for e in d['events'] if e['type']=='death']
print("DEATHS", [(e['time'], e['agent'], e.get('cause')) for e in deaths])
# night ticks & lit
print("\nNIGHT TICKS", sum(f['night'] for f in F), "weather", collections.Counter(f['weather'] for f in F))
# time continuity
ts = [f['t'] for f in F]; print("time continuous:", ts == list(range(ts[0], ts[0]+len(ts))))
# age advance per tick
a0 = F[0]['agents'][0]['age']; a1 = F[-1]['agents'][0]['age']; print("age advance over 2 days:", round(a1-a0,2), "bio-years")
# idle: rest share by day vs night
rest_day = sum(1 for f in F for a in f['agents'] if a['was_alive'] and a['act']=='rest' and not f['night'])
rest_night = sum(1 for f in F for a in f['agents'] if a['was_alive'] and a['act']=='rest' and f['night'])
print("rest day/night", rest_day, rest_night)
# moves: oscillation (back-and-forth) rate
osc = 0; moves = 0
for i in range(N):
    prev = None
    for f in F:
        a = f['agents'][i]
        if a['act']=='move':
            moves += 1
            if prev and prev == opp(a['action']) if False else False: pass
print("plants mean per cell start/end", round(sum(int(c) for c in F[0]['plants'])/G/G,2), round(sum(int(c) for c in F[-1]['plants'])/G/G,2))
# heard_by plausibility: agents within 5 cells
far = 0
for f in F:
    for i,a in enumerate(f['agents']):
        for j in a['heard_by']:
            b = f['agents'][j]
            if max(abs(a['r']-b['r']), abs(a['c']-b['c'])) > 5: far += 1
print("heard_by beyond radius:", far)
# distances between agents (contact)
import itertools
dists = [max(abs(f['agents'][i]['r']-f['agents'][j]['r']), abs(f['agents'][i]['c']-f['agents'][j]['c'])) for f in F for i,j in itertools.combinations(range(N),2) if f['agents'][i]['alive'] and f['agents'][j]['alive']]
print("pair distance: mean", round(sum(dists)/len(dists),1), "share within 5:", round(sum(x<=5 for x in dists)/len(dists),3))
